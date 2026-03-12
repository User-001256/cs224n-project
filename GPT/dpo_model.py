import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT2DPOWrapper(nn.Module):
    def __init__(self, policy_model, beta=0.1):
        super().__init__()
        self.policy = policy_model
        self.reference = copy.deepcopy(policy_model)
        self.reference.eval()
        for p in self.reference.parameters():
            p.requires_grad = False
        self.beta = beta

    def token_logits(self, input_ids, attention_mask):
        outputs = self.policy(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs["last_hidden_state"]
        logits = self.policy.hidden_state_to_token(hidden_states)
        return logits

    def sequence_logprob(self, model, input_ids, attention_mask, completion_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs["last_hidden_state"]
        logits = model.hidden_state_to_token(hidden_states)

        log_probs = F.log_softmax(logits, dim=-1)

        shifted_log_probs = log_probs[:, :-1, :]
        shifted_labels = input_ids[:, 1:]
        shifted_completion_mask = completion_mask[:, 1:]

        token_log_probs = shifted_log_probs.gather(
            dim=-1,
            index=shifted_labels.unsqueeze(-1)
        ).squeeze(-1)

        seq_log_probs = (token_log_probs * shifted_completion_mask).sum(dim=-1)
        return seq_log_probs

    def preference_logprobs(
        self,
        chosen_input_ids,
        chosen_attention_mask,
        chosen_completion_mask,
        rejected_input_ids,
        rejected_attention_mask,
        rejected_completion_mask,
    ):
        policy_chosen = self.sequence_logprob(
            self.policy,
            chosen_input_ids,
            chosen_attention_mask,
            chosen_completion_mask,
        )
        policy_rejected = self.sequence_logprob(
            self.policy,
            rejected_input_ids,
            rejected_attention_mask,
            rejected_completion_mask,
        )

        with torch.no_grad():
            ref_chosen = self.sequence_logprob(
                self.reference,
                chosen_input_ids,
                chosen_attention_mask,
                chosen_completion_mask,
            )
            ref_rejected = self.sequence_logprob(
                self.reference,
                rejected_input_ids,
                rejected_attention_mask,
                rejected_completion_mask,
            )

        return policy_chosen, policy_rejected, ref_chosen, ref_rejected

    def dpo_loss(
        self,
        chosen_input_ids,
        chosen_attention_mask,
        chosen_completion_mask,
        rejected_input_ids,
        rejected_attention_mask,
        rejected_completion_mask,
    ):
        policy_chosen, policy_rejected, ref_chosen, ref_rejected = self.preference_logprobs(
            chosen_input_ids,
            chosen_attention_mask,
            chosen_completion_mask,
            rejected_input_ids,
            rejected_attention_mask,
            rejected_completion_mask,
        )

        chosen_adv = policy_chosen - ref_chosen
        rejected_adv = policy_rejected - ref_rejected
        logits = self.beta * (chosen_adv - rejected_adv)
        loss = -F.logsigmoid(logits).mean()

        return loss
