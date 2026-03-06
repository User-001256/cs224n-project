"""
This module contains scripts for constructing a GPT2 Based Classifier
"""
import torch
import torch.nn as nn
from typing import Optional, Dict
from types import SimpleNamespace
from models.gpt2 import GPT2Model

def construct_prediction_head(dim_in, dim_out, kind = "linear", dim_act = None):
    """
    Constrcts a prediction head for the GPT2SentimentClassifier
    """
    if dim_act is None:
        dim_act = 4 * dim_in
    # Linear Prediction Head
    if kind == "linear":
        return nn.Linear(dim_in, dim_out)
    if kind == "mlp":
        return torch.nn.Sequential(
        nn.Linear(dim_in, dim_in * 3),
        nn.GELU(),
        nn.Linear(dim_in * 3, dim_out)
        )
    else:
        raise RuntimeError(f"Classifier {kind} is not recognized.")
    
class GPT2Classifier(nn.Module):
    def __init__(self, cls_config:SimpleNamespace, gpt_backbone : Optional[GPT2Model] = None):
        """
        GPT2 Based Classifier

        Args:
            cls_config: Configuration for the classifier. 
                Must contain "hidden_dropout_prob", "hidden_size", and "num_labels"
                Optionally: "cls_kind", "mlp_act_dim"
            gpt_backbone: GPT Backbone Model. Defaults to gpt2-base.
        """
        super().__init__()
        if gpt_backbone is None:
            self.gpt = GPT2Model.from_pretrained()
            # Freeze all backbone parameters
            for p in self.gpt.parameters():
                p.requires_grad = False
        else:
            self.gpt = gpt_backbone
            
        # Initialize prediction head
        self.cls_dropout = nn.Dropout(cls_config.hidden_dropout_prob)
        act_dim, kind = None, "linear"
        if hasattr(cls_config, "clf_kind") :
            kind = cls_config.clf_kind
        if hasattr(cls_config, "mlp_act_dim"):
            act_dim = cls_config.mlp_act_dim
        self.cls_prediction_head = construct_prediction_head(
            cls_config.hidden_size,
            cls_config.num_labels,
            kind,
            act_dim
        )
        # Log the classifier config file:
        self.cls_config = cls_config
        
    def forward(self, input_ids, attention_mask):
        gpt_out = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        return self.cls_prediction_head(self.cls_dropout(gpt_out["last_token"]))
        
        
        
        