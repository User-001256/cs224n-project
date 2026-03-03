"""
This script contains generation and logit processors for generation tasks.
Implementations in this script is inspired by the HuggingFace script:
https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
"""

import torch
import torch.nn.functional as F

class TemperatureScaling(object):
    """
    Scale the input logits by 1/T in temperature sampling.
    
    - T > 1 : Flattens distribution, increases generation diversity.
    - T < 1 : Decreases randomness, shifting to most likely token
    """
    def __init__(self, t:float):
        assert t > 0
        self.temp = t

    def __call__(self, logits:torch.Tensor):
        return logits / self.temp


class TopPFilter(object):
    """
    Performs top-p / nucleus sampling. Restrict the search space to top tokens
    with cumulative probabilities <= p. Can be used with top-k and temperature sampling.
    
    Args:
        p: Active when set to < 1. Only the most probable tokens with cumulative
            probability <= p will be retailed.
        min_count: Minimum number of tokens that will be retained in our search set.
    """
    def __init__(self, p:float, min_count = 1):
        assert p > 0.0 and p <= 1.0
        self.p = p
        self.keep = min_count
    
    def __call__(self, logits:torch.Tensor):
        # Sort the input logits
        sorted_logits, sorted_idxs = torch.sort(logits, descending=True)
        # Apply softmax and calculate cumulative sum
        cumulative_probs = F.softmax(sorted_logits, dim = -1).cumsum(dim = -1)
        
        # Remove tokens with cumulative probs that exceeds the threshold
        sorted_removal = cumulative_probs > self.p
        # Keep at least min_count tokens within the search space
        sorted_removal[..., :self.keep] = 0
        
        # GPU Scatter Magic...
        # removal_idxs[sorted_idxs[i]] <- sorted_removal[i]
        removal_idxs = torch.scatter(sorted_removal, dim = 1, index = sorted_idxs, src = sorted_removal)
        return torch.masked_fill(logits, removal_idxs, -torch.inf)
    
class TopKFilter(object):
    """
    Performs top-k / nucleus sampling. Restrict the search space to top tokens
    with the largest probability mass
    
    Args:
        k: Top token count to sample from. Must be >= 1. (Usually 5 - 50).
    """
    def __init__(self, k: int):
        assert k >= 1
        self.k = k
        
    def __call__(self, logits:torch.Tensor):
        # Compute the filter threshold
        threshold = min(self.k, logits.shape[-1])
        
        # Remove all tokens not in the top k. Unsqueeze to keep dim.
        threshold_value = torch.topk(logits, threshold, dim=-1).values[..., -1:]
        mask = logits < threshold_value
        return torch.masked_fill(logits, mask, -torch.inf)
    
class MinPFilter(object):
    """
    Performs min-p filtering. Keep all tokens that are above a minimum probability
    that is scaled by the probability of the most likely token. Usually, p is in (0.01, 0.1).
    
    Intuitively, more tokens will be filtered out if the probability mass of the
    most likely token is higher
    
    This is used as an alternative to TopPFilter.
    """
    def __init__(self, p:float):
        assert p > 0.0 and p <= 1.0
        self.p = p
    
    def __call__(self, logits:torch.Tensor):
        # Compute the probabilities by applying softmax
        probs = F.softmax(logits, dim = -1)
        top = torch.amax(probs, dim = -1, keepdim=True)
        # Compute the probability thresholds
        threshold = self.p * top
        # mask out logits below the threshold
        return torch.masked_fill(logits, probs < threshold, -torch.inf)
    
def sample_from_logits(logit:torch.Tensor):
    """
    Sample a token from a multinomial distribution where the probability weight is defined by the logits.
    
    Hint: Recommended logit processing workflow: Temperature -> top_k -> top_p / min_p
    """
    # First convert the logits into probabilities via softmax
    probs = F.softmax(logit, dim = -1)
    # Then draw sample...
    return torch.multinomial(probs, num_samples=1)
          