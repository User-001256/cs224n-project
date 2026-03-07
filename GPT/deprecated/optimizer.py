from typing import Callable, Iterable, Optional, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)
        

    def step(self, closure: Optional[Callable] = None):
        loss = None # Never used?
        if closure is not None:
            loss = closure()
            
        # No need to track gradients within the optimizer step
        with torch.no_grad():
            for group in self.param_groups:
                # Access hyperparameters from the parameter group:
                alpha = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                decay = group["weight_decay"]
                
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                    # State should be stored in this dictionary.
                    state = self.state[p]
                    
                    # Initialize the optimizer states if necessary
                    if "t" not in state: # Time step
                        state["t"] = 0
                    if "m" not in state: # First moment
                        state["m"] = torch.zeros_like(p)
                    if "v" not in state: # Second moment
                        state["v"] = torch.zeros_like(p)
                        
                    # Now perform the AdamW update algorithm:
                    # Update time step:
                    state["t"] += 1
                    
                    # Update first and second moments (use more efficient in-place methods)
                    state["m"].mul_(beta1).add_((1-beta1)* grad)                 # m <- beta1 * m + (1-beta1) * g
                    state["v"].mul_(beta2).addcmul_(grad, grad, value = 1-beta2) # v <- beta2 * v + (1-beta2) * g ** 2

                    # Efficient version for parameter update (again, use in-place methods)
                    at = alpha * math.sqrt(1 - beta2 ** state["t"]) / (1 - beta1 ** state["t"])
                    p.addcdiv_(state["m"], (torch.sqrt(state["v"]) + eps), value = -at)
                        # p <- p - at * m / (sqrt(v) + eps)
                
                    # Now perform parameter weight decay. Incorporating learning rate.
                    p.mul_(1 - alpha * decay)

        return loss
