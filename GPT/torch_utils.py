import torch
import numpy as np
import random
from torch.optim import SGD, Adam, AdamW

def select_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Selected device {device}.")
    return device

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def set_optimizer(model_params, optim_class = "AdamW", **optim_kwargs):
    """
    Set up model optimizer for PyTorch Model

    Args:
        model_params: Parameters to Optimize
        optim_class: Optimizer Class. Supports ("SGD", "Adam", "AdamW") Defaults to "AdamW".
    
    Returns:
        Optimizer
    """
    if optim_class == "SGD":
        return SGD(model_params, **optim_kwargs)
    elif optim_class == "Adam":
        return Adam(model_params, **optim_kwargs)
    elif optim_class == "AdamW":
        return AdamW(model_params, **optim_kwargs)
    else:
        raise RuntimeError(f"Optimizer Class {optim_class} is not recognized. Please use 'SGD', 'Adam', or 'AdamW'.")


def save_model_checkpoint(out_path, model, config, optimizer = None, scheduler = None, args = None):
    """
    Saving pytorch model checkpoint

    Args:
        out_path: Output file path
        model: Pytorch Model
        config: PyTorch Model Configuration Datastructure
        optimizer: Optimizer State. Defaults to None
        scheduler: Scheduler State. Defaults to None.
        config: Model config. Defaults to None.
        args: Model arguments. Defaults to None.
    """
    save = {"model": model.state_dict(), "model_config": config,
            "system_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.random.get_rng_state()}
    if scheduler is not None:
        save["scheduler"] = scheduler.state_dict()
    if optimizer is not None:
        save["optimizer"] = optimizer.state_dict()
    if args is not None:
        save["args"] = args
    
    try:
        torch.save(save, out_path)
    except Exception:
        print(f"Failed to save model checkpoint to {out_path}.")
    else:
        print(f"Model checkpoint saved to {out_path}.")
        
def load_checkpoint(load_path, device = torch.device("cpu"), model = None,
                    optimizer = None, scheduler = None):
    """
    Loading an existing PyTorch model checkpoint.

    Args:
        load_path: Checkpoint path to load.
        device: Device to map data to. Defaults to torch.device("cpu").
        model: PyTorch model. Defaults to None.
        optimizer: Optimizer. Defaults to None.
        scheduler: Scheduler. Defaults to None.
        
    Returns:
        Returns the dictionary of loaded object. 
        If model, optimizer, or shceduler are supplied, loads states from the checkpoint.
    """
    ckpt = torch.load(load_path, weights_only=False, map_location=device)
    if model is not None and "model" in ckpt:
        model = model.to(device)
        model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt