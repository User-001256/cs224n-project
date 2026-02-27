import torch
import numpy as np
import random

def select_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Selected device {device}.")
    return device

def save_model_checkpoint(out_path, model, optimizer, scheduler = None, config = None, args = None):
    """
    Saving pytorch model checkpoint

    Args:
        out_path: Output file path
        model: Pytorch Model
        optimizer: Optimizer State
        scheduler: Scheduler State. Defaults to None.
        config: Model config. Defaults to None.
        args: Model arguments. Defaults to None.
    """
    save = {"model": model, "optimizer": optimizer,
            "system_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.random.get_rng_state()}
    if scheduler is not None:
        save["scheduler"] = scheduler
    if config is not None:
        save["model_config"] = config
    if args is not None:
        save["args"] = args
    
    try:
        torch.save(save, out_path)
    except Exception:
        print(f"Failed to save model checkpoint to {out_path}.")
    else:
        print(f"Model checkpoint saved to {out_path}.")
        
def load_checkpoint(load_path, model, device = torch.device("cpu"),
                    optimizer = None, scheduler = None):
    """
    Loading an existing pyTorch model checkpoint.

    Args:
        load_path: Checkpoint path to load.
        model: pyTorch model
        device: Device to map data to. Defaults to torch.device("cpu").
        optimizer: Optimizer. Defaults to None.
        scheduler: Scheduler. Defaults to None.
    """
    ckpt = torch.load(load_path, weights_only=True, map_location=device)
    model = model.to(device)
    model.load_weights(ckpt["model"])
    out = {"model": model}
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        out["optimizer"] = optimizer
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
        out["scheduler"] = scheduler
    # Load other things
    for k in ckpt.keys():
        if k in {"model", "optimizer", "scheduler"}: continue
        out[k] = ckpt[k]