import torch
import numpy as np
import random
from torch.optim import SGD, Adam, AdamW
from models.gpt2_adapt import *
from models.gpt2 import GPT2Model
from models.gpt2_classifier import GPT2Classifier, SimpleNamespace

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

def save_fine_tune_checkpoint(out_path:str, 
                              model: Union[GPT2Model, GPT2Classifier],
                              model_config:SimpleNamespace, 
                              optimizer = None, scheduler = None):
    """
    Saving a fine-tune model checkpoont.

    Args:
        out_path: Output file path
        model: Pytorch Model
        model_config: Model Config. Should contain the following:
            "gpt_kind"          : Pre-trained GPT Size
            "fine_tune_mode"    : Full Model or PEFT Fine Tune mode
            "peft_config"       : Config dict for PEFT (Optional)
            "clf_config"        : Classifier Config (Optional)
        optimizer: Optimizer State. Defaults to None.
        scheduler: Scheduler State. Defaults to None.
    """
    sd = model.state_dict()
    if not (hasattr(model_config, "gpt_kind") and hasattr(model_config, "fine_tune_mode")):
        print("Model Saving Failed. Reason: Invalid Model Config.")
        return
    # If not full fine tune, only save the trainable parameters.
    if model_config.fine_tune_mode != "full-model":
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        sd = {k: sd[k] for k in trainable}
    # Optionally Save scheduler and optimizer states
    ckpt = {"model": sd, "model_config": model_config}
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    
    try:
        torch.save(ckpt, out_path)
    except Exception:
        print(f"Failed to save model checkpoint to {out_path}.")
    else:
        print(f"Model checkpoint saved to {out_path}.")

def load_fine_tune_checkpoint(load_path, device = torch.device("cpu"), model = None, optimizer = None, scheduler = None):
    ckpt = torch.load(load_path, weights_only=False, map_location=device)
    model_config = ckpt["model_config"]
    if model is None:
        print("Model not provided. Inferring from checkpoint config.")
        if not (hasattr(model_config, "gpt_kind") and hasattr(model_config, "fine_tune_mode")):
            raise RuntimeError("Cannot infer model when loading checkpoint. Reason: Corrupted Config.")
        model = GPT2Model.from_pretrained(model_config.gpt_kind)
        # Need to initialize peft structure for all partial checkpoints
        if model_config.fine_tune_mode == "peft":
            if not hasattr(model_config, "peft_config"):
                raise RuntimeError("Cannot infer PEFT structure when loading checkpoint. Reason: Corrupted Config.")
            apply_gpt_peft(model, model_config.peft_config)
        # If the checkpoint is for a classifier:
        if hasattr(model_config, "clf_config"):
            model = GPT2Classifier(model_config.clf_config, model)
    # Now we are finally ready to load our weights into the model
    if model_config.fine_tune_mode != "full-model":
        missing_keys, unexpected_keys = model.load_state_dict(ckpt["model"], strict=False)
        print(f"Successfully loaded partial PEFT checkpoint. Ignored {len(missing_keys)} frozen parameters, and skipped {len(unexpected_keys)} unrecognized parameters.")
    else:
        model.load_state_dict(ckpt["model"])
        print("Successfully loaded model checkpoint.")
        
    # Load optimizer and scheduler states if required:
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    
    # Return the loaded model
    return model
        
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

def summarize_peft(model:nn.Module):
    """
    Summarizes PEFT model trainable parameters
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Model Class: {type(model).__name__}")
    print(
        f"Trainable Parameters: {trainable_params:,d} || "
        f"Total Parameters: {all_param:,d} || "
        f"Percent Trainable: {100 * trainable_params / all_param:.5f}%"
        
    )
    
# Early Stopping Scheduler
class EarlyStoppingScheduler(object):
    def __init__(self, patience = 5, min_delta = 1e-3):
        """
        Early Stopping Scheduler

        Args:
            patience: Number of no-progress steps before stopping. Defaults to 5.
            min_delta: Loss below the previous best loss to count as progress. Defaults to 1e-3.
        """
        self.patience = patience
        self.min_delta = min_delta
        
        self.no_progress_counter = 0
        self.best_loss = np.inf
        
        self.stopped = False
        
    def step(self, valid_loss):
        # Significant Progress
        if valid_loss < (self.best_loss - self.min_delta):
            # Reset Counter and Log Loss
            self.best_loss = valid_loss
            self.no_progress_counter = 0
        else:
            self.no_progress_counter += 1
            if self.no_progress_counter >= self.patience:
                self.stopped = True
            else:
                print(f"Early Stopping: {self.no_progress_counter} / {self.patience}.")