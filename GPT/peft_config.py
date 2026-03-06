"""
This file contains all the peft configuration files
"""

# Comment out keys if you don't want to apply this peft.
CLASSIFIER_PEFT = {
    "ia3" : None,
    "adapter" : {"adapter_dim" : 64},
    "lora": {"target_modules": ["query", "value"], "rank": 8, "dropout": 0.05, "alpha": 32},
}

PARAPHRASE_PEFT = {
    "ia3" : None,
    "adapter" : {"adapter_dim" : 64},
    "lora": {"target_modules": ["query", "value"], "rank": 8, "dropout": 0.05, "alpha": 32},
}

SONNET_PEFT = {
    "ia3" : None,
    "adapter" : {"adapter_dim" : 64},
    "lora": {"target_modules": ["query", "value"], "rank": 8, "dropout": 0.05, "alpha": 32},
}