"""
This file contains all the peft configuration files
"""

# Comment out keys if you don't want to apply this peft.
CLASSIFIER_PEFT = {
    # "ia3" : dict(),
    "adapter" : {"adapter_dim" : 64, "target_modules": ["ffn"], "gated": True},
    "lora": {"target_modules": ["query", "value"], "rank": 8, "dropout": 0.05, "alpha": 32, "gated": True},
    "prefix": {"num_prefixes": 10}
}

PARAPHRASE_PEFT = {
    "ia3" : dict(),
    "adapter" : {"adapter_dim" : 64},
    "lora": {"target_modules": ["query", "value"], "rank": 8, "dropout": 0.05, "alpha": 32},
}

SONNET_PEFT = {
    "ia3" : dict(),
    "adapter" : {"adapter_dim" : 16, "gated": True},
    "lora": {"target_modules": ["query", "value"], "rank": 4, "dropout": 0.05, "alpha": 16, "gated": True},
    "prefix": {
            "num_prefixes": 10,
            "bottleneck_dim": 64,
            "intermediate_dim": 512,
        },
}