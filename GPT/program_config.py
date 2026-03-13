"""
This file contains program parameters for classifier.py, paraphrase_detection.py,
sonnet_generation.py, and DPO sonnet fine-tuning.
"""
from types import SimpleNamespace

DATA_FOLDER = "../data/"

DATA_PATHS = SimpleNamespace(
    sst_train = DATA_FOLDER + "ids-sst-train.csv",
    sst_valid = DATA_FOLDER + "ids-sst-dev.csv",
    sst_test = DATA_FOLDER + "ids-sst-test-student.csv",
    imdb_train = DATA_FOLDER + "ids-cfimdb-train.csv",
    imdb_valid = DATA_FOLDER + "ids-cfimdb-dev.csv",
    imdb_test = DATA_FOLDER + "ids-cfimdb-test-student.csv",
    
    para_train = DATA_FOLDER + "quora-train.csv",
    para_valid = DATA_FOLDER + "quora-dev.csv",
    para_test = DATA_FOLDER + "quora-test-student.csv",
    
    sonnet_train = DATA_FOLDER + "sonnets_sft_train_split.txt",
    sonnet_test = DATA_FOLDER + "sonnets_held_out.txt",
    sonnet_dev = DATA_FOLDER + "sonnets_held_out_dev.txt"
)

CLASSIFIER_ARGS = SimpleNamespace(
    seed = 42,                              # Reproduceability Random Seed
    lr = 1e-5,                              # learning rate
    weight_decay = 5e-2,                    # Weight Decay Regularization
    epochs = 1,                             # Training Epochs
    batch_size = 64,                        # Training Batch Size
    num_workers = 0,                        # Dataloader parallelism
    gpt_kind = "gpt2",                      # Pre-trained Model Kind
    
    prediction_head = "linear",             # Prediction Head Construction
    fine_tune_mode = "peft",                # Fine-tune Mode: Use full-model, peft, or last-layer
    hidden_dropout_prob = 0.0               # Dropout probability for prediciton head
)

PARAPHRASE_ARGS = SimpleNamespace(
    seed = 42,                              # Reproduceability Random Seed
    lr = 1e-5,                              # learning rate
    weight_decay = 5e-2,                    # Weight Decay Regularization
    epochs = 1,                             # Training Epochs
    batch_size = 8,                         # Training Batch Size
    num_workers = 0,                        # Dataloader parallelism
    gpt_kind = "gpt2",                      # Pre-trained Model Kind
    fine_tune_mode = "peft",                # Fine-tune Mode: Use full-model or peft
)

SONNET_ARGS = SimpleNamespace(
    seed = 42,                              # Reproducibility Random Seed
    lr = 1e-5,                              # learning rate
    weight_decay = 5e-2,                    # Weight Decay Regularization
    epochs = 1,                             # Training Epochs
    batch_size = 8,                         # Training Batch Size
    num_workers = 4,                        # Dataloader parallelism
    gpt_kind = "gpt2-medium",               # Pre-trained Model Kind
    fine_tune_mode = "peft",                # Fine-tune Mode: Use full-model or peft
    use_sonnet_dev = True,                  # If True, use sonnet_dev instead of sonnet_test for generation
    top_p = 0.9,                            # Top P Sampling
    temperature = 1.2,                      # Temperature Sampling
    top_k = 50,                             # Top K Sampling
    preview_valid = False,                   # Preview Validation Sets
    sonnet_out = "./predictions/generated_sonnets_dev.txt"                   # Sonnet output path
)

DPO_ARGS = SimpleNamespace(
    seed = 42,                              # Reproducibility Random Seed
    prefs_path = "../data/sonnet_dpo_prefs.jsonl",  # Preference pairs built from held-out prompts
    checkpoint = "7_8-3e-05-sonnet.pt",  # SFT checkpoint used to initialize DPO
    output_dir = "./DPO/checkpoints",       # Where to save DPO checkpoints
    batch_size = 4,                         # DPO batch size
    num_workers = 4,                        # Dataloader parallelism
    epochs = 3,                             # DPO training epochs
    lr = 1e-6,                              # DPO learning rate
    weight_decay = 5e-2,                    # Weight Decay Regularization
    beta = 0.1,                             # DPO preference sharpness
    grad_clip = 1.0,                        # Gradient clipping norm
    gpt_kind = "gpt2-medium",               # Pre-trained Model Kind
    fine_tune_mode = "peft",                # Fine-tune Mode: Use full-model or peft
    temperature = 1.2,                      # Temperature Sampling during evaluation
    top_k = 50,                             # Top K Sampling during evaluation
    top_p = 0.9,                            # Top P Sampling during evaluation
    max_new_tokens = 128,                   # Max generated tokens for evaluation
    eval_path = "../data/sonnets_held_out_dev.txt",  # Held-out sonnets for DPO evaluation
    generations_out = "./predictions/generated_sonnets_dpo_dev.txt",  # Evaluation output path
    save_every_epoch = False                # If True, save every DPO epoch checkpoint
)
