"""
This file contains program parameters for classifier.py, paraphrase_detection.py and sonnet_generation.py
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
    
    sonnet_train = DATA_FOLDER + "sonnets.txt",
    sonnet_test = DATA_FOLDER + "sonnets_held_out.txt"
)

CLASSIFIER_ARGS = SimpleNamespace(
    seed = 42,                              # Reproduceability Random Seed
    lr = 1e-5,                              # learning rate
    weight_decay = 5e-2,                    # Weight Decay Regularization
    epochs = 1,                            # Training Epochs
    batch_size = 64,                        # Training Batch Size
    num_workers = 0,                        # Dataloader parallelism
    gpt_kind = "gpt2",               # Pre-trained Model Kind
    
    prediction_head = "linear",             # Prediction Head Construction
    fine_tune_mode = "peft",                # Fine-tune Mode: Use full-model, peft, or last-layer
    hidden_dropout_prob = 0.0               # Dropout probability for prediciton head
)

PARAPHRASE_ARGS = SimpleNamespace(
    seed = 42,                              # Reproduceability Random Seed
    lr = 1e-5,                              # learning rate
    weight_decay = 5e-2,                    # Weight Decay Regularization
    epochs = 1,                            # Training Epochs
    batch_size = 8,                        # Training Batch Size
    num_workers = 0,                        # Dataloader parallelism
    gpt_kind = "gpt2",               # Pre-trained Model Kind
    fine_tune_mode = "peft",                # Fine-tune Mode: Use full-model or peft
)

SONNET_ARGS = SimpleNamespace(
    seed = 42,                              # Reproduceability Random Seed
    lr = 1e-5,                              # learning rate
    weight_decay = 5e-2,                    # Weight Decay Regularization
    epochs = 1,                            # Training Epochs
    batch_size = 8,                        # Training Batch Size
    num_workers = 4,                        # Dataloader parallelism
    gpt_kind = "gpt2-medium",               # Pre-trained Model Kind
    fine_tune_mode = "peft",                # Fine-tune Mode: Use full-model or peft

    top_p = 0.9,                            # Top P Sampling
    temperature = 1.2,                      # Temperature Sampling
    top_k = 50,                             # Top K Sampling
    preview_valid = False                   # Preview Validation Sets
)