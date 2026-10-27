# config.py
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 128 
    block_size: int = 256      # The maximum context length in tokenizer tokens
    n_embed: int = 384          
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2

    def __post_init__(self):
        assert self.n_embed % self.n_head == 0, "[n_embed] should be divisible by [n_head]"

@dataclass
class TrainConfig:
    batch_size: int = 64       # How many independent sequences are processed in parallel
    max_iters: int = 8000      # Maximum number of training iterations
    learning_rate: float = 3e-4 # Learning rate for the optimizer
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_iters: int = 500
    eval_interval: int = 300   # Evaluate the model every this many iterations
    eval_iters: int = 200      # Number of iterations to average the evaluation loss over
    device: str = "cuda"
    seed: int = 1337
    ckpt_dir: str = "checkpoints"
    model_path: str = "model_dailydialog.pth"
