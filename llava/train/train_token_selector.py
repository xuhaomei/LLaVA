from llava.train.train import train
import torch
from transformers import set_seed

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    seed = 42
    set_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train(attn_implementation="flash_attention_2", token_selector=True)