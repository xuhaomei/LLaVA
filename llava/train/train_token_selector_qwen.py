from llava.train.train_qwen import train
import torch
from transformers import set_seed

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    set_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train(attn_implementation="flash_attention_2")