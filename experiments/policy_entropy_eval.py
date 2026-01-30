from llava.train.train import LazySupervisedDataset
import torch
from types import SimpleNamespace
from llava.model.builder import load_pretrained_model
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenselector import init_token_selector, load_token_selector

model_name_or_path = "liuhaotian/llava-v1.5-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(model_name_or_path, None, "llava-v1.5-7b",device_map="cuda")

data_args = SimpleNamespace(
    image_folder="/home/weiliu/student/xhm/LLaVA/playground/data",
    image_processor=image_processor,
    image_aspect_ratio="pad",
    is_multimodal=True,
    mm_use_im_start_end=False
)
dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path="/home/weiliu/student/xhm/LLaVA/playground/data/llava_v1_5_mix665k_shuffled.json",
                                data_args=data_args)
images = [dataset[i]['image'] for i in range(1000)]
images = torch.stack(images).to(dtype=model.dtype, device=model.device)

k = 128
init_token_selector(model, mode=0, k=k)
other_cfg = "t128followt32_400"
# tokenselector_bin_path_template="/home/weiliu/student/xhm/LLaVA/checkpoints/llava-v1.5-7b-mix665k_shuffled-lambda0.01-sn50-2e-2-mode0-linearhigh-freq-eval/checkpoint-{}/token_selector.bin"
tokenselector_bin_path_template="/home/weiliu/student/xhm/LLaVA/checkpoints/llava-v1.5-7b-mix665k_shuffled-k{}-lambda0.01-sn50-2e-2-mode0-linear{}/checkpoint-{}/token_selector.bin"
h_list = []
x = range(10, 910, 10)
for step in x:
    bin_path = tokenselector_bin_path_template.format(str(k),other_cfg,str(step))
    load_token_selector(model, bin_path)
    image_features = model.get_vision_tower()(images)
    logits = model.model.token_selector(image_features)
    p = torch.softmax(logits.squeeze(-1), dim=1).float() # (B,N)
    p_safe = torch.clamp(p, min=1e-8)
    h = (-p_safe*torch.log(p_safe)).sum(dim=1).mean()
    h_list.append(h.item())

import matplotlib.pyplot as plt

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制折线图
plt.plot(x, h_list, marker='o')

# 添加标签和标题
plt.title('Policy entropy per step', fontsize=16)
plt.xlabel('step', fontsize=12)
plt.ylabel('entropy', fontsize=12)

# 显示网格
plt.grid(True, alpha=0.3)
plt.savefig('policy_entropy_k{}_{}.pdf'.format(str(k),other_cfg), bbox_inches='tight')
plt.show()