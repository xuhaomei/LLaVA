from torch import nn
import torch
from .llava_arch import encode_images, prepare_inputs_labels_for_multimodal
from .llava_llama import llava_tokenselector_forward
import types
# from .qwen_selector import replace_qwen2_vl_attention_class, qwen25vl_tokenselector_vision_tower_forward, qwen25vl_tokenselector_generation_forward

def init_token_selector(model, mode=0, k=64):
    model.mode = mode
    model.k = k
    model.model.mode = mode
    model.model.k = k
    model.model.token_selector = nn.Linear(model.config.mm_hidden_size, 1, bias=False, device=model.device, dtype=model.dtype)
    nn.init.normal_(model.model.token_selector.weight, std=0.001)
    model.encode_images = types.MethodType(encode_images, model)
    model.prepare_inputs_labels_for_multimodal = types.MethodType(prepare_inputs_labels_for_multimodal, model)
    model.forward = types.MethodType(llava_tokenselector_forward, model)
    return model

def load_token_selector(model, tokenselector_bin_path):
    try:
        state = torch.load(tokenselector_bin_path, map_location=model.device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Missing keys when loading token selector:", missing)
        print("Unexpected keys when loading token selector:", unexpected)
        # self._model.model.token_selector.to(device=self._device, dtype=torch.float16)
        print(">>>> Successfully loaded token selector from {} <<<<".format(tokenselector_bin_path))
    except Exception as e:
        print(">>>> Failed to load token selector from {}, error: {} <<<<".format(tokenselector_bin_path, e))
    return model

def init_token_selector_qwen(model, mode=0, keep_ratio=0.1):
    replace_qwen2_vl_attention_class()
    model.visual.mode = mode
    model.visual.keep_ratio = keep_ratio
    model.visual.forward = types.MethodType(qwen25vl_tokenselector_vision_tower_forward, model.visual)
    model.visual.token_selector = nn.Linear(3584, 1, dtype=model.dtype, bias=False)
    nn.init.normal_(model.visual.token_selector.weight, std=0.001)
    model.forward = types.MethodType(qwen25vl_tokenselector_generation_forward, model)
    return model

def load_token_selector_qwen(model, tokenselector_bin_path):
    try:
        state = torch.load(tokenselector_bin_path, map_location=model.device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Missing keys when loading token selector:", missing)
        print("Unexpected keys when loading token selector:", unexpected)
        # self._model.model.token_selector.to(device=self._device, dtype=torch.float16)
        print(">>>> Successfully loaded token selector from {} <<<<".format(tokenselector_bin_path))
    except Exception as e:
        print(">>>> Failed to load token selector from {}, error: {} <<<<".format(tokenselector_bin_path, e))
    return model