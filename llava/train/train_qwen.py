# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import pathlib
import torch
import transformers
import shutil
import sys
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from tokenselector import (
   make_qwen_supervised_data_module,
   QwenTokenSelectorTrainer,
   init_token_selector_qwen
)

local_rank = None

import transformers
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    mode: int = 0
    keep_ratio: float = 0.1
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-7B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    tune_selector: bool = field(default=False)
    gumbel_start_tau: float = field(default=1.0, metadata={"help": "Starting value for gumbel_tau"})
    gumbel_end_tau: float = field(default=0.1, metadata={"help": "Ending value for gumbel_tau"})

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    alpha_pg_loss: float = 1.0
    lambda_r: float = 0.1
    sample_num: int = 10
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False
    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False
    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        for n, p in model.lm_head.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        for n, p in model.lm_head.named_parameters():
            p.requires_grad = False
    # -------------------------add selector tuning---------------------------------
    if model_args.tune_selector:
        for n, p in model.visual.token_selector.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.token_selector.named_parameters():
            p.requires_grad = False
    # -------------------------------------------------------------------------------

def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    if "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            device_map="auto"
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        raise ValueError("Model not currently supported")
        
    init_token_selector_qwen(model, model_args.mode, model_args.keep_ratio)
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    # --- print trainable parameters ---
    if local_rank == 0: 
        print("="*80)
        print("Printing trainable parameters...")
        trainable_param_names = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_param_names.append(name)
        
        for name in trainable_param_names:
            print(f"- {name}")
        print("="*80)
    # -----------------------------------
    
    data_module = make_qwen_supervised_data_module(tokenizer=tokenizer, data_args=data_args)    
    data_module['eval_dataset'] = data_module['train_dataset']  # For quick eval during token selector training
    trainer = QwenTokenSelectorTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module
    )
    trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # After training, copy preprocessor_config.json and chat_template.json
    if local_rank == 0:
        source_dir = pathlib.Path(model_args.model_name_or_path)
        dest_dir = pathlib.Path(training_args.output_dir)
        
        preprocessor_file = "preprocessor_config.json"
        chat_template_file = "chat_template.json"
        
        # Copy preprocessor_config.json
        source_preprocessor_path = source_dir / preprocessor_file
        dest_preprocessor_path = dest_dir / preprocessor_file
        if source_preprocessor_path.exists():
            shutil.copy(source_preprocessor_path, dest_preprocessor_path)
            rank0_print(f"Copied {source_preprocessor_path} to {dest_preprocessor_path}")
        else:
            rank0_print(f"Warning: {source_preprocessor_path} not found.")

        # Copy chat_template.json
        source_chat_template_path = source_dir / chat_template_file
        dest_chat_template_path = dest_dir / chat_template_file
        if source_chat_template_path.exists():
            shutil.copy(source_chat_template_path, dest_chat_template_path)
            rank0_print(f"Copied {source_chat_template_path} to {dest_chat_template_path}")
        else:
            rank0_print(f"Warning: {source_chat_template_path} not found.")


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")