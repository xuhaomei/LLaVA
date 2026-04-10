from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from .utils import gumbel_top_k, bernoulli_sample, multinomial_sample

from transformers.utils import (
    is_flash_attn_2_available
)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb

else:
    flash_attn_varlen_func = None
    apply_rotary_emb = None


if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
else:
    flash_attn_varlen_func = None
from .rope2d import get_rope_index_25
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast


def _get_llm_vision_token_count(grid_thw: torch.Tensor, spatial_merge_size: int) -> int:
    return int(grid_thw[0].item() * (grid_thw[1].item() // spatial_merge_size) * (grid_thw[2].item() // spatial_merge_size))


def _count_media_per_sample(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    vision_start_token_id: int,
    media_token_id: int,
) -> List[int]:
    media_counts = []
    valid_segments = _get_valid_token_segments(input_ids, attention_mask)
    packed_row = 0 if _is_cu_seqlens_attention_mask(attention_mask, input_ids) else None

    for batch_idx, valid_positions in enumerate(valid_segments):
        current_row = packed_row if packed_row is not None else batch_idx
        valid_tokens = input_ids[current_row][valid_positions]
        vision_start_indices = torch.argwhere(valid_tokens == vision_start_token_id).squeeze(1)
        if vision_start_indices.numel() == 0:
            media_counts.append(0)
            continue
        next_token_indices = vision_start_indices + 1
        next_token_indices = next_token_indices[next_token_indices < valid_tokens.size(0)]
        media_counts.append((valid_tokens[next_token_indices] == media_token_id).sum().item())
    return media_counts


def _pad_and_stack_1d(sequences: List[torch.Tensor], padding_value: int) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=padding_value)


def _pad_and_stack_position_ids(sequences: List[torch.Tensor], padding_value: int = 1) -> torch.Tensor:
    max_len = max(seq.size(-1) for seq in sequences)
    output = sequences[0].new_full((sequences[0].size(0), len(sequences), max_len), padding_value)
    for batch_idx, seq in enumerate(sequences):
        output[:, batch_idx, : seq.size(-1)] = seq
    return output


def _is_cu_seqlens_attention_mask(attention_mask: Optional[torch.Tensor], input_ids: torch.Tensor) -> bool:
    return (
        attention_mask is not None
        and attention_mask.ndim == 1
        and input_ids.size(0) == 1
        and attention_mask.numel() >= 2
        and attention_mask[0].item() == 0
    )


def _get_valid_token_segments(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[torch.Tensor]:
    if _is_cu_seqlens_attention_mask(attention_mask, input_ids):
        return [
            torch.arange(
                attention_mask[idx].item(),
                attention_mask[idx + 1].item(),
                device=input_ids.device,
                dtype=torch.long,
            )
            for idx in range(attention_mask.size(0) - 1)
        ]

    return [
        torch.nonzero(attention_mask[batch_idx] == 1, as_tuple=False).squeeze(1)
        for batch_idx in range(input_ids.size(0))
    ]


def _prune_media_tokens(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    media_grid_thw: Optional[torch.Tensor],
    keep_indices_list: Optional[List[torch.Tensor]],
    media_token_id: int,
):
    if media_grid_thw is None or keep_indices_list is None:
        return input_ids, attention_mask, labels, position_ids

    if len(keep_indices_list) == 0:
        return input_ids, attention_mask, labels, position_ids

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    else:
        attention_mask = attention_mask.to(input_ids.device)
    is_packed = _is_cu_seqlens_attention_mask(attention_mask, input_ids)

    spatial_merge_size = self.config.vision_config.spatial_merge_size
    media_counts = _count_media_per_sample(
        input_ids,
        attention_mask,
        self.config.vision_start_token_id,
        media_token_id,
    )

    seq_keep_indices = []
    media_offset = 0
    valid_segments = _get_valid_token_segments(input_ids, attention_mask)
    for batch_idx, valid_positions in enumerate(valid_segments):
        current_row = 0 if is_packed else batch_idx
        valid_tokens = input_ids[current_row][valid_positions]
        valid_keep_mask = torch.ones(valid_tokens.size(0), dtype=torch.bool, device=input_ids.device)
        media_positions = torch.nonzero(valid_tokens == media_token_id, as_tuple=False).squeeze(1)

        media_cursor = 0
        for local_media_idx in range(media_counts[batch_idx]):
            global_media_idx = media_offset + local_media_idx
            media_token_count = _get_llm_vision_token_count(media_grid_thw[global_media_idx], spatial_merge_size)
            cur_positions = media_positions[media_cursor : media_cursor + media_token_count]
            if cur_positions.numel() != media_token_count:
                raise ValueError(
                    f"Media token count mismatch for batch {batch_idx}: expected {media_token_count}, got {cur_positions.numel()}"
                )

            cur_keep_indices = keep_indices_list[global_media_idx].to(device=input_ids.device, dtype=torch.long)
            if cur_keep_indices.numel() == 0:
                raise ValueError(f"Media token selector kept zero tokens for batch {batch_idx}, media {global_media_idx}")
            if cur_keep_indices.max().item() >= media_token_count or cur_keep_indices.min().item() < 0:
                raise ValueError(
                    f"Media keep indices out of range for batch {batch_idx}, media {global_media_idx}: "
                    f"max={cur_keep_indices.max().item()}, token_count={media_token_count}"
                )

            valid_keep_mask[cur_positions] = False
            valid_keep_mask[cur_positions[cur_keep_indices]] = True
            media_cursor += media_token_count

        if media_cursor != media_positions.numel():
            raise ValueError(
                f"Unused media tokens remain in batch {batch_idx}: used {media_cursor}, total {media_positions.numel()}"
            )

        seq_keep_indices.append(valid_positions[valid_keep_mask])
        media_offset += media_counts[batch_idx]

    if media_offset != len(keep_indices_list):
        raise ValueError(f"Media keep metadata mismatch: consumed {media_offset}, expected {len(keep_indices_list)}")

    if is_packed:
        input_ids = torch.cat([input_ids[0, keep_idx] for keep_idx in seq_keep_indices], dim=0).unsqueeze(0)
        kept_seq_lens = torch.tensor(
            [keep_idx.size(0) for keep_idx in seq_keep_indices],
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([attention_mask.new_zeros(1), kept_seq_lens.cumsum(dim=0)], dim=0)

        if labels is not None:
            labels = torch.cat([labels[0, keep_idx] for keep_idx in seq_keep_indices], dim=0).unsqueeze(0)

        if position_ids is not None:
            if position_ids.ndim == 3:
                position_ids = torch.cat(
                    [position_ids[:, 0, keep_idx] for keep_idx in seq_keep_indices], dim=-1
                ).unsqueeze(1)
            elif position_ids.ndim == 2:
                position_ids = torch.cat([position_ids[0, keep_idx] for keep_idx in seq_keep_indices], dim=0).unsqueeze(0)
            else:
                raise ValueError(f"Unsupported position_ids ndim: {position_ids.ndim}")
    else:
        pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else 0
        input_ids = _pad_and_stack_1d([input_ids[idx, keep_idx] for idx, keep_idx in enumerate(seq_keep_indices)], pad_token_id)
        attention_mask = _pad_and_stack_1d(
            [attention_mask[idx, keep_idx].new_ones(keep_idx.size(0)) for idx, keep_idx in enumerate(seq_keep_indices)],
            0,
        )

        if labels is not None:
            labels = _pad_and_stack_1d([labels[idx, keep_idx] for idx, keep_idx in enumerate(seq_keep_indices)], -100)

        if position_ids is not None:
            if position_ids.ndim == 3:
                position_ids = _pad_and_stack_position_ids(
                    [position_ids[:, idx, keep_idx] for idx, keep_idx in enumerate(seq_keep_indices)]
                )
            elif position_ids.ndim == 2:
                position_ids = _pad_and_stack_1d(
                    [position_ids[idx, keep_idx] for idx, keep_idx in enumerate(seq_keep_indices)],
                    1,
                )
            else:
                raise ValueError(f"Unsupported position_ids ndim: {position_ids.ndim}")

    return input_ids, attention_mask.to(torch.int32), labels, position_ids


def _compute_per_sample_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    vocab_size: int,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loss_fct = CrossEntropyLoss(reduction="none", ignore_index=ignore_index)

    if _is_cu_seqlens_attention_mask(attention_mask, labels):
        cu_seqlens = attention_mask.to(device=logits.device, dtype=torch.long)
        per_sample_losses = []

        for idx in range(cu_seqlens.numel() - 1):
            start = cu_seqlens[idx].item()
            end = cu_seqlens[idx + 1].item()

            if end - start <= 1:
                per_sample_losses.append(logits.new_zeros(()))
                continue

            sample_shift_logits = logits[:, start : end - 1, :].contiguous().view(-1, vocab_size)
            sample_shift_labels = labels[:, start + 1 : end].contiguous().view(-1).to(logits.device)
            sample_token_loss = loss_fct(sample_shift_logits, sample_shift_labels)
            valid_mask = sample_shift_labels != ignore_index

            if valid_mask.any():
                per_sample_losses.append(sample_token_loss[valid_mask].mean())
            else:
                per_sample_losses.append(logits.new_zeros(()))

        my_loss = torch.stack(per_sample_losses, dim=0)
        loss = my_loss.mean()
        return loss, my_loss

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(logits.device)
    token_loss = loss_fct(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
    ).view_as(shift_labels)
    valid_mask = shift_labels != ignore_index
    valid_token_count = valid_mask.sum(dim=-1).clamp_min(1)
    my_loss = (token_loss * valid_mask).sum(dim=-1) / valid_token_count
    loss = my_loss.mean()
    return loss, my_loss


def qwen25vl_tokenselector_vision_tower_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
    """
    Args:
        hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
            The final hidden states of the model.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width of feature shape of each image in LLM.

    Returns:
        `torch.Tensor`: hidden_states.
    """
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    for layer_num, blk in enumerate(self.blocks):
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens
        if self.gradient_checkpointing and self.training:
            hidden_states = self._gradient_checkpointing_func(
                blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
            )
        else:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)

    hidden_states = self.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]

    # ---------------------------add start---------------------------------------------
    num_images_or_videos = grid_thw.size(0)
    hidden_states = hidden_states.unsqueeze(0)
    logits = self.token_selector(hidden_states)
    hidden_states_new_list = []
    log_probs_list = []
    keep_indices_list = []
    for i in range(num_images_or_videos):
        cur_logits = logits[:,cu_seqlens[i]//self.spatial_merge_unit:cu_seqlens[i+1]//self.spatial_merge_unit, :]
        cur_hidden_states = hidden_states[:,cu_seqlens[i]//self.spatial_merge_unit:cu_seqlens[i+1]//self.spatial_merge_unit, :]
        cur_logits_len = cur_logits.size(1)
        k = int(cur_logits_len * self.keep_ratio)
        k = max(1, min(cur_logits_len, k))
        if self.training:
            if self.mode == 0:
                mask, log_probs = gumbel_top_k(cur_logits, k)
                row_idx = torch.arange(mask.size(0)).unsqueeze(1).repeat(1, k).to(mask.device)
                col_idx = torch.stack([torch.where(mask[i])[0].sort().values for i in range(mask.size(0))])
                hidden_states_new = cur_hidden_states[row_idx, col_idx]
                log_probs_list.append(log_probs[0])
            elif self.mode == 1:
                assert cur_logits.size(0) == 1, "Bernoulli sample is NOT avaliable when batch_size > 1."
                mask, log_probs = bernoulli_sample(cur_logits, k)
                col_idx = torch.where(mask.squeeze(0).squeeze(-1).bool())[0].sort().values
                hidden_states_new = cur_hidden_states[:, col_idx, :]
                log_probs_list.append(log_probs[0])
            elif self.mode == 2:
                mask, log_probs = multinomial_sample(cur_logits, k)
                row_idx = torch.arange(mask.size(0)).unsqueeze(1).repeat(1, k).to(mask.device)
                col_idx = torch.stack([torch.where(mask[i])[0].sort().values for i in range(mask.size(0))])
                hidden_states_new = cur_hidden_states[row_idx, col_idx]
                log_probs_list.append(log_probs[0])
            else:
                raise NotImplementedError("Mode:3 is not implemented. Please check your config.")
        else:
            _, topk_idx = cur_logits.squeeze(-1).topk(k, dim=-1)
            row_idx = torch.arange(topk_idx.size(0)).unsqueeze(1).repeat(1, k).to(topk_idx.device)
            col_idx = topk_idx.sort(dim=-1).values
            hidden_states_new = cur_hidden_states[row_idx, col_idx]
            log_probs = None
        keep_indices_list.append(col_idx.squeeze(0))
        hidden_states_new_list.append(hidden_states_new.squeeze(0))
    hidden_states_new = torch.cat(hidden_states_new_list, dim=0)
    log_probs = torch.stack(log_probs_list, dim=0).unsqueeze(0) if self.training else None
    if getattr(self, "eval_policy_entropy", False):
        return logits, num_images_or_videos, cu_seqlens
    return hidden_states_new, log_probs, {"keep_indices": keep_indices_list}
    # ---------------------------add end---------------------------------------------



def qwen25vl_tokenselector_generation_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    original_input_ids = input_ids
    original_attention_mask = attention_mask

    if inputs_embeds is None:
        # For the prefill stage, multimodal RoPE must be computed on the original sequence.
        # After that, we prune tokens and prune position_ids with the same keep indices.
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = get_rope_index_25(
                    self.config.vision_config.spatial_merge_size,
                    original_input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    original_attention_mask,
                )
                self.rope_deltas = rope_deltas

        image_selector_outputs = None
        video_selector_outputs = None
        selector_log_probs = []
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds, image_log_probs, image_selector_outputs = self.visual(pixel_values, grid_thw=image_grid_thw)
            if image_log_probs is not None:
                selector_log_probs.append(image_log_probs)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds, video_log_probs, video_selector_outputs = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            if video_log_probs is not None:
                selector_log_probs.append(video_log_probs)

        if image_selector_outputs is not None:
            input_ids, attention_mask, labels, position_ids = _prune_media_tokens(
                self,
                input_ids,
                attention_mask,
                labels,
                position_ids,
                image_grid_thw,
                image_selector_outputs.get("keep_indices"),
                self.config.image_token_id,
            )

        if video_selector_outputs is not None:
            input_ids, attention_mask, labels, position_ids = _prune_media_tokens(
                self,
                input_ids,
                attention_mask,
                labels,
                position_ids,
                video_grid_thw,
                video_selector_outputs.get("keep_indices"),
                self.config.video_token_id,
            )

        inputs_embeds = self.model.embed_tokens(input_ids)

        if pixel_values is not None:
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]

            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match after pruning: "
                    f"tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match after pruning: "
                    f"tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)
    else:
        selector_log_probs = []

    if self.training:
        log_probs = torch.cat(selector_log_probs, dim=-1) if len(selector_log_probs) > 0 else None
    else:
        log_probs = None

    # Decode steps do not carry visual tokens anymore, so position_ids are derived from
    # cache_position and the cached rope_deltas from the prefill stage.
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        if (
            (cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        ):
            position_ids, rope_deltas = get_rope_index_25(
                self.config.vision_config.spatial_merge_size,
                original_input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                original_attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        loss, my_loss = _compute_per_sample_causal_lm_loss(
            logits,
            labels,
            attention_mask,
            self.config.vocab_size,
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    output = Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
    if self.training:
        return output, my_loss, log_probs
    else:
        return output

def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_top_left_mask (`bool`, defaults to `False`):
            flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
    """
    assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
    query_states = query_states.squeeze(0)
    key_states = key_states.squeeze(0)
    value_states = value_states.squeeze(0)
    cu_seqlens = attention_mask

    with torch.no_grad():
        max_seqlen = max(
            [
                cu_seqlens[idx + 1] - cu_seqlens[idx]
                for idx in range(cu_seqlens.size(0) - 1)
            ]
        ).item()

    if not use_top_left_mask:
        causal = is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1.
        causal = is_causal and query_length != 1

    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    flash_kwargs = {}

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    attn_output = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=dropout,
        softmax_scale=softmax_scale,
        causal=causal,
        **flash_kwargs,
    )

    attn_output = attn_output.unsqueeze(0)
    query_states = query_states.unsqueeze(0)
    key_states = key_states.unsqueeze(0)
    value_states = value_states.unsqueeze(0)

    return attn_output


def _update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Any,
    output_attentions: bool,
):
    return attention_mask


def replace_qwen2_vl_attention_class():
    import transformers

    transformers.models.qwen2_vl.modeling_qwen2_vl._flash_attention_forward = (
        _flash_attention_forward
    )
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel._update_causal_mask = (
        _update_causal_mask
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl._flash_attention_forward = (
        _flash_attention_forward
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel._update_causal_mask = (
        _update_causal_mask
    )