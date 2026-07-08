"""
Echo-only training (NO trajectory / NO LM loss) with LocationQualityTracker.

This is the echo-only counterpart to the trajectory+echo fine-tuning model.
It trains ONLY the echo classification task: encode 5 echo units from a
location, run them through the NoPE GPT backbone, classify the last hidden
state with aux_classifier into one of num_aux_classes locations.

The LocationQualityTracker is integrated so this run uses the SAME
learnability-focusing mechanism as the trajectory+echo model, making the
two directly comparable.

Tracker integration — loss-weighting (NOT collator sampling):
  - compute_loss computes per-sample CE loss (reduction='none').
  - Each sample is weighted by quality = exp(-mean_loss_for_its_location),
    clamped to a floor. Unlearnable locations (persistently high loss)
    contribute proportionally less gradient.
  - The tracker is updated with the RAW per-sample losses (so it measures
    true learnability, not the down-weighted loss).
  - All of this happens in the main process, so it works identically for
    any dataloader_num_workers value. This is what makes the echo-only run
    and the echo+trajectory run comparable.

Output:
  - The trained model checkpoint.
  - quality_trackers.pt          : {loc_key: mean_loss, ...} + seen counts
  - location_learnability.csv    : per-location mean_loss and seen count,
                                   the artifact for the echo-only vs
                                   echo+trajectory comparison.
"""

import os, sys, glob, copy, math, random, re, time
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import (
    TrainingArguments, GPTNeoForCausalLM, PreTrainedTokenizerFast,
    Trainer, AutoConfig,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.gpt_neo.modeling_gpt_neo import (
    GPTNeoModel, _prepare_4d_causal_attention_mask,
)
from transformers.utils import logging
from datasets import load_from_disk
from safetensors.torch import load_file

logger = logging.get_logger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --- Global dicts ---
location_map_global = {}
echo_dataset_global = None
echo_indices_by_location_global = None


def build_location_index_map(dataset):
    print("Building in-memory index map (location_key -> dataset indices)...")
    index_map = defaultdict(list)
    if 'location_key' not in dataset.column_names:
        raise ValueError("Dataset loaded from disk is missing 'location_key' column.")
    location_keys = dataset['location_key']
    for i, key in enumerate(tqdm(location_keys, desc="Indexing Locations")):
        index_map[int(key)].append(i)
    print(f"Built index map for {len(index_map)} unique locations.")
    return index_map


# =============================================================================
# LocationQualityTracker
# =============================================================================

class LocationQualityTracker:
    """
    Tracks a per-location running-mean loss. quality() turns that into a
    weight in [floor, 1] via exp(-mean_loss): learnable locations (low loss)
    get high weight, unlearnable ones get the floor.

    Keyed by the ORIGINAL location_key (not the contiguous class index), so
    the saved tracker is directly comparable to the echo+trajectory model's.
    """
    def __init__(self, momentum=0.99, init_loss=7.0, floor=0.05, name=""):
        self.momentum = momentum
        self.init_loss = init_loss
        self.floor = floor
        self.name = name
        self.mean_loss = defaultdict(lambda: init_loss)
        self.seen = defaultdict(int)

    def update(self, loc_keys, losses):
        for k, l in zip(loc_keys, losses):
            if not math.isfinite(l):
                continue
            n = self.seen[k]
            if n < 10:
                # simple running mean for the first few observations
                self.mean_loss[k] = (n * self.mean_loss[k] + l) / (n + 1)
            else:
                self.mean_loss[k] = (
                    self.momentum * self.mean_loss[k] + (1 - self.momentum) * l
                )
            self.seen[k] += 1

    def quality(self, loc_keys):
        out = []
        for k in loc_keys:
            q = math.exp(-self.mean_loss[k])
            out.append(max(q, self.floor))
        return out

    def sample_weighted(self, loc_keys, n=1):
        """Quality-weighted sampling — provided for completeness; not used
        in this script (we loss-weight instead)."""
        weights = self.quality(loc_keys)
        total = sum(weights)
        probs = [w / total for w in weights]
        return random.choices(loc_keys, weights=probs, k=n)

    def stats(self):
        if not self.mean_loss:
            return {}
        losses = np.array(list(self.mean_loss.values()), dtype=np.float64)
        return {
            'n_locations':     int(len(losses)),
            'mean_loss':       float(losses.mean()),
            'median_loss':     float(np.median(losses)),
            'min_loss':        float(losses.min()),
            'max_loss':        float(losses.max()),
            'frac_loss_below_1': float((losses < 1.0).mean()),
            'frac_loss_below_2': float((losses < 2.0).mean()),
        }


# =============================================================================
# Model definitions (unchanged from the echo-only script)
# =============================================================================

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, output_size, flash=True):
        super(CrossAttention, self).__init__()
        self.flash = flash
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, query, key, value):
        bsz, q_len, _ = query.size()
        _, k_len, _ = key.size()
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)
        query = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        if self.flash:
            output = nn.functional.scaled_dot_product_attention(query, key, value)
        else:
            att = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
            att = nn.functional.softmax(att, dim=-1)
            output = torch.matmul(att, value)
        output = output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        output = self.o_proj(output)
        return output


class BatNavEncoder_attn2(nn.Module):
    def __init__(self, input_size=15000, patch_size=150, in_channels=1,
                 embed_dim=1024, num_encoder_layers=2, num_heads=8):
        super(BatNavEncoder_attn2, self).__init__()
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=[1, patch_size], stride=[1, patch_size]),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=[1, 3], stride=[1, 1], padding=[0, 1]),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=[1, 3], stride=[1, 1], padding=[0, 1]),
        )
        self.positional_embedding = nn.Parameter(
            torch.randn(int(6 * input_size / patch_size), embed_dim) / (embed_dim ** 0.5)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, batch_first=True),
            num_encoder_layers,
        )
        self.echo_attn = CrossAttention(embed_dim, num_heads, embed_dim, flash=True)
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.feature_weights = nn.Parameter(torch.rand(embed_dim))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        cnn_patches = self.patch_embedding(x)
        if torch.isnan(cnn_patches[0]).any():
            print('Error: cnn_patches has NaN!')
            sys.exit(1)
        cnn_patches = cnn_patches.flatten(2).permute(0, 2, 1)
        cnn_patches_pos = cnn_patches + self.positional_embedding[
            None, :cnn_patches.size(1), :
        ].to(cnn_patches.dtype)
        encoded_patches = self.transformer_encoder(cnn_patches_pos)
        weights = F.softmax(self.feature_weights, dim=0)
        query = (weights * encoded_patches).sum(dim=1, keepdim=True)
        attn_output = self.echo_attn(query, encoded_patches, cnn_patches)
        enhanced_encoding = attn_output + encoded_patches[:, :1, :]
        prediction = self.output_layer(enhanced_encoding.squeeze(1))
        return prediction


class GPTNeoModel_NoPE(GPTNeoModel):
    def __init__(self, config):
        super().__init__(config)
        if hasattr(self, 'wpe'):
            print("Replacing standard absolute positional embedding layer (wpe) with Identity.")
            self.wpe = nn.Identity()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length,
                                        dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        head_mask = self.get_head_mask(head_mask, self.config.num_layers)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds  # wpe removed / Identity

        if hasattr(self, '_use_flash_attention_2') and self._use_flash_attention_2 and self.training:
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif attention_mask is not None:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_length
            )

        hidden_states = self.drop(hidden_states)
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("use_cache=True incompatible with gradient checkpointing.")
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__, hidden_states, None, attention_mask,
                    head_mask[i], use_cache, output_attentions,
                )
            else:
                outputs = block(hidden_states, layer_past=layer_past,
                                attention_mask=attention_mask, head_mask=head_mask[i],
                                use_cache=use_cache, output_attentions=output_attentions)
            hidden_states = outputs[0]
            if use_cache:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states,
                                     all_self_attentions] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states, past_key_values=presents,
            hidden_states=all_hidden_states, attentions=all_self_attentions,
        )


@dataclass
class SplitTaskOutput(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    echo_hidden_states: Optional[torch.FloatTensor] = None


class GPTNeoForCausalLM_RawEcho_WithAux(GPTNeoForCausalLM):
    _tied_weights_keys = []

    def __init__(self, config, num_aux_classes: int, echo_encoder_config: dict):
        super().__init__(config)
        self.transformer = GPTNeoModel_NoPE(config)
        if hasattr(self, 'lm_head'):
            del self.lm_head
        self.echo_encoder = BatNavEncoder_attn2(**echo_encoder_config)
        self.echo_embed_dim = echo_encoder_config.get('embed_dim', config.hidden_size)
        if self.echo_embed_dim != config.hidden_size:
            logger.warning(f"Echo encoder embed_dim ({self.echo_embed_dim}) != "
                           f"transformer hidden_size ({config.hidden_size}).")
        if num_aux_classes <= 0:
            raise ValueError("num_aux_classes must be positive.")
        self.aux_classifier = nn.Linear(config.hidden_size, num_aux_classes)
        print(f"Initialized Echo-Only Model with Aux Classifier: "
              f"{config.hidden_size} -> {num_aux_classes} classes")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        raw_echo_tensors: Optional[List[torch.Tensor]] = None,
        echo_placeholder_indices=None,
        aux_labels_info=None,
        past_key_values=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple, SplitTaskOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        device = self.aux_classifier.weight.device
        dtype = self.aux_classifier.weight.dtype

        if not raw_echo_tensors:
            logger.warning("raw_echo_tensors not provided. Echo path skipped.")
            return SplitTaskOutput(
                loss=None, logits=None, echo_hidden_states=None,
                past_key_values=None, hidden_states=None, attentions=None,
            )

        self.echo_encoder.to(device=device, dtype=dtype)

        num_echo_samples = len(raw_echo_tensors)
        combined_raw_echoes = torch.stack(raw_echo_tensors).to(device, dtype=dtype)
        flat_batch_echoes = combined_raw_echoes.view(
            num_echo_samples * 5,
            combined_raw_echoes.size(2),
            combined_raw_echoes.size(3),
            combined_raw_echoes.size(4),
        )
        encoded_flat_batch = self.echo_encoder(flat_batch_echoes)
        encoded_echo_groups = encoded_flat_batch.view(
            num_echo_samples, 5, self.echo_embed_dim
        ).to(dtype)

        echo_batch_size, echo_seq_length, _ = encoded_echo_groups.shape
        if position_ids is None:
            echo_position_ids = torch.arange(0, echo_seq_length, dtype=torch.long, device=device)
            echo_position_ids = echo_position_ids.unsqueeze(0).expand(echo_batch_size, -1)
        else:
            echo_position_ids = position_ids

        echo_transformer_outputs = self.transformer(
            inputs_embeds=encoded_echo_groups,
            attention_mask=None,
            past_key_values=past_key_values,
            token_type_ids=token_type_ids,
            position_ids=echo_position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden_states_echo_last = echo_transformer_outputs.last_hidden_state

        return SplitTaskOutput(
            loss=None,
            logits=None,
            echo_hidden_states=hidden_states_echo_last,
            past_key_values=echo_transformer_outputs.past_key_values,
            hidden_states=echo_transformer_outputs.hidden_states,
            attentions=echo_transformer_outputs.attentions,
        )


# =============================================================================
# Collate function — adds loc_key to aux_labels_info so the tracker can be
# keyed by the original location id.
# =============================================================================

def custom_collate_fn(batch, tokenizer, location_map, hf_echo_dataset, echo_index_map):
    texts = [example["text"].replace(" ", "") for example in batch]
    encodings = tokenizer(texts, truncation=True, max_length=1024,
                          padding="max_length", return_tensors="pt")
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    location_token_pattern = re.compile(r'\[LOC_(\d+)\]')

    raw_echo_groups_to_batch = []
    aux_labels_info = []   # list of (idx_in_echo_batch, class_index, loc_key)

    valid_location_keys_list = [
        key for key in location_map.keys()
        if key in echo_index_map and len(echo_index_map[key]) >= 5
    ]
    if not valid_location_keys_list:
        logger.warning("No valid locations available for selection / fallback.")

    for i in range(len(texts)):
        ids_list = input_ids[i].tolist()

        echo_source_key = None
        final_aux_target_label = None
        label_determined = False

        # Attempt 1: a LOC token present in the text
        possible_loc_candidates = []
        for token_id in ids_list:
            if token_id == tokenizer.pad_token_id:
                break
            token_str = tokenizer.decode([token_id])
            match = location_token_pattern.fullmatch(token_str)
            if match:
                try:
                    location_key = int(match.group(1))
                    if (location_key in location_map
                            and location_key in echo_index_map
                            and len(echo_index_map[location_key]) >= 5):
                        possible_loc_candidates.append(location_key)
                except ValueError:
                    pass

        if possible_loc_candidates:
            selected_loc_key = random.choice(possible_loc_candidates)
            echo_source_key = selected_loc_key
            final_aux_target_label = location_map[selected_loc_key]
            label_determined = True
        elif valid_location_keys_list:
            random_loc_key = random.choice(valid_location_keys_list)
            echo_source_key = random_loc_key
            final_aux_target_label = location_map[random_loc_key]
            label_determined = True

        if label_determined and echo_source_key is not None:
            available_ds_indices = echo_index_map.get(echo_source_key, [])
            if len(available_ds_indices) >= 5:
                sampled_ds_indices = random.sample(available_ds_indices, 5)
                try:
                    echo_arrays_np = [
                        np.array(hf_echo_dataset[idx]['echo_array'], dtype=np.float16)
                        for idx in sampled_ds_indices
                    ]
                    stacked = np.stack(echo_arrays_np, axis=0)
                    raw_echo_groups_to_batch.append(
                        torch.tensor(stacked, dtype=torch.float32).unsqueeze(1)
                    )
                    aux_labels_info.append((
                        len(raw_echo_groups_to_batch) - 1,
                        final_aux_target_label,
                        echo_source_key,           # <-- NEW: original loc_key
                    ))
                except Exception as e:
                    logger.error(f"Error fetching echoes for {echo_source_key}: {e}")

    batch_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if raw_echo_groups_to_batch:
        batch_dict["raw_echo_tensors"] = raw_echo_groups_to_batch
        batch_dict["aux_labels_info"] = aux_labels_info
    return batch_dict


# =============================================================================
# CustomTrainer — tracker-aware compute_loss (loss-weighting)
# =============================================================================

class CustomTrainer(Trainer):
    def __init__(self, *args, alpha=1.0, location_map=None, loc_tracker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.location_map = location_map
        self.loc_tracker = loc_tracker
        print(f"CustomTrainer initialized: alpha={self.alpha}, "
              f"tracker={'ON' if loc_tracker is not None else 'OFF'}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        raw_echo_tensors = inputs.pop("raw_echo_tensors", None)
        aux_labels_info = inputs.pop("aux_labels_info", None)

        outputs: SplitTaskOutput = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            raw_echo_tensors=raw_echo_tensors,
            return_dict=True,
        )

        device = inputs["input_ids"].device
        aux_loss = torch.tensor(0.0, device=device)

        if (aux_labels_info and outputs.echo_hidden_states is not None
                and hasattr(model, "aux_classifier")):
            target_aux_labels = [info[1] for info in aux_labels_info]
            loc_keys          = [info[2] for info in aux_labels_info]

            states_for_aux = outputs.echo_hidden_states[:, -1, :]   # [N_echo, H]
            target_tensor = torch.tensor(
                target_aux_labels, dtype=torch.long, device=states_for_aux.device,
            )
            aux_logits = model.aux_classifier(states_for_aux)        # [N_echo, C]

            # Per-sample loss — needed for both weighting and tracker update.
            per_sample = F.cross_entropy(
                aux_logits, target_tensor, reduction='none',
            )                                                        # [N_echo]

            # Quality-weighted mean. Weights computed from the tracker's
            # CURRENT state (i.e. before this batch's update).
            if self.loc_tracker is not None:
                q = self.loc_tracker.quality(loc_keys)
                w = torch.tensor(q, dtype=per_sample.dtype, device=per_sample.device)
                weighted = (per_sample * w).sum() / w.sum().clamp_min(1e-8)
            else:
                weighted = per_sample.mean()

            if torch.isfinite(weighted):
                aux_loss = weighted
            else:
                logger.warning("Auxiliary loss is NaN/Inf!")

            # Update the tracker with the RAW (un-weighted) per-sample losses.
            if self.loc_tracker is not None:
                with torch.no_grad():
                    raw_losses = per_sample.detach().float().cpu().tolist()
                    self.loc_tracker.update(loc_keys, raw_losses)

        total_loss = self.alpha * aux_loss

        if random.random() < 0.01:
            n_echo = len(aux_labels_info) if aux_labels_info else 0
            msg = f"compute_loss - Aux: {aux_loss.item():.4f}  (n_echo={n_echo})"
            if self.loc_tracker is not None and random.random() < 0.3:
                s = self.loc_tracker.stats()
                if s:
                    msg += (f"  | tracker: {s['n_locations']} locs, "
                            f"mean_loss {s['mean_loss']:.3f}, "
                            f"frac<1.0 {s['frac_loss_below_1']:.3f}")
            print(msg)

        return (total_loss, outputs) if return_outputs else total_loss


# =============================================================================
# Main
# =============================================================================

def main():
    global echo_dataset_global, echo_indices_by_location_global, location_map_global

    # --- 1. Echo dataset ---
    echo_dataset_path = "/scratch200/xingchen/processed_echo_dataset_parallel_with_coords_v2"
    if not os.path.exists(echo_dataset_path):
        sys.exit(f"Echo dataset not found at {echo_dataset_path}")
    print(f"Loading echo dataset from {echo_dataset_path}...")
    echo_dataset_global = load_from_disk(echo_dataset_path)
    print(f"Loaded echo dataset with {len(echo_dataset_global)} samples.")

    # --- 2. Index + filter + location map ---
    raw_echo_indices_map = build_location_index_map(echo_dataset_global)
    MIN_SAMPLES_PER_LOCATION = 15
    filtered_echo_indices_map = {
        k: v for k, v in raw_echo_indices_map.items()
        if len(v) >= MIN_SAMPLES_PER_LOCATION
    }
    echo_indices_by_location_global = filtered_echo_indices_map
    print(f"After filtering (>= {MIN_SAMPLES_PER_LOCATION} samples): "
          f"{len(echo_indices_by_location_global)} locations.")

    valid_locations = sorted(echo_indices_by_location_global.keys())
    location_map_global = {loc_id: i for i, loc_id in enumerate(valid_locations)}
    num_aux_classes = len(location_map_global)
    if num_aux_classes <= 0:
        sys.exit("No valid locations after filtering.")
    print(f"Built location map for {num_aux_classes} aux classes.")

    # Reverse map (class index -> loc_key), saved for downstream analysis
    index_to_location = {v: k for k, v in location_map_global.items()}

    # --- 3. Tokenizer ---
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="./ops/tokenizer_v9_2.json", max_len=1024,
        pad_token='[PAD]', unk_token='[UNK]', sep_token='[SEP]',
        cls_token='[CLS]', bos_token='[BOS]', eos_token='[EOS]',
        additional_special_tokens=['[INST]', '[SRA]', '[POS]'],
    )
    print(f"PAD:{tokenizer.pad_token_id} UNK:{tokenizer.unk_token_id} "
          f"SEP:{tokenizer.sep_token_id} CLS:{tokenizer.cls_token_id}")
    assert all(x is not None for x in
               [tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.sep_token_id])

    # --- 4. Trajectory text dataset ---
    dataset_path = './train_dataset_finetune_5400_loc_v1'
    if not os.path.exists(dataset_path):
        sys.exit(f"Dataset not found at {dataset_path}")
    dataset = load_from_disk(dataset_path)
    total = len(dataset)
    split_idx = int(total * 0.9996)
    train_dataset = dataset.select(range(0, split_idx))
    eval_dataset  = dataset.select(range(split_idx, total))
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    # --- 5. Model ---
    echo_encoder_config = {
        'input_size': 15000, 'patch_size': 150, 'in_channels': 1,
        'embed_dim': 1024, 'num_encoder_layers': 2, 'num_heads': 8,
    }
    model_path = "./GPT_NEO_base_AUX_v1_location_v4token/checkpoint-10"
    if not os.path.exists(model_path):
        sys.exit(f"Model checkpoint not found at {model_path}")
    config = AutoConfig.from_pretrained(model_path)
    model = GPTNeoForCausalLM_RawEcho_WithAux(
        config=config,
        num_aux_classes=num_aux_classes,
        echo_encoder_config=echo_encoder_config,
    )

    # --- 6. LocationQualityTracker ---
    # init_loss = ln(num_classes): a uniform classifier's CE. So every
    # location starts at floor quality (uniform weighting), and the tracker
    # differentiates locations as training reveals which are learnable.
    init_loss = math.log(num_aux_classes)
    loc_tracker = LocationQualityTracker(
        momentum=0.99, init_loss=init_loss, floor=0.05, name="loc",
    )
    print(f"LocationQualityTracker: init_loss = ln({num_aux_classes}) = {init_loss:.3f}")

    # --- 7. Training args ---
    output_dir = "./GPT_NEO_ECHO_CLS_ONLY_with_tracker_v1"
    deepspeed_config = './ops/ds_config3_6_large_lr.json'
    if not os.path.exists(deepspeed_config):
        deepspeed_config = None

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=192,
        per_device_eval_batch_size=192,
        eval_strategy='no',                  # avoid all-gather deadlock on custom output
        logging_strategy='steps',
        logging_steps=1000,
        save_strategy='steps',
        save_steps=1000,
        save_total_limit=2,
        weight_decay=0.1,
        learning_rate=3.0e-4,
        adam_beta1=0.9, adam_beta2=0.95,
        fp16=False, bf16=True,
        gradient_accumulation_steps=2,
        deepspeed=deepspeed_config,
        dataloader_drop_last=True,
        # The tracker lives in compute_loss (main process), NOT the collator,
        # so num_workers > 0 is safe and does not break the tracker.
        dataloader_num_workers=4,
        max_grad_norm=200.0,
    )

    collator = partial(
        custom_collate_fn, tokenizer=tokenizer, location_map=location_map_global,
        hf_echo_dataset=echo_dataset_global, echo_index_map=echo_indices_by_location_global,
    )

    trainer = CustomTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=collator, tokenizer=tokenizer,
        alpha=1.0, location_map=location_map_global,
        loc_tracker=loc_tracker,
    )

    # --- 8. Train ---
    print("\nStarting echo-only training with LocationQualityTracker...")
    train_result = trainer.train()
    trainer.save_model()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # --- 9. Save the tracker (the comparison artifact) ---
    qpath = os.path.join(output_dir, "quality_trackers.pt")
    torch.save({
        'loc_tracker': dict(loc_tracker.mean_loss),
        'loc_seen':    dict(loc_tracker.seen),
        'init_loss':   init_loss,
    }, qpath)
    print(f"Saved quality tracker -> {qpath}")

    # CSV: per-location learnability, easy to load for the echo-only vs
    # echo+trajectory comparison plot.
    rows = []
    for loc_key, mean_loss in loc_tracker.mean_loss.items():
        rows.append({
            'location_key': loc_key,
            'class_index':  location_map_global.get(loc_key, -1),
            'mean_loss':    float(mean_loss),
            'quality':      max(math.exp(-float(mean_loss)), loc_tracker.floor),
            'times_seen':   int(loc_tracker.seen.get(loc_key, 0)),
        })
    csv_path = os.path.join(output_dir, "location_learnability.csv")
    pd.DataFrame(rows).sort_values('mean_loss').to_csv(csv_path, index=False)
    print(f"Saved per-location learnability -> {csv_path}")

    s = loc_tracker.stats()
    print(f"\nFinal tracker stats: {s}")
    print("Echo-only training finished.")


if __name__ == '__main__':
    main()