"""
Fine-tuning with cross-attention fusion of echo (sensory) information.

Two modes of operation, routed by special tokens at sequence start:
  - Mode T ([BOS]):  trajectory only, no SLx   — model self-reliant
  - Mode S ([CLS]):  SLx only, predict SLm      — translocation case
"""

import os, sys, math, random, re, time
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

from transformers import (
    TrainingArguments, GPTNeoForCausalLM, PreTrainedTokenizerFast,
    Trainer, AutoConfig,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from datasets import load_from_disk, concatenate_datasets

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_TIMEOUT"] = "3600"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"


# =============================================================================
# Globals & config
# =============================================================================

# Mode mix probabilities. Order: T, S.
MODE_MIX_PROBS = (0.30, 0.70)

QUALITY_MOMENTUM = 0.99
QUALITY_FLOOR    = 0.05

SAMPLING_RATE_NOISE = 204800


# =============================================================================
# Helpers
# =============================================================================

def add_gaussian_noise(echo_waveform, noise_level_std_fraction=0.1):
    signal_std = np.std(echo_waveform)
    if signal_std == 0:
        signal_std = 1.0
    noise_std = signal_std * noise_level_std_fraction
    noise = np.random.normal(0, noise_std, echo_waveform.shape).astype(echo_waveform.dtype)
    return np.clip(echo_waveform + noise, -1.0, 1.0)


def fetch_echo_group_for_S(loc_key, hf_echo_dataset, echo_index_map,
                            n_units=3, apply_noise=True):
    """
    For Mode S: return [n_units, 1, 6, 15000]. None on failure.
    """
    indices = echo_index_map.get(loc_key, [])
    if len(indices) < n_units:
        return None
    sampled = random.sample(indices, n_units)
    rows = hf_echo_dataset[sampled]
    arrs = np.array(rows['echo_array'], dtype=np.float32)
    if apply_noise:
        for g in range(arrs.shape[0]):
            for e in range(arrs.shape[1]):
                if random.random() < 0.3:
                    arrs[g, e, :] = add_gaussian_noise(
                        arrs[g, e, :].copy(),
                        noise_level_std_fraction=random.uniform(0.05, 0.25),
                    )
    return torch.tensor(arrs, dtype=torch.float32).unsqueeze(1)


def find_locs_in_text(ids_list, tokenizer, valid_keys_set, pad_token_id, location_pattern):
    """
    Returns (loc_key, token_id) pairs for all LOC tokens in the sequence.
    """
    candidates = []
    for tok_id in ids_list:
        if tok_id == pad_token_id:
            break
        token_str = tokenizer.decode([tok_id])
        m = location_pattern.fullmatch(token_str)
        if m:
            try:
                k = int(m.group(1))
                if k in valid_keys_set:
                    candidates.append((k, tok_id))
            except ValueError:
                pass
    return candidates


# =============================================================================
# Echo encoder
# =============================================================================

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, output_size, flash=True):
        super().__init__()
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
        query = self.q_proj(query).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key   = self.k_proj(key  ).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(value).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        if self.flash:
            output = F.scaled_dot_product_attention(query, key, value)
        else:
            att = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
            att = F.softmax(att, dim=-1)
            output = torch.matmul(att, value)
        output = output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(output)


class BatNavEncoder_attn2(nn.Module):
    def __init__(self, input_size=15000, patch_size=150, in_channels=1,
                 embed_dim=1024, num_encoder_layers=2, num_heads=8):
        super().__init__()
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
            nn.TransformerEncoderLayer(embed_dim, num_heads), num_encoder_layers
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
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        cnn_patches = self.patch_embedding(x)
        if torch.isnan(cnn_patches[0]).any():
            print('Error: cnn_patches has NaN!'); sys.exit(1)
        cnn_patches = cnn_patches.flatten(2).permute(0, 2, 1)
        cnn_patches_pos = cnn_patches + self.positional_embedding[None, :, :].to(cnn_patches.dtype)
        encoded_patches = self.transformer_encoder(cnn_patches_pos.permute(1, 0, 2)).permute(1, 0, 2)
        weights = F.softmax(self.feature_weights, dim=0)
        query = (weights * encoded_patches).sum(dim=1, keepdim=True)
        attn_output = self.echo_attn(query, encoded_patches, cnn_patches)
        enhanced_encoding = attn_output + encoded_patches[:, :1, :]
        prediction = self.output_layer(enhanced_encoding.reshape(x.size(0), -1))
        return prediction


# =============================================================================
# Output container for the LM
# =============================================================================

@dataclass
class TSOutput(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    last_hidden: Optional[torch.FloatTensor] = None    # [B, T, H]
    clf_logits: Optional[torch.FloatTensor] = None     # for Mode S aux loss


# =============================================================================
# Top-level model  (plain GPT-Neo, no fusion injection)
# =============================================================================

class GPTNeoForCausalLM_TS(GPTNeoForCausalLM):
    def __init__(self, config, echo_encoder_config=None):
        super().__init__(config)
        self.cls_token_id = 4
        if echo_encoder_config is None:
            echo_encoder_config = dict(
                input_size=15000, patch_size=150, in_channels=1,
                embed_dim=config.hidden_size, num_encoder_layers=2, num_heads=8,
            )
        self.echo_encoder = BatNavEncoder_attn2(**echo_encoder_config)
        self.echo_embed_dim = echo_encoder_config.get('embed_dim', config.hidden_size)
        H = config.hidden_size
        self.slx_projection = nn.Sequential(
            nn.Linear(H, H), nn.GELU(), nn.Linear(H, H)
        )
        self.init_weights()

    def _echo_path(self, raw_units, device, dtype):
        """
        raw_units: list of N tensors, each [n_units, 1, 6, 15000].
        Returns:
          slx:        [N, H]
          clf_logits: [N, V]
          slx_seq:    [N, M_max, H]
          slx_mask:   [N, M_max]
        """
        N = len(raw_units)
        M = max(t.size(0) for t in raw_units)

        padded   = torch.zeros(N, M, 1, 6, 15000, device=device, dtype=dtype)
        slx_mask = torch.zeros(N, M, device=device, dtype=dtype)
        for n, t in enumerate(raw_units):
            padded[n, :t.size(0)] = t.to(device=device, dtype=dtype)
            slx_mask[n, :t.size(0)] = 1.0

        flat         = padded.view(N * M, 1, 6, 15000)
        encoded_flat = self.echo_encoder(flat)
        encoded      = encoded_flat.view(N, M, self.echo_embed_dim)

        cls_ids = torch.full((N, 1), self.cls_token_id, dtype=torch.long, device=device)
        cls_emb = self.transformer.wte(cls_ids)
        echo_in = torch.cat((cls_emb, encoded), dim=1)
        full_mask = torch.cat([torch.ones(N, 1, device=device, dtype=dtype), slx_mask], dim=1)

        out = self.transformer(
            inputs_embeds=echo_in,
            attention_mask=full_mask,
            use_cache=False,
            return_dict=True,
        )
        h     = out.last_hidden_state       # [N, M+1, H]
        pos_h = h[:, 1:, :]                 # [N, M, H]

        # Attention-pool over real positions.
        pool_w = (pos_h.mean(-1).masked_fill(slx_mask == 0, -1e4)).softmax(-1)
        pooled = (pos_h * pool_w.unsqueeze(-1)).sum(1)   # [N, H]

        clf_logits = self.lm_head(pooled)                # [N, V]
        slx        = self.slx_projection(pooled)         # [N, H]
        slx_seq    = self.slx_projection(pos_h)          # [N, M, H]
        return slx, clf_logits, slx_seq, slx_mask

    def forward(
        self,
        input_ids:      Optional[torch.LongTensor]  = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels:         Optional[torch.LongTensor]  = None,
        **kw,
    ) -> TSOutput:
        out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        lm_logits = self.lm_head(out.last_hidden_state)
        return TSOutput(
            loss=None,
            logits=lm_logits,
            last_hidden=out.last_hidden_state,
            past_key_values=out.past_key_values,
        )


# =============================================================================
# Quality tracker
# =============================================================================

class LocationQualityTracker:
    def __init__(self, momentum=QUALITY_MOMENTUM, init_loss=5.0, floor=QUALITY_FLOOR):
        self.momentum  = momentum
        self.init_loss = init_loss
        self.floor     = floor
        self.mean_loss = defaultdict(lambda: init_loss)
        self.seen      = defaultdict(int)

    def update(self, loc_keys: List[int], losses: List[float]):
        for k, l in zip(loc_keys, losses):
            n = self.seen[k]
            if n < 10:
                self.mean_loss[k] = (n * self.mean_loss[k] + l) / (n + 1)
            else:
                self.mean_loss[k] = self.momentum * self.mean_loss[k] + (1 - self.momentum) * l
            self.seen[k] += 1

    def quality(self, loc_keys: List[int]) -> List[float]:
        return [max(math.exp(-self.mean_loss[k]), self.floor) for k in loc_keys]


# =============================================================================
# Index builder
# =============================================================================

def build_location_index_map(dataset):
    print("Building in-memory index map (location_key -> dataset indices)...")
    index_map = defaultdict(list)
    if 'location_key' not in dataset.column_names:
        raise ValueError("Dataset is missing 'location_key' column.")
    for i, key in enumerate(tqdm(dataset['location_key'], desc="Indexing")):
        index_map[int(key)].append(i)
    print(f"Built index for {len(index_map)} unique locations.")
    return index_map


# =============================================================================
# Two-mode collator  (T + S only)
# =============================================================================

def two_mode_collate_fn(
    batch,
    tokenizer,
    location_map,
    hf_echo_dataset,
    echo_index_map,
    quality_tracker,
    mode_token_ids,     # (bos_id, cls_id)
    sep_token_id,       # [INST]
    pad_token_id,
):
    t0 = time.time()

    bos_id, cls_id = mode_token_ids

    location_pattern = re.compile(r'\[LOC_(\d+)\]')
    valid_keys_set   = set(location_map.keys())
    valid_keys_list  = list(valid_keys_set)

    texts = [example["text"].replace(" ", "") for example in batch]

    base_enc = tokenizer(texts, truncation=True, max_length=1024,
                         padding="max_length", return_tensors="pt")
    base_input_ids      = base_enc["input_ids"]
    base_attention_mask = base_enc["attention_mask"]

    final_input_ids      = base_input_ids.clone()
    final_attention_mask = base_attention_mask.clone()
    final_labels         = torch.full_like(final_input_ids, -100)

    # Mode S bookkeeping
    s_echo_tensors     = []
    s_sample_indices   = []
    s_target_token_ids = []
    s_loc_keys         = []

    mode_tags = []

    for i in range(len(texts)):
        ids_list = base_input_ids[i].tolist()
        mode = random.choices(['T', 'S'], weights=MODE_MIX_PROBS, k=1)[0]

        text_loc_candidates = find_locs_in_text(
            ids_list, tokenizer, valid_keys_set, pad_token_id, location_pattern
        )

        # ---- MODE T ----
        if mode == 'T':
            input_ids = base_input_ids[i].clone()
            input_ids[0] = bos_id
            labels = input_ids.clone()
            row = labels.tolist()
            try:
                idx = row.index(sep_token_id)
                for jj in range(idx + 1):
                    row[jj] = -100
            except ValueError:
                pass
            labels = torch.tensor(row, dtype=labels.dtype)
            labels[final_attention_mask[i] == 0] = -100
            final_input_ids[i] = input_ids
            final_labels[i]    = labels
            mode_tags.append('T')
            continue

        # ---- MODE S ----
        # Pick a target location + fetch echoes.
        chosen_key = None
        chosen_tok = None
        if text_loc_candidates:
            chosen_key, chosen_tok = random.choice(text_loc_candidates)
        elif valid_keys_list:
            k = random.choice(valid_keys_list)
            tok_str = f"[LOC_{str(k).zfill(5)}]"
            t = tokenizer.convert_tokens_to_ids(tok_str)
            if t != tokenizer.unk_token_id:
                chosen_key, chosen_tok = k, t

        echo_tensor = None
        if chosen_key is not None:
            echo_tensor = fetch_echo_group_for_S(chosen_key, hf_echo_dataset, echo_index_map)

        if echo_tensor is None:
            # Downgrade to T if echoes unavailable.
            input_ids = base_input_ids[i].clone()
            input_ids[0] = bos_id
            labels = input_ids.clone()
            row = labels.tolist()
            try:
                idx = row.index(sep_token_id)
                for jj in range(idx + 1):
                    row[jj] = -100
            except ValueError:
                pass
            labels = torch.tensor(row, dtype=labels.dtype)
            labels[final_attention_mask[i] == 0] = -100
            final_input_ids[i] = input_ids
            final_labels[i]    = labels
            mode_tags.append('T')
            continue

        # Build minimal sequence: [CLS] target_token PAD PAD ...
        seq = torch.full((base_input_ids.size(1),), pad_token_id,
                         dtype=base_input_ids.dtype)
        seq[0] = cls_id
        seq[1] = chosen_tok
        attn = torch.zeros_like(seq)
        attn[:2] = 1
        labels = torch.full_like(seq, -100)
        labels[1] = chosen_tok

        final_input_ids[i]      = seq
        final_attention_mask[i] = attn
        final_labels[i]         = labels

        s_echo_tensors.append(echo_tensor)
        s_sample_indices.append(i)
        s_target_token_ids.append(chosen_tok)
        s_loc_keys.append(chosen_key)
        mode_tags.append('S')

    out = {
        "input_ids":           final_input_ids,
        "attention_mask":      final_attention_mask,
        "labels":              final_labels,
        "mode_tags":           mode_tags,
        "s_echo_tensors":      s_echo_tensors,
        "s_sample_indices":    s_sample_indices,
        "s_target_token_ids":  s_target_token_ids,
        "s_loc_keys":          s_loc_keys,
    }

    elapsed = time.time() - t0
    if elapsed > 7.0:
        print(f"[slow collate] {elapsed:.1f}s for batch of {len(batch)}, "
              f"S samples: {len(s_echo_tensors)}")
    return out


# =============================================================================
# Trainer
# =============================================================================

class TSTrainer(Trainer):
    def __init__(self, *args,
                 alpha_s_clf=0.5,
                 quality_tracker=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_s_clf     = alpha_s_clf
        self.quality_tracker = quality_tracker
        self._dbg_step       = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self._dbg_step += 1

        labels             = inputs.pop("labels")
        mode_tags          = inputs.pop("mode_tags", None)
        s_echo_tensors     = inputs.pop("s_echo_tensors", [])
        s_sample_indices   = inputs.pop("s_sample_indices", [])
        s_target_token_ids = inputs.pop("s_target_token_ids", [])
        s_loc_keys         = inputs.pop("s_loc_keys", [])

        device = inputs["input_ids"].device
        dtype  = next(model.parameters()).dtype
        B      = inputs["input_ids"].size(0)

        # ----- 1. Plain LM forward -----
        out       = model(input_ids=inputs["input_ids"],
                          attention_mask=inputs["attention_mask"])
        lm_logits = out.logits    # [B, T, V]

        # ----- 2. Standard LM loss -----
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100, reduction='none',
        ).view(B, -1)
        tok_mask       = (shift_labels != -100).float()
        per_sample_lm  = (ce * tok_mask).sum(1) / tok_mask.sum(1).clamp(min=1.0)
        lm_loss        = per_sample_lm.mean()

        # ----- 3. Mode S aux classifier loss + quality tracker update -----
        s_clf_loss = torch.tensor(0.0, device=device)
        if s_echo_tensors:
            _slx_s, clf_logits_s, _, _ = model._echo_path(s_echo_tensors, device, dtype)
            target_ids = torch.tensor(s_target_token_ids,
                                      dtype=torch.long, device=device)
            s_clf_loss = F.cross_entropy(clf_logits_s, target_ids)

            if self.quality_tracker is not None:
                with torch.no_grad():
                    per_s_loss = F.cross_entropy(clf_logits_s, target_ids,
                                                 reduction='none')
                    self.quality_tracker.update(
                        s_loc_keys, per_s_loss.float().cpu().tolist()
                    )

        total = lm_loss + self.alpha_s_clf * s_clf_loss

        # ----- 4. Logging -----
        if self._dbg_step % 20 == 0 and mode_tags is not None:
            counts = {'T': 0, 'S': 0}
            for m in mode_tags:
                counts[m] = counts.get(m, 0) + 1
            tot = max(sum(counts.values()), 1)

            max_mem_gb  = torch.cuda.max_memory_allocated() / (1024 ** 3)
            curr_mem_gb = torch.cuda.memory_allocated()     / (1024 ** 3)

            print(f"step {self._dbg_step} | LM:{lm_loss.item():.4f} "
                  f"S_clf:{s_clf_loss.item():.4f} | "
                  f"T:{counts['T']/tot:.2f} S:{counts['S']/tot:.2f} | "
                  f"Q-tracked:{len(self.quality_tracker.mean_loss) if self.quality_tracker else 0} | "
                  f"VRAM: Peak {max_mem_gb:.2f}GB (Curr {curr_mem_gb:.2f}GB)")

        return (total, out) if return_outputs else total


# =============================================================================
# Main
# =============================================================================

def main():
    # --- 1. Load echo dataset ---
    echo_dataset_path = "/scratch200/xingchen/processed_echo_dataset_parallel_with_coords_v2"
    if not os.path.exists(echo_dataset_path):
        sys.exit(f"Echo dataset not found at {echo_dataset_path}")
    echo_dataset = load_from_disk(echo_dataset_path)
    print(f"Loaded echo dataset with {len(echo_dataset)} samples.")

    raw_idx_map = build_location_index_map(echo_dataset)
    MIN_SAMPLES = 20
    echo_indices_by_location = {
        k: v for k, v in raw_idx_map.items() if len(v) >= MIN_SAMPLES
    }
    print(f"After filtering >={MIN_SAMPLES} samples/loc: "
          f"{len(echo_indices_by_location)} locations.")

    valid_locs   = sorted(echo_indices_by_location.keys())
    location_map = {loc: i for i, loc in enumerate(valid_locs)}
    if len(location_map) <= 0:
        sys.exit("No valid locations after filtering.")

    # --- 2. Tokenizer ---
    tokenizer_path = "./ops/tokenizer_v9_2.json"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path, max_len=1024,
        pad_token='[PAD]', unk_token='[UNK]', sep_token='[SEP]',
        cls_token='[CLS]', bos_token='[BOS]', eos_token='[EOS]',
        additional_special_tokens=['[INST]', '[SRA]', '[WID]', '[POS]', '[SP_ONE]'],
    )
    bos_id = tokenizer.bos_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.convert_tokens_to_ids('[INST]')
    pad_id = tokenizer.pad_token_id
    print(f"Mode tokens -> [BOS]={bos_id}  [CLS]={cls_id}  [INST]={sep_id}")

    mode_token_ids = (bos_id, cls_id)

    # --- 3. Datasets ---
    dataset_path1 = "/scratch100/xingchen/navigation_code_without_GL/train_dataset_finetune_5400_loc_v2_qc_only"
    dataset_path2 = "/scratch100/xingchen/navigation_code_without_GL/train_dataset_finetune_5400_loc_v2_no_loc_has_mov_v2"
    for p in (dataset_path1, dataset_path2):
        if not os.path.exists(p):
            sys.exit(f"Dataset not found: {p}")
    d1 = load_from_disk(dataset_path1)
    d2 = load_from_disk(dataset_path2)
    dataset   = concatenate_datasets([d1, d2]).shuffle(seed=42)
    total     = len(dataset)
    split_idx = int(total * 0.9995)
    train_dataset = dataset.select(range(0, split_idx))
    eval_dataset  = dataset.select(range(split_idx, total))
    print(f"Train: {len(train_dataset)}   Eval: {len(eval_dataset)}")

    print("Warming page cache...")
    warmup_indices = random.sample(range(len(echo_dataset)),
                                   min(500, len(echo_dataset)))
    for idx in tqdm(warmup_indices, desc="Warmup"):
        _ = echo_dataset[idx]['echo_array']
    print("Cache warm.")

    # --- 4. Model ---
    echo_encoder_config = {
        'input_size': 15000, 'patch_size': 150, 'in_channels': 1,
        'embed_dim': 1024, 'num_encoder_layers': 2, 'num_heads': 8,
    }
    model_path = "/scratch100/xingchen/navigation_code_without_GL/GPT_NEO_finetune_v1"
    if not os.path.exists(model_path):
        sys.exit(f"Base model not found at {model_path}")

    model = GPTNeoForCausalLM_TS.from_pretrained(
        model_path,
        echo_encoder_config=echo_encoder_config,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
    )
    print("Loaded model (T/S only, no fusion injection).")

    quality_tracker = LocationQualityTracker()

    output_dir      = "./GPT_NEO_finetune"
    deepspeed_config = './ops/ds_config3_6_large_lr.json'
    if not os.path.exists(deepspeed_config):
        deepspeed_config = None

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy='steps',
        logging_strategy='steps',
        logging_steps=200,
        eval_steps=200,
        save_strategy='steps',
        save_steps=200,
        save_total_limit=3,
        weight_decay=0.1,
        learning_rate=3.0e-4,
        adam_beta1=0.9, adam_beta2=0.95,
        fp16=False, bf16=True,
        gradient_accumulation_steps=24,
        deepspeed=deepspeed_config,
        dataloader_drop_last=True,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=6,
        max_grad_norm=200.0,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )

    collator = partial(
        two_mode_collate_fn,
        tokenizer=tokenizer,
        location_map=location_map,
        hf_echo_dataset=echo_dataset,
        echo_index_map=echo_indices_by_location,
        quality_tracker=quality_tracker,
        mode_token_ids=mode_token_ids,
        sep_token_id=sep_id,
        pad_token_id=pad_id,
    )

    trainer = TSTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        alpha_s_clf=0.5,
        quality_tracker=quality_tracker,
    )

    print("Starting T/S training...")
    train_result = trainer.train()
    trainer.save_model()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print("Done.")


if __name__ == '__main__':
    main()
