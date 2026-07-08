"""
Extract hidden states from high-confidence files identified by the
confidence-vs-distance evaluation.

Workflow:
  1. Load per_file_results.feather (output of eval_confidence_vs_distance.py)
  2. Filter by configurable confidence range (default conf_avg_softmax in [0.5, 1.0])
  3. Re-parse each surviving file's echo units
  4. Run the echo path -> save the per-unit hidden state sequence
  5. Write one feather per range_label, in the format your downstream MLP
     script expects (location, hidden_state_sequence, mean_azimuth,
     mean_coordinate_x/y, plus confidence as bonus).
"""

import os
import sys
import math
import time
import gc
import json
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPTNeoForCausalLM, AutoConfig, PreTrainedTokenizerFast
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gpt_neo.modeling_gpt_neo import (
    GPTNeoModel, _prepare_4d_causal_attention_mask,
)
from safetensors.torch import load_file


# =============================================================================
# Configuration
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1

# normal model
CHECKPOINT_VERSION = 2000
# CHECKPOINT_PATH = f"./GPT_NEO_finetune_FUSION_v1"#/checkpoint-{CHECKPOINT_VERSION}"

# only echo cls model
CHECKPOINT_PATH = f"./GPT_NEO_ECHO_CLS_ONLY_with_tracker_v1/checkpoint-{CHECKPOINT_VERSION}"
TOKENIZER_PATH  = "./ops/tokenizer_v9_2.json"

# Input: the per-file results table from eval_confidence_vs_distance.py
PER_FILE_RESULTS = f"./only_cls_eval_confidence_vs_distance_ckpt{CHECKPOINT_VERSION}/per_file_results.feather"
# PER_FILE_RESULTS = "./eval_confidence_vs_distance_ckpt15000/per_file_results.feather"


# Output directory for hidden-state feathers
OUTPUT_DIR = f"./map_echo_with_linear_layer/only_cls_smart_1535loc_hidden_states_ckpt{CHECKPOINT_VERSION}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK_SIZE        = 6
NUM_UNITS_PER_FILE = 5
EXPECTED_ECHO_LEN = 15000

BATCH_UNITS = 192
NUM_WORKERS = 16

# Which confidence column to use for filtering. Options:
#   'conf_avg_softmax', 'conf_mean', 'conf_max'
CONFIDENCE_COLUMN = 'conf_avg_softmax'


# =============================================================================
# Model definitions (same as eval script — copied for self-containment)
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
        Q = self.q_proj(query).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key  ).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        if self.flash:
            out = F.scaled_dot_product_attention(Q, K, V)
        else:
            att = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)
            att = F.softmax(att, dim=-1)
            out = torch.matmul(att, V)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(out)


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

    def forward(self, x):
        cnn = self.patch_embedding(x)
        cnn = cnn.flatten(2).permute(0, 2, 1)
        cnn_pos = cnn + self.positional_embedding[None, :, :].to(cnn.dtype)
        enc = self.transformer_encoder(cnn_pos.permute(1, 0, 2)).permute(1, 0, 2)
        w = F.softmax(self.feature_weights, dim=0)
        query = (w * enc).sum(dim=1, keepdim=True)
        attn_out = self.echo_attn(query, enc, cnn)
        enhanced = attn_out + enc[:, :1, :]
        return self.output_layer(enhanced.reshape(x.size(0), -1))


class FusionCrossAttention(nn.Module):
    """Present so loaded state_dict matches; never invoked here."""
    def __init__(self, hidden_size, slx_dim, num_heads=8):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(slx_dim,    hidden_size, bias=False)
        self.v_proj = nn.Linear(slx_dim,    hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Linear(hidden_size, 1)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, slx_memory, presence_mask):
        return torch.zeros_like(hidden_states)


class GPTNeoModel_NoPE_Fusion(GPTNeoModel):
    def __init__(self, config, fusion_layer_idx=8, slx_dim=1024):
        super().__init__(config)
        if hasattr(self, 'wpe'):
            self.wpe = nn.Identity()
        self.fusion_layer_idx = fusion_layer_idx
        self.fusion = FusionCrossAttention(config.hidden_size, slx_dim)

    def forward(self, input_ids=None, past_key_values=None, attention_mask=None,
                token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None,
                slx_memory=None, presence_mask=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("specify input_ids xor inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("specify input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        past_key_values = tuple([None] * len(self.h)) if past_key_values is None else past_key_values
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        hidden_states = inputs_embeds
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        attention_mask = _prepare_4d_causal_attention_mask(attention_mask, input_shape, inputs_embeds, 0)
        hidden_states = self.drop(hidden_states)

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            outputs = block(
                hidden_states, layer_past=layer_past, attention_mask=attention_mask,
                head_mask=head_mask[i], use_cache=False, output_attentions=False,
            )
            hidden_states = outputs[0]
            if (i == self.fusion_layer_idx and slx_memory is not None and presence_mask is not None):
                hidden_states = hidden_states + self.fusion(hidden_states, slx_memory, presence_mask)

        hidden_states = self.ln_f(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class GPTNeoForCausalLM_Fusion(GPTNeoForCausalLM):
    def __init__(self, config, num_aux_classes, echo_encoder_config, fusion_layer_idx=8):
        super().__init__(config)
        self.transformer = GPTNeoModel_NoPE_Fusion(
            config, fusion_layer_idx=fusion_layer_idx, slx_dim=config.hidden_size,
        )
        self.cls_token_id = 4
        self.echo_encoder = BatNavEncoder_attn2(**echo_encoder_config)
        self.echo_embed_dim = echo_encoder_config.get('embed_dim', 1024)
        H = config.hidden_size
        self.slx_projection = nn.Sequential(
            nn.Linear(H, H), nn.GELU(), nn.Linear(H, H),
        )

    @torch.no_grad()
    def echo_extract_hidden_states(self, raw_units: torch.Tensor) -> torch.Tensor:
        """
        Run the echo path to extract the sequence of hidden states for each
        echo unit. This is the same forward pass as echo_classify_units,
        but instead of returning lm_head logits we return the pre-head
        hidden states corresponding to the 5 echo positions.

        raw_units: [B, 1, 6, 15000]  — a batch of 6-echo units.
        Returns:   [B, 5, H]          — hidden states at echo positions 1..5.
        """
        device = raw_units.device
        dtype = self.transformer.wte.weight.dtype
        raw_units = raw_units.to(device=device, dtype=dtype)

        B = raw_units.size(0)
        encoded = self.echo_encoder(raw_units)                       # [B, E]
        encoded_seq = encoded.unsqueeze(1).expand(-1, 5, -1)         # [B, 5, E]

        cls_ids = torch.full((B, 1), self.cls_token_id, dtype=torch.long, device=device)
        cls_emb = self.transformer.wte(cls_ids)
        echo_inputs = torch.cat((cls_emb, encoded_seq), dim=1)       # [B, 6, H]

        out = self.transformer(
            inputs_embeds=echo_inputs, use_cache=False, return_dict=True,
            slx_memory=None, presence_mask=None,
        )
        # Return hidden states for positions 1..5 (the 5 echo positions),
        # skipping the [CLS] at position 0.
        hs = out.last_hidden_state[:, 1:, :]                         # [B, 5, H]
        return hs


class ParallelHiddenStateExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, raw_units):
        return self.model.echo_extract_hidden_states(raw_units)


# =============================================================================
# Helpers
# =============================================================================

def circular_mean(azimuths_deg):
    if not len(azimuths_deg):
        return float('nan')
    az = np.asarray(azimuths_deg, dtype=float)
    az = az[~np.isnan(az)]
    if not len(az):
        return float('nan')
    az_rad = np.deg2rad(az % 360.0)
    return float((np.rad2deg(np.arctan2(np.sin(az_rad).mean(), np.cos(az_rad).mean())) + 360) % 360)


def parse_coordinate(val):
    if isinstance(val, str):
        try:
            parts = val.strip('[]').split(',')
            if len(parts) == 2:
                return float(parts[0].strip()), float(parts[1].strip())
        except Exception:
            return None
    elif isinstance(val, (list, np.ndarray)) and len(val) == 2:
        try:
            return float(val[0]), float(val[1])
        except Exception:
            return None
    return None


def parse_file(file_path: str) -> Optional[Dict]:
    """Read a feather file; return units + metadata. Same logic as eval script."""
    try:
        df = pd.read_feather(file_path)
    except Exception:
        return None
    if 'echoes' not in df.columns:
        return None

    n_rows = len(df)
    if n_rows < CHUNK_SIZE:
        return None

    units = []
    for start in range(0, n_rows - CHUNK_SIZE + 1, CHUNK_SIZE):
        if len(units) >= NUM_UNITS_PER_FILE:
            break
        block = df.iloc[start : start + CHUNK_SIZE]
        if len(block) != CHUNK_SIZE:
            break
        try:
            echoes = [np.clip(np.array(e, dtype=np.float32) / 1000.0, -1.0, 1.0)
                      for e in block['echoes'].tolist()]
        except Exception:
            continue
        if any(e.shape != (EXPECTED_ECHO_LEN,) for e in echoes):
            continue
        units.append(np.stack(echoes, axis=0))

    if not units:
        return None

    coord_x_list, coord_y_list = [], []
    if 'coordinate' in df.columns:
        for v in df['coordinate'].tolist():
            xy = parse_coordinate(v)
            if xy is not None:
                coord_x_list.append(xy[0])
                coord_y_list.append(xy[1])
    centroid_x = float(np.mean(coord_x_list)) if coord_x_list else float('nan')
    centroid_y = float(np.mean(coord_y_list)) if coord_y_list else float('nan')

    mean_az = float('nan')
    if 'azimuth' in df.columns:
        az_vals = pd.to_numeric(df['azimuth'], errors='coerce').dropna().values
        mean_az = circular_mean(az_vals)

    return {
        'units': units,
        'n_units': len(units),
        'centroid_x': centroid_x,
        'centroid_y': centroid_y,
        'mean_azimuth': mean_az,
    }


# =============================================================================
# Filtering function (the configurable part)
# =============================================================================

def filter_per_file_table(
    per_file_df: pd.DataFrame,
    confidence_range: Tuple[float, float] = (0.5, 1.0),
    confidence_column: str = CONFIDENCE_COLUMN,
    require_n_units: int = 5,
    extra_filters: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Filter the per-file results table.

    Args:
      per_file_df:        the DataFrame loaded from per_file_results.feather
      confidence_range:   (lo, hi) — keep rows with lo <= confidence_column <= hi
      confidence_column:  which column to filter on
      require_n_units:    minimum number of echo units required (default 5)
      extra_filters:      optional dict of {col_name: (op, value)} for further
                          filtering, e.g. {'agreement_rate': ('>=', 0.6)}

    Returns:
      Filtered DataFrame.
    """
    df = per_file_df.copy()
    n0 = len(df)

    # Confidence range
    lo, hi = confidence_range
    df = df[(df[confidence_column] >= lo) & (df[confidence_column] <= hi)]
    print(f"  After confidence {confidence_column} in [{lo}, {hi}]: {len(df)} rows "
          f"(was {n0}).")

    # Minimum units
    df = df[df['n_units'] >= require_n_units]
    print(f"  After n_units >= {require_n_units}: {len(df)} rows.")

    # Drop rows with any NaN in critical columns
    critical = [confidence_column, 'centroid_x', 'centroid_y', 'mean_azimuth',
                'source_file', 'loc_id', 'range_label']
    before = len(df)
    df = df.dropna(subset=critical)
    if len(df) < before:
        print(f"  After dropping NaN in critical cols: {len(df)} rows "
              f"(dropped {before - len(df)}).")

    # Extra user-supplied filters
    if extra_filters:
        for col, (op, val) in extra_filters.items():
            if col not in df.columns:
                print(f"  Warning: extra filter column '{col}' not found, skipping.")
                continue
            if op == '>=':  df = df[df[col] >= val]
            elif op == '<=':df = df[df[col] <= val]
            elif op == '>': df = df[df[col] >  val]
            elif op == '<': df = df[df[col] <  val]
            elif op == '==':df = df[df[col] == val]
            elif op == '!=':df = df[df[col] != val]
            else:
                print(f"  Warning: unsupported op '{op}' for {col}, skipping.")
                continue
            print(f"  After {col} {op} {val}: {len(df)} rows.")

    return df.reset_index(drop=True)


# =============================================================================
# Resolve filtered rows back to absolute file paths
# =============================================================================

def resolve_file_paths(filtered_df: pd.DataFrame,
                       data_ranges: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    The per-file table has source_dir + source_file but not the absolute path.
    Reconstruct the absolute path from range_label + DATA_RANGES mapping.
    """
    range_to_base = dict(data_ranges)
    paths = []
    for _, row in filtered_df.iterrows():
        base = range_to_base.get(row['range_label'])
        if base is None:
            paths.append(None)
            continue
        path = os.path.join(base, row['source_dir'], row['source_file'])
        paths.append(path)
    out = filtered_df.copy()
    out['file_path'] = paths
    out = out.dropna(subset=['file_path']).reset_index(drop=True)

    # Verify a few exist
    n_check = min(5, len(out))
    n_exists = sum(os.path.exists(out.iloc[i]['file_path']) for i in range(n_check))
    print(f"  Verified {n_exists}/{n_check} sample paths exist on disk.")
    if n_exists == 0 and n_check > 0:
        print(f"  WARNING: First sample path: {out.iloc[0]['file_path']}")
        print(f"           Check that DATA_RANGES base directories are correct.")

    return out


# =============================================================================
# Hidden state extraction
# =============================================================================

@torch.no_grad()
def extract_hidden_states_for_filtered(
    model_wrapper, filtered_df: pd.DataFrame,
    chunk_size_files: int = 26000,
):
    """
    Re-parse each filtered file, run echo path, save hidden states.
    Returns a list of dicts ready to be turned into a DataFrame.
    """
    model_wrapper.eval()
    rows_with_hidden = []
    n_total = len(filtered_df)
    print(f"\nExtracting hidden states for {n_total} files...")

    for chunk_start in range(0, n_total, chunk_size_files):
        chunk_end = min(chunk_start + chunk_size_files, n_total)
        chunk_df = filtered_df.iloc[chunk_start:chunk_end].reset_index(drop=True)
        chunk_idx = chunk_start // chunk_size_files + 1
        n_chunks = (n_total + chunk_size_files - 1) // chunk_size_files
        print(f"\n--- Chunk {chunk_idx}/{n_chunks} | rows {chunk_start} to {chunk_end} ---")

        # Parse files in parallel
        file_paths = chunk_df['file_path'].tolist()
        parsed_files = []
        with Pool(processes=NUM_WORKERS) as pool:
            for parsed in tqdm(pool.imap(parse_file, file_paths, chunksize=50),
                                total=len(file_paths), desc="Parse"):
                parsed_files.append(parsed)

        # Build flat list of (chunk_row_idx, unit_idx, unit_array) for batching
        pending = []
        for i, parsed in enumerate(parsed_files):
            if parsed is None:
                continue
            for u_idx, u in enumerate(parsed['units']):
                pending.append((i, u_idx, u))

        if not pending:
            del parsed_files, pending; gc.collect()
            continue

        # Per-row accumulators: list of [5, H] tensors per unit
        per_row_hidden = [[None] * NUM_UNITS_PER_FILE for _ in range(len(chunk_df))]

        for start in tqdm(range(0, len(pending), BATCH_UNITS), desc="Extract hidden"):
            batch = pending[start : start + BATCH_UNITS]
            if not batch:
                break
            units_np = np.stack([item[2] for item in batch], axis=0)
            units_t = torch.from_numpy(units_np).unsqueeze(1).pin_memory().to(
                DEVICE, non_blocking=True
            )

            hidden = model_wrapper(units_t)              # [B, 5, H]
            hidden_cpu = hidden.float().cpu().numpy()    # [B, 5, H]

            for j, (row_idx, u_idx, _u) in enumerate(batch):
                per_row_hidden[row_idx][u_idx] = hidden_cpu[j]   # [5, H]

            # Free the unit arrays we just consumed
            for k in range(start, min(start + BATCH_UNITS, len(pending))):
                pending[k] = None
            del units_np, units_t, hidden, hidden_cpu

        # Free original unit arrays
        for parsed in parsed_files:
            if parsed is not None:
                parsed['units'] = None

        # Build output rows for this chunk
        for i, parsed in enumerate(parsed_files):
            if parsed is None:
                continue
            unit_hiddens = per_row_hidden[i]
            valid_units = [u for u in unit_hiddens if u is not None]
            if not valid_units:
                continue

            # Concatenate hidden states across units along the sequence dim:
            # each unit gives [5, H]; if we have 5 units that's [25, H].
            # The downstream MLP script takes the LAST k states (k = 2/5 * S),
            # so order matters. We concatenate units in their natural order.
            full_seq = np.concatenate(valid_units, axis=0).astype(np.float32)  # [N*5, H]

            row = chunk_df.iloc[i]
            rows_with_hidden.append({
                'location_id':         int(row['loc_id']),
                'source_file':         row['source_file'],
                'range_label':         row['range_label'],
                'hidden_state_sequence': full_seq,        # [seq_len, H]
                'mean_azimuth':        parsed['mean_azimuth'],
                'mean_coordinate_x':   parsed['centroid_x'],
                'mean_coordinate_y':   parsed['centroid_y'],
                'confidence':          float(row[CONFIDENCE_COLUMN]),
                'conf_mean':           float(row.get('conf_mean', np.nan)),
                'conf_max':            float(row.get('conf_max', np.nan)),
                'conf_avg_softmax':    float(row.get('conf_avg_softmax', np.nan)),
                'agreement_rate':      float(row.get('agreement_rate', np.nan)),
                'distance_m':          float(row.get('distance_m', np.nan)),
                'mean_entropy':        float(row.get('mean_entropy', np.nan)),
            })

        # Cleanup
        del parsed_files, pending, per_row_hidden
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rows_with_hidden


# =============================================================================
# Main
# =============================================================================

def main(
    confidence_range: Tuple[float, float] = (0.5, 1.0),
    confidence_column: str = CONFIDENCE_COLUMN,
    extra_filters: Optional[Dict] = None,
    output_suffix: Optional[str] = None,
):
    """
    Args:
      confidence_range: tuple (lo, hi)
      confidence_column: which conf metric to filter on
      extra_filters:    e.g. {'agreement_rate': ('>=', 0.6)}
      output_suffix:    e.g. 'high_conf' produces files like
                        hidden_states_near_0_80_high_conf.feather
    """
    t0 = time.time()

    # --- Load tokenizer (needed only if we save token-level metadata; kept for parity) ---
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=TOKENIZER_PATH, max_len=1024,
        pad_token='[PAD]', unk_token='[UNK]', sep_token='[SEP]',
        cls_token='[CLS]', bos_token='[BOS]', eos_token='[EOS]',
        additional_special_tokens=['[INST]', '[SRA]', '[WID]', '[POS]', '[SP_ONE]'],
    )

    # --- Load per-file results ---
    if not os.path.exists(PER_FILE_RESULTS):
        sys.exit(f"per_file_results.feather not found at {PER_FILE_RESULTS}. "
                 "Run eval_confidence_vs_distance.py first.")
    per_file_df = pd.read_feather(PER_FILE_RESULTS)
    print(f"Loaded {len(per_file_df)} rows from {PER_FILE_RESULTS}")
    print(f"Columns: {per_file_df.columns.tolist()}")
    print(f"\nConfidence column statistics ({confidence_column}):")
    print(per_file_df[confidence_column].describe())

    # --- Filter ---
    print(f"\nApplying filters (conf in {confidence_range})...")
    filtered = filter_per_file_table(
        per_file_df,
        confidence_range=confidence_range,
        confidence_column=confidence_column,
        extra_filters=extra_filters,
    )
    if filtered.empty:
        sys.exit("No rows remain after filtering. Check thresholds.")

    print(f"\nFiltered breakdown by range_label:")
    print(filtered.groupby('range_label').size())

    # --- Resolve file paths ---
    DATA_RANGES = [
        ("near_0_80",  "/home/xingchen/Projects/navigation_code_without_GL/echo_dataset/loc_923_entropy"),
        ("far_60_120", "/home/xingchen/Projects/navigation_code_without_GL/echo_dataset/echo_923_wide_range_entropy"),
    ]
    filtered = resolve_file_paths(filtered, DATA_RANGES)
    print(f"After path resolution: {len(filtered)} rows.")

    # --- Load model ---
    if not os.path.exists(CHECKPOINT_PATH):
        sys.exit(f"Checkpoint not found at {CHECKPOINT_PATH}")
    config = AutoConfig.from_pretrained(CHECKPOINT_PATH)
    echo_encoder_config = {
        'input_size': EXPECTED_ECHO_LEN, 'patch_size': 150, 'in_channels': 1,
        'embed_dim': 1024, 'num_encoder_layers': 2, 'num_heads': 8,
    }
    num_aux_classes = getattr(config, 'vocab_size', 1535)
    model = GPTNeoForCausalLM_Fusion(
        config, num_aux_classes=num_aux_classes,
        echo_encoder_config=echo_encoder_config, fusion_layer_idx=8,
    )
    safetensor = os.path.join(CHECKPOINT_PATH, 'model.safetensors')
    pytorch_bin = os.path.join(CHECKPOINT_PATH, 'pytorch_model.bin')
    if os.path.exists(safetensor):
        state_dict = load_file(safetensor, device='cpu')
    elif os.path.exists(pytorch_bin):
        state_dict = torch.load(pytorch_bin, map_location='cpu')
    else:
        sys.exit(f"No weights in {CHECKPOINT_PATH}")
    msd = model.state_dict()
    filtered_sd = {k: v for k, v in state_dict.items()
                   if k in msd and v.shape == msd[k].shape}
    res = model.load_state_dict(filtered_sd, strict=False)
    print(f"Model loaded. missing={len(res.missing_keys)} "
          f"unexpected={len(res.unexpected_keys)}")
    model.to(DEVICE).eval()

    parallel_model = ParallelHiddenStateExtractor(model)
    if NUM_GPUS > 1:
        parallel_model = nn.DataParallel(parallel_model)
        print(f"Using DataParallel across {NUM_GPUS} GPUs.")

    # --- Extract hidden states ---
    rows_with_hidden = extract_hidden_states_for_filtered(parallel_model, filtered)
    print(f"\nExtracted hidden states for {len(rows_with_hidden)} files.")

    if not rows_with_hidden:
        sys.exit("No hidden states extracted.")

    # --- Save: one feather per range_label ---
    suffix = f"_{output_suffix}" if output_suffix else ""
    suffix += f"_conf{confidence_range[0]:.2f}_{confidence_range[1]:.2f}"

    df_out = pd.DataFrame(rows_with_hidden)
    # Convert numpy arrays to lists so feather can store them
    df_out['hidden_state_sequence'] = df_out['hidden_state_sequence'].apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    )

    for range_label, sub in df_out.groupby('range_label'):
        out_path = os.path.join(
            OUTPUT_DIR,
            f"hidden_states_{range_label}{suffix}.feather"
        )
        sub_out = sub.reset_index(drop=True)
        try:
            sub_out.to_feather(out_path)
            print(f"  Saved {len(sub_out)} rows -> {out_path}")
        except Exception as e:
            print(f"  Feather save failed ({e}), saving as parquet...")
            sub_out.to_parquet(out_path.replace('.feather', '.parquet'))

    # # Combined file too, for convenience
    # combined_path = os.path.join(OUTPUT_DIR, f"hidden_states_combined{suffix}.feather")
    # try:
    #     df_out.to_feather(combined_path)
    #     print(f"  Saved combined {len(df_out)} rows -> {combined_path}")
    # except Exception as e:
    #     print(f"  Combined feather save failed: {e}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


# =============================================================================
# Quick run-time interface
# =============================================================================

if __name__ == '__main__':
    # Default: confidence > 0.5 on conf_avg_softmax.
    # main(confidence_range=(0, 0.5))
    main(confidence_range=(0, 1))

    # Other example invocations (uncomment to run):
    #
    # # Only very high confidence:
    # main(confidence_range=(0.7, 1.0), output_suffix='very_high')
    #
    # # Mid-confidence band, requiring high agreement among the 5 units:
    # main(
    #     confidence_range=(0.5, 0.8),
    #     extra_filters={'agreement_rate': ('>=', 0.6)},
    #     output_suffix='mid_high_agreement',
    # )
    #
    # # Low-confidence baseline for comparison:
    # main(confidence_range=(0.0, 0.3), output_suffix='low_baseline')