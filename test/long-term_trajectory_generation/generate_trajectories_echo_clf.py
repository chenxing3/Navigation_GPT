import os
import sys
import math
import random
from typing import Callable, List, Dict, Tuple, Optional, Union
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import cKDTree
import pyproj
from pyproj import Transformer
import json

from transformers import TrainingArguments, GPTNeoForCausalLM, PreTrainedTokenizerFast, AutoConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoPreTrainedModel, GPTNeoModel, GPTNeoBlock, _prepare_4d_causal_attention_mask
import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache


# --- Constants ---
AZIMUTH_STEP_DISTANCE = 5.0  # meters per Azi_ token
TARGET_PROXIMITY_THRESHOLD = 100.0 # meters
MIDDLE_LOC_PROXIMITY_THRESHOLD = 30.0 # meters
MAX_REFINEMENTS = 6
NUM_SAMPLES = 10 # How many start/end pairs to try


# =============================================================================
# Echo MLP classifier  (same architecture as map_echo_with_linear_layer)
# =============================================================================

class LocationClassifierModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, input_dim // 2)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.linear2 = nn.Linear(input_dim // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)


def load_echo_classifier(
    model_pth_path: str,
    mapping_json_path: str,
    device: torch.device,
) -> Tuple[LocationClassifierModel, Dict[int, int], Dict[int, int]]:
    """
    Load the MLP classifier and its index<->location-id mappings.
    Returns  clf, index_to_loc_id, loc_id_to_index
    """
    with open(mapping_json_path, 'r') as f:
        raw = json.load(f)
    index_to_loc_id = {int(k): int(v) for k, v in raw.items()}
    loc_id_to_index = {v: k for k, v in index_to_loc_id.items()}
    num_classes     = len(index_to_loc_id)

    state     = torch.load(model_pth_path, map_location='cpu')
    input_dim = state['linear1.weight'].shape[1]   # auto-detect

    clf = LocationClassifierModel(input_dim, num_classes)
    clf.load_state_dict(state)
    clf.to(device).eval()
    print(f"[EchoClf] Loaded  input_dim={input_dim}  num_classes={num_classes}")
    return clf, index_to_loc_id, loc_id_to_index


def predict_location_from_hidden_states(
    hidden_state_sequence: np.ndarray,   # [S, H]  last-layer states for the sequence
    clf: LocationClassifierModel,
    index_to_loc_id: Dict[int, int],
    device: torch.device,
    sub_seq_fraction: float = 2 / 5,
) -> Tuple[int, float]:
    """
    Run the MLP on the last k hidden states (same sub-sequence logic as
    training) and return (predicted_loc_id, confidence).
    """
    S, H = hidden_state_sequence.shape
    k    = max(1, math.ceil(S * sub_seq_fraction))
    k    = min(k, S)

    sub_seq = hidden_state_sequence[-k:, :]          # [k, H]
    x       = torch.tensor(sub_seq, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits    = clf(x)                           # [k, num_classes]
        avg_probs = torch.softmax(logits, dim=-1).mean(dim=0)   # [num_classes]

    top_idx    = int(avg_probs.argmax().item())
    confidence = float(avg_probs[top_idx].item())
    return index_to_loc_id[top_idx], confidence


# --- Helper Functions --- (ALL UNCHANGED from original)

def get_coordinates(token_text: str, hex_centers_df: pd.DataFrame) -> Optional[Tuple[float, float]]:
    """Fetches (x, y) for a given LOC_ or GRID_ token from hex_centers_df."""
    match = hex_centers_df[hex_centers_df['text'] == token_text]
    if not match.empty:
        return match['x'].iloc[0], match['y'].iloc[0]
    return None

class GPTNeoModel_NoPE(GPTNeoModel):
    def __init__(self, config):
        super().__init__(config)
        if hasattr(self, 'wpe'):
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
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
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
        elif hasattr(past_key_values, "get_seq_length"):
             past_length = past_key_values.get_seq_length()
             if past_length is None: past_length = 0
        else:
            try:
                first_layer = past_key_values[0]
                first_item = first_layer[0]
                if isinstance(first_item, torch.Tensor):
                    past_length = first_item.size(-2)
                elif isinstance(first_item, (tuple, list)):
                    past_length = first_item[0].size(-2)
                else:
                    past_length = 0
            except (IndexError, AttributeError):
                past_length = 0

        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        head_mask = self.get_head_mask(head_mask, self.config.num_layers)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        if hasattr(self, '_use_flash_attention_2') and self._use_flash_attention_2:
             attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
             attention_mask = _prepare_4d_causal_attention_mask(attention_mask, input_shape, inputs_embeds, past_length)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, block in enumerate(self.h):
            if past_key_values is None:
                layer_past = None
            elif hasattr(past_key_values, "get_seq_length"):
                layer_past = past_key_values
            elif isinstance(past_key_values, (tuple, list)):
                layer_past = past_key_values[i]
            else:
                layer_past = None

            if output_hidden_states: all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(block.__call__, hidden_states, None, attention_mask, head_mask[i], use_cache, output_attentions)
            else:
                outputs = block(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask[i], use_cache=use_cache, output_attentions=output_attentions)

            hidden_states = outputs[0]

            if use_cache:
                presents = presents + (outputs[1],)

            if output_attentions: all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states: all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict: return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=presents, hidden_states=all_hidden_states, attentions=all_self_attentions)


class GPTNeoForCausalLM_NoPE(GPTNeoForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPTNeoModel_NoPE(config)


# --- ALL GENERATION FUNCTIONS UNCHANGED from original ---

def generate_path_plan(model, tokenizer, prompt_text, force_start_location, device, **kwargs):

    prompt_text = prompt_text + force_start_location
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask = inputs.attention_mask.to(device),
            max_new_tokens=500,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs
        )

    force_prompt = []
    for my_idx, my_id in enumerate(generated_ids[0]):
        token = tokenizer.decode([my_id])
        force_prompt.append(token)
        if token == "[POS]":
            break

    return force_prompt


def generate_azi_traj(model, tokenizer, prompt_texts, force_start_location, device, **kwargs):

    my_length = len(prompt_texts)

    prompt_text = "".join(prompt_texts) + force_start_location

    print("generate_azi_traj:" , prompt_text)

    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask = inputs.attention_mask.to(device),
            max_new_tokens=1000-my_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs
        )

    generated_token_strs = []
    tmp = ""
    for my_idx, my_id in enumerate(generated_ids[0]):

        token = tokenizer.decode([my_id])
        if "[" in token:
            generated_token_strs.append(token)

            tmps = tmp.split("_")
            for ss in tmps:
                if len(ss) > 0:
                    generated_token_strs.append(ss)
            tmp = ""
        else:
            tmp = tmp + token

        if len(tmp) != 0 and len(generated_ids[0]) == my_idx+1:
            tmps = tmp.split("_")
            for ss in tmps:
                if len(ss) > 0:
                    generated_token_strs.append(ss)
            tmp = ""

    return generated_token_strs


def generate_azi_traj_and_hidden_states(model, tokenizer, prompt_texts, force_start_location, device, **kwargs):
    """
    Same as generate_azi_traj but also returns the last-layer hidden states
    for the generated portion as np.ndarray [T, H].

    We run a second forward pass over the already-generated sequence so that
    generate_azi_traj itself is completely untouched.
    """
    # Step 1: generate tokens exactly as before
    generated_token_strs = generate_azi_traj(
        model, tokenizer, prompt_texts, force_start_location, device, **kwargs
    )

    # Step 2: re-encode the full generated text and do ONE forward pass to
    #         collect hidden states — no sampling, no changed behaviour.
    full_text = "".join(prompt_texts) + force_start_location + "".join(generated_token_strs)
    enc       = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(device)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=enc.attention_mask.to(device),
            output_hidden_states=True,
            return_dict=True,
        )
    # last layer hidden states: [1, T_full, H]  -> [T_full, H]
    hidden_states_np = out.hidden_states[-1][0].float().cpu().numpy()

    return generated_token_strs, hidden_states_np


def parse_model_segment_output(
    raw_full_token_strings: List[str],
    prompt_token_len: int
    ) -> Dict:

    generated_tokens = raw_full_token_strings[prompt_token_len:]

    result = {
        'locs_before_pos': [],
        'azi_calc_start_loc': None,
        'post_pos_loc_sequence': [],
        'explicit_final_loc': None,
        'eos_reached': False
    }

    pos_idx = generated_tokens.index("[POS]")

    for i in range(pos_idx):
        token = generated_tokens[i]
        if token.startswith("[LOC_"):
            result['locs_before_pos'].append(token)

    azi_start_loc_found_idx = -1
    for i in range(pos_idx + 1, len(generated_tokens)):
        token = generated_tokens[i]
        if token == "[EOS]":
            result['eos_reached'] = True
            break

        if "LOC_" in token:
            result['azi_calc_start_loc'] = token
            azi_start_loc_found_idx = i
            break

    eos_boundary_idx = len(generated_tokens)
    last_relevant_token_before_eos_was_loc = None
    for i in range(azi_start_loc_found_idx + 1, eos_boundary_idx):
        token = generated_tokens[i]
        parsed_item = None
        current_item_is_loc = False

        if token.isdigit():
            angle = float(token)
            parsed_item = {'type': 'AZI', 'value': angle, 'raw_token': token}
        elif "LOC_" in token:
            parsed_item = {'type': 'LOC', 'value': token}
            current_item_is_loc = True

        if parsed_item:
            result['post_pos_loc_sequence'].append(parsed_item)
            if current_item_is_loc:
                last_relevant_token_before_eos_was_loc = token
            else:
                last_relevant_token_before_eos_was_loc = None

    if result['eos_reached'] and last_relevant_token_before_eos_was_loc:
        result['explicit_final_loc'] = last_relevant_token_before_eos_was_loc
        if result['post_pos_loc_sequence'] and \
           result['post_pos_loc_sequence'][-1]['type'] == 'LOC' and \
           result['post_pos_loc_sequence'][-1]['value'] == result['explicit_final_loc']:
            result['post_pos_loc_sequence'].pop()

    return result

def calculate_azimuth_path(
    start_xy: Tuple[float, float],
    post_pos_loc_sequence: List[Dict],
    step_distance: float = AZIMUTH_STEP_DISTANCE
    ) -> List[Tuple[float, float]]:
    path_points = [start_xy]
    current_x, current_y = start_xy

    for item in post_pos_loc_sequence:
        if item['type'] == 'AZI':
            angle_azi = item['value']
            math_angle_deg = 90.0 - angle_azi
            math_angle_rad = math.radians(math_angle_deg)
            dx = step_distance * math.cos(math_angle_rad)
            dy = step_distance * math.sin(math_angle_rad)
            current_x += dx
            current_y += dy
            path_points.append((current_x, current_y))
    return path_points


def get_path_min_dist_to_point(
    path_coords: List[Tuple[float, float]],
    target_xy: Tuple[float, float]
    ) -> Tuple[float, int]:
    if not path_coords:
        return float('inf'), -1
    distances = [np.linalg.norm(np.array(pt) - np.array(target_xy)) for pt in path_coords]
    min_dist = min(distances)
    idx = distances.index(min_dist)
    return min_dist, idx


def collect_tokens_for_segment(
    parsed_segment_data: Dict,
    num_azi_steps_to_keep: int,
    ) -> List[str]:
    path_plan_part = parsed_segment_data['locs_before_pos']
    traj_tokens = []

    if parsed_segment_data['azi_calc_start_loc']:
        traj_tokens.append(parsed_segment_data['azi_calc_start_loc'])
        azi_count = 0
        for item in parsed_segment_data['post_pos_loc_sequence']:
            traj_tokens.append(item['raw_token'] if item['type'] == 'AZI' else item['value'])
            if item['type'] == 'AZI':
                azi_count += 1
                if azi_count >= num_azi_steps_to_keep:
                    break

    return path_plan_part, traj_tokens


# =============================================================================
# CHANGED: find_best_middle_loc_contact — precision table replaced by clf
# =============================================================================

def find_best_middle_loc_contact(
    segment_path_points: List[Tuple[float, float]],
    middle_loc_ids_texts: List[str],
    hex_centers_df: pd.DataFrame,
    middle_loc_tree: cKDTree,
    radius_m: float,
    overall_target_xy: Tuple[float, float],
    refinement_attempt: int,
    current_start_loc_text: str,
    current_distance: float,
    # --- replaced parameters (precision table removed) ---
    hidden_state_sequence: np.ndarray,      # [S, H] from generate_azi_traj_and_hidden_states
    echo_clf: LocationClassifierModel,
    index_to_loc_id: Dict[int, int],
    clf_device: torch.device,
) -> Optional[Dict]:
    """
    Replaces the precision-table gate with the echo MLP classifier.

    For every trajectory step that is within radius_m of a candidate location,
    we slice the hidden states UP TO THAT STEP and ask the classifier
    independently — so every step in the zone gets its own prediction.
    A candidate is accepted only if the per-step prediction matches it.
    """
    candidate_cuts = []
    if not segment_path_points:
        return None

    # hidden_state_sequence is [T_full, H] covering the whole segment.
    # segment_path_points[i] corresponds to hidden_state_sequence[i] because
    # each AZI token produces one path point and one hidden state row.
    T_hidden = hidden_state_sequence.shape[0]

    for traj_idx, traj_pt_xy in enumerate(segment_path_points):
        nearby_middle_indices_in_tree = middle_loc_tree.query_ball_point(traj_pt_xy, r=radius_m)

        if not nearby_middle_indices_in_tree:
            continue

        # --- per-step classifier query: use hidden states up to this step ---
        step_hidden = hidden_state_sequence[:min(traj_idx + 1, T_hidden)]
        predicted_loc_id, confidence = predict_location_from_hidden_states(
            step_hidden, echo_clf, index_to_loc_id, clf_device
        )

        for middle_idx_in_tree in nearby_middle_indices_in_tree:
            middle_loc_text_contacted = middle_loc_ids_texts[middle_idx_in_tree]

            if middle_loc_text_contacted == current_start_loc_text:
                continue

            # parse integer id from e.g. "[LOC_00123]"
            try:
                loc_int_id = int(middle_loc_text_contacted.strip('[]').split('_')[1])
            except (IndexError, ValueError):
                continue

            # --- gate: accept only if this step's prediction matches this candidate ---
            if loc_int_id != predicted_loc_id:
                continue

            middle_loc_contacted_xy = get_coordinates(middle_loc_text_contacted, hex_centers_df)
            if middle_loc_contacted_xy is None:
                continue

            dist_middle_to_overall_target = np.linalg.norm(
                np.array(middle_loc_contacted_xy) - np.array(overall_target_xy)
            )

            print(f"    [EchoClf] step {traj_idx}: matched {middle_loc_text_contacted} "
                  f"(predicted_loc_id={predicted_loc_id}, confidence={confidence:.3f})")

            candidate_cuts.append({
                "traj_step_idx": traj_idx,
                "middle_loc_text": middle_loc_text_contacted,
                "middle_loc_xy": middle_loc_contacted_xy,
                "dist_from_middle_to_overall_target": dist_middle_to_overall_target,
                "clf_confidence": confidence,
            })

    if not candidate_cuts:
        print(f"    [EchoClf] No nearby candidate matched predicted loc_id={predicted_loc_id} — regenerating.")
        return None

    # drop candidates that don't make progress toward the target
    all_above_current_distance = all(
        x["dist_from_middle_to_overall_target"] > current_distance + 150
        for x in candidate_cuts
    )
    if all_above_current_distance:
        return None

    random.shuffle(candidate_cuts)
    if refinement_attempt != 1:
        candidate_cuts.sort(key=lambda x: (x["dist_from_middle_to_overall_target"], -x["traj_step_idx"]))

    return candidate_cuts[0]


# --- Main `run` function ---
def run():
    if not os.path.exists(TOKENIZER_PATH):
        print("Tokenizer missing"); sys.exit(1)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=TOKENIZER_PATH, max_len=1024, pad_token='[PAD]',
        unk_token='[UNK]', sep_token='[SEP]', cls_token='[CLS]',
        bos_token='[BOS]', eos_token='[EOS]',
        additional_special_tokens=['[INST]', '[SRA]', '[WID]', '[POS]']
    )

    model_load_start_time = time.time()
    model = GPTNeoForCausalLM_NoPE.from_pretrained(CHECKPOINT_PATH, use_safetensors=True, torch_dtype=torch.bfloat16).to(DEVICE).eval()
    print(f"Model loaded from {CHECKPOINT_PATH} to {DEVICE}. Time: {time.time() - model_load_start_time:.2f}s")

    # --- Load echo classifier (replaces precision table) ---
    if not os.path.exists(ECHO_CLF_PTH) or not os.path.exists(ECHO_CLF_MAPPING):
        print(f"ERROR: Echo classifier files not found.\n  {ECHO_CLF_PTH}\n  {ECHO_CLF_MAPPING}")
        sys.exit(1)
    echo_clf, index_to_loc_id, loc_id_to_index = load_echo_classifier(
        ECHO_CLF_PTH, ECHO_CLF_MAPPING, CLF_DEVICE
    )

    # --- Load Location Data --- (unchanged)
    hex_centers_df = pd.read_csv('./ops/hex_centers_50_0_extend_no_GL_version2.csv')
    hex_centers_df['text'] = hex_centers_df['text'].astype(str)

    file_perserves = './ops/extracted_active_locations_all.csv' # all the 1532 location zones that have enough echoes
    df_perserves = pd.read_csv(file_perserves)
    location_perserves = df_perserves['location'].tolist()
    location_perserves_set = set(location_perserves)

    LOC_INFO_FILE = "./ops/step3_add_more_locations_cluster_all.csv"
    df_loc_all = pd.read_csv(LOC_INFO_FILE)

    valid_loc_texts_in_hex = set(hex_centers_df[hex_centers_df['Type'] == 'LOC']['text'])
    candidate_loc_data = []

    print("Filtering and mapping major locations from df_loc_all against hex_centers_df...")
    for _, row in tqdm(df_loc_all.iterrows(), total=len(df_loc_all)):
        loc_parts = str(row['location']).split(' ')
        loc_id_text_commm, loc_id_text = loc_parts
        if loc_id_text in location_perserves_set:
            loc_id_text = f"[LOC_{str(row.name).zfill(5)}]"
            if loc_id_text in valid_loc_texts_in_hex:
                coords = get_coordinates(loc_id_text, hex_centers_df)
                if coords:
                    candidate_loc_data.append({
                        'id_text': loc_id_text,
                        'x': coords[0],
                        'y': coords[1],
                        'original_df_loc_id': row.name
                    })

    if not candidate_loc_data:
        print("Error: No valid candidate locations found after filtering."); sys.exit(1)
    print(f"Found {len(candidate_loc_data)} valid major locations for sampling.")

    all_candidate_loc_texts = [item['id_text'] for item in candidate_loc_data]
    all_candidate_loc_coords = np.array([[item['x'], item['y']] for item in candidate_loc_data])
    tree_all_candidates = cKDTree(all_candidate_loc_coords)

    # --- Main Generation Loop --- (unchanged except clf passed to find_best_middle_loc_contact)
    results_pool = []
    generation_params = {"do_sample": True, "top_p": 0.84, "top_k": 50, "temperature": 1.0}

    for sample_idx in range(NUM_SAMPLES):
        if len(candidate_loc_data) < 2:
            print("Not enough candidate locations to pick a pair."); break

        idx1, idx2 = 798, 413
        print("idx1, idx2: ", idx1, idx2)
        for idx, i in enumerate(candidate_loc_data):
            if i['id_text'] == f"[LOC_00{idx1}]":
                new_idx1 = idx
            if i['id_text'] == f"[LOC_00{idx2}]":
                new_idx2 = idx

        print("new_idx1, new_idx2: ", new_idx1, new_idx2)

        initial_start_loc_data  = candidate_loc_data[new_idx1]
        initial_target_loc_data = candidate_loc_data[new_idx2]

        print("initial_start_loc_data: ", initial_start_loc_data)
        print("initial_target_loc_data: ", initial_target_loc_data)

        initial_start_loc_text  = initial_start_loc_data['id_text']
        initial_start_xy        = (initial_start_loc_data['x'], initial_start_loc_data['y'])
        initial_target_loc_text = initial_target_loc_data['id_text']
        initial_target_xy       = (initial_target_loc_data['x'], initial_target_loc_data['y'])
        actual_dist = np.linalg.norm(np.array(initial_start_xy) - np.array(initial_target_xy))

        print("actual_dist: ", actual_dist)
        print(f"\n--- Sample {sample_idx+1}/{NUM_SAMPLES}: {initial_start_loc_text} -> {initial_target_loc_text} (Dist: {actual_dist:.1f}m) ---")

        current_start_loc_text  = initial_start_loc_text
        target_reached_flag     = False
        stitched_sequence_parts = []
        used_middle_loc_texts   = []
        current_distance        = 999999
        failed_trajs_strs       = []

        for refinement_attempt in range(MAX_REFINEMENTS):
            final_num_refinements = refinement_attempt + 1
            print(f"  Refinement Attempt {refinement_attempt + 1}/{MAX_REFINEMENTS}: Current Start: {current_start_loc_text} -> Target: {initial_target_loc_text}")

            prompt_text = f"{tokenizer.bos_token}{current_start_loc_text}{initial_target_loc_text}[SRA][INST]"
            print("force the prompt the same : ", prompt_text)
            prompt_token_len = len(tokenizer.tokenize(prompt_text))

            path_plan_traj = generate_path_plan(model, tokenizer, prompt_text, current_start_loc_text, DEVICE, **generation_params)
            print("path_plan_traj: ", path_plan_traj)

            if len(path_plan_traj) > 300:
                print("    Path plan too long (>300 tokens). Regenerating path plan.")
                continue

            # CHANGED: use generate_azi_traj_and_hidden_states instead of generate_azi_traj
            raw_output_tokens_str_list, hidden_state_sequence = generate_azi_traj_and_hidden_states(
                model, tokenizer, path_plan_traj, current_start_loc_text, DEVICE, **generation_params
            )

            print("raw_output_tokens_str_list: ", raw_output_tokens_str_list)

            if not raw_output_tokens_str_list or len(raw_output_tokens_str_list) <= prompt_token_len:
                print("    Generation failed or too short.")
                failed_trajs_strs.append("GENERATION_FAILED_OR_TOO_SHORT")
                continue

            try:
                parsed_segment = parse_model_segment_output(raw_output_tokens_str_list, prompt_token_len)
            except Exception as e:
                print(" The format of the generated segment is not good!!!")
                failed_trajs_strs.append("PARSE_ERROR: " + " ".join(raw_output_tokens_str_list[prompt_token_len:]))
                continue

            segment_achieved_target = False

            if refinement_attempt == 0:
                if parsed_segment['azi_calc_start_loc'] == current_start_loc_text:
                    print("!!!!!!! correct initial_start_loc_text: ", parsed_segment['azi_calc_start_loc'], initial_start_loc_text)
                else:
                    parsed_segment['azi_calc_start_loc'] = current_start_loc_text
                    print("!!!!!! ========= force init one changed to ", initial_start_loc_text, "===========")

            if not parsed_segment['azi_calc_start_loc']:
                print(f"    Failed to parse azi_calc_start_loc.")
                failed_trajs_strs.append(" ".join(raw_output_tokens_str_list[prompt_token_len:]))
                final_num_refinements = final_num_refinements - 1
                continue

            azi_calc_start_xy = get_coordinates(parsed_segment['azi_calc_start_loc'], hex_centers_df)
            segment_azi_path_points = calculate_azimuth_path(azi_calc_start_xy, parsed_segment['post_pos_loc_sequence'])

            if segment_azi_path_points and len(segment_azi_path_points) > 10:
                min_dist_to_target, pt_idx_near_target = get_path_min_dist_to_point(segment_azi_path_points, initial_target_xy)

                if min_dist_to_target <= TARGET_PROXIMITY_THRESHOLD:
                    print(f"    Target {initial_target_loc_text} reached by path proximity ({min_dist_to_target:.1f}m) at segment step {pt_idx_near_target}.")
                    num_azi_to_keep = pt_idx_near_target
                    path_plan_part, traj_tokens = collect_tokens_for_segment(parsed_segment, num_azi_to_keep)
                    stitched_sequence_parts.append([prompt_text, path_plan_part, traj_tokens, initial_target_loc_text])
                    target_reached_flag = True
                    segment_achieved_target = True

            if segment_achieved_target:
                break

            if not segment_achieved_target and (not segment_azi_path_points or len(segment_azi_path_points) <= 1):
                print(f"    No azimuth steps in segment.")
                failed_trajs_strs.append(" ".join(raw_output_tokens_str_list[prompt_token_len:]))
                continue

            # CHANGED: pass clf + hidden states instead of location_precisions_map
            potential_middle_contact = find_best_middle_loc_contact(
                segment_path_points=segment_azi_path_points,
                middle_loc_ids_texts=all_candidate_loc_texts,
                hex_centers_df=hex_centers_df,
                middle_loc_tree=tree_all_candidates,
                radius_m=MIDDLE_LOC_PROXIMITY_THRESHOLD,
                overall_target_xy=initial_target_xy,
                refinement_attempt=refinement_attempt,
                current_start_loc_text=current_start_loc_text,
                current_distance=current_distance,
                hidden_state_sequence=hidden_state_sequence,
                echo_clf=echo_clf,
                index_to_loc_id=index_to_loc_id,
                clf_device=CLF_DEVICE,
            )

            if potential_middle_contact:
                current_distance = potential_middle_contact['dist_from_middle_to_overall_target']
                contact_info = potential_middle_contact
                if contact_info['middle_loc_text'] == current_start_loc_text or contact_info['middle_loc_text'] == initial_target_loc_text or contact_info['middle_loc_text'] in used_middle_loc_texts:
                    print(f"    Middle contact ({contact_info['middle_loc_text']}) is current start, target, or already used.")
                    failed_trajs_strs.append(" ".join(raw_output_tokens_str_list[prompt_token_len:]))
                    continue

                print(f"    Found valid middle contact: {contact_info['middle_loc_text']} (dist to target: {contact_info['dist_from_middle_to_overall_target']:.1f}m).")
                num_azi_to_keep = contact_info['traj_step_idx']
                path_plan_part, traj_tokens = collect_tokens_for_segment(parsed_segment, num_azi_to_keep)
                stitched_sequence_parts.append([prompt_text, path_plan_part, traj_tokens, contact_info['middle_loc_text']])
                used_middle_loc_texts.append(contact_info['middle_loc_text'])
                current_start_loc_text = contact_info['middle_loc_text']

            else:
                print(f"    No target or valid middle location reached. Continuing.")
                failed_trajs_strs.append(" ".join(raw_output_tokens_str_list[prompt_token_len:]))
                continue

        # --- reconstruct full path (completely unchanged) ---
        print("stitched_sequence_parts: ", stitched_sequence_parts)
        print("\nused_middle_loc_texts: ", used_middle_loc_texts)

        all_azi_tokens = []
        all_path_plan_tokens = []
        used_middle_loc_texts = []

        temp_current_path_x = initial_start_xy[0]
        temp_current_path_y = initial_start_xy[1]
        final_stitched_tokens_all = []

        for i in range(len(stitched_sequence_parts)):
            prompt_part = stitched_sequence_parts[i][0]
            path_plan_part = stitched_sequence_parts[i][1]
            traj_tokens = stitched_sequence_parts[i][2]
            current_destination = stitched_sequence_parts[i][3]

            if i != len(stitched_sequence_parts)-1:
                used_middle_loc_texts.append(current_destination)

            final_stitched_tokens_all.extend(traj_tokens)
            my_xy = get_coordinates(current_destination, hex_centers_df)

            all_distances = []
            for my_token in path_plan_part:
                tmp_xy = get_coordinates(my_token, hex_centers_df)
                distance = np.linalg.norm(np.array(tmp_xy) - np.array(my_xy))
                all_distances.append(distance)

            if len(all_distances) == 0:
                print("all_distances is empty, cannot find minimum distance.")
                continue
            min_distance = min(all_distances)
            min_index = all_distances.index(min_distance)

            target_path_plan = path_plan_part[:min_index+1]
            all_path_plan_tokens.extend(target_path_plan)
            if all_path_plan_tokens[-1] != current_destination:
                all_path_plan_tokens.append(current_destination)

            for token in traj_tokens:
                if token.isdigit():
                    all_azi_tokens.append(token)

        print("\nall_path_plan_tokens: ", all_path_plan_tokens)
        print("\nall_azi_tokens: ", all_azi_tokens)

        full_recalculated_path_points = []
        full_recalculated_path_points.append((temp_current_path_x, temp_current_path_y))
        for token in all_azi_tokens:
            try:
                angle = float(token)
                math_angle_deg = 90.0 - angle
                math_angle_rad = math.radians(math_angle_deg)
                dx = AZIMUTH_STEP_DISTANCE * math.cos(math_angle_rad)
                dy = AZIMUTH_STEP_DISTANCE * math.sin(math_angle_rad)
                temp_current_path_x += dx
                temp_current_path_y += dy
                full_recalculated_path_points.append((temp_current_path_x, temp_current_path_y))
            except (IndexError, ValueError):
                pass

        predict_last_dist = float('inf')
        predict_nearest_dist = float('inf')

        if full_recalculated_path_points:
            if len(full_recalculated_path_points) > 0:
                predict_last_dist = np.linalg.norm(np.array(full_recalculated_path_points[-1]) - np.array(initial_target_xy))
                min_d, _idx = get_path_min_dist_to_point(full_recalculated_path_points, initial_target_xy)
                predict_nearest_dist = min_d
        else:
            predict_last_dist = np.linalg.norm(np.array(initial_start_xy if initial_start_xy else (0,0)) - np.array(initial_target_xy))
            predict_nearest_dist = predict_last_dist
            sys.exit(1)

        print("predict_nearest_dist: ", predict_nearest_dist, "predict_last_dist: ", predict_last_dist)

        results_pool.append({
            "actual_distance": actual_dist,
            'loc1_x': initial_start_xy[0] if initial_start_xy else None,
            'loc1_y': initial_start_xy[1] if initial_start_xy else None,
            'loc2_x': initial_target_xy[0], 'loc2_y': initial_target_xy[1],
            'initial_start_loc_text': initial_start_loc_text,
            'initial_target_loc_text': initial_target_loc_text,
            'azimuths_tokens_str': " ".join(all_azi_tokens),
            'generated_grids_str': " ".join(all_path_plan_tokens),
            'middle_locations_texts': ", ".join(used_middle_loc_texts),
            'predict_last_distance_to_target': predict_last_dist,
            'predict_nearest_distance_to_target': predict_nearest_dist,
            'target_reached': target_reached_flag,
            'refinement_steps_taken': final_num_refinements,
            'full_generated_sequence': " ".join(final_stitched_tokens_all),
            'failed_trajectories': " | ".join(failed_trajs_strs),
        })
        print(f"  Sample Result: Target Reached: {target_reached_flag}, Refinements: {final_num_refinements}, NearestDist: {predict_nearest_dist:.1f}m")

    # --- Save Results --- (unchanged)
    df_save = pd.DataFrame(results_pool)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_file = f'./result_4vtoken/trajectories_iterative_with_echo_{timestamp}.feather'
    if os.path.exists(save_file):
        save_file = save_file.replace(".feather", f"_{random.randint(0,10000)}.feather")

    if not df_save.empty:
        print(f"Saving results ({len(df_save)} rows) to: {save_file}")
        df_save.to_feather(save_file)
    else:
        print("No results to save.")


if __name__ == '__main__':
    DEVICE     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    CLF_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    CHECKPOINT_PATH = "./GPT_NEO_finetune_FUSION_v1"
    TOKENIZER_PATH  = "./ops/tokenizer_v9_2.json"

    _CLF_DIR         = ("/home/xingchen/Projects/navigation_code_without_GL/"
                        "map_echo_with_linear_layer/"
                        "saved_navigation_models_epochs_15000_5")
    ECHO_CLF_PTH     = os.path.join(_CLF_DIR, "model_loc_clf.pth") # MLP for untrained dataset
    ECHO_CLF_MAPPING = os.path.join(_CLF_DIR, "location_mapping.json")

    print("CHECKPOINT_PATH: ", os.path.basename(CHECKPOINT_PATH))
    print("TOKENIZER_PATH: ",  os.path.basename(TOKENIZER_PATH))
    print("ECHO_CLF_PTH: ",    ECHO_CLF_PTH)

    run()