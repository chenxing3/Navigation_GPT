

"""
Main script for generating bat-inspired trajectory data for language models.

This script simulates bat flight paths and converts them into a tokenized text format
suitable for either pre-training or fine-tuning a GPT-style model.

Modes:
- 'pretrain': Generates long, continuous sequences of movement and location tokens.
- 'finetune': Generates structured instruction-response pairs, where the instruction
             is a path plan and the response is the detailed movement sequence.

Example Usage:
    # Generate data for pre-training
    python Step1_generate_trajectory_reverse.py pretrain --num_samples 10000 --output_dir ./pretrain_data

    # Generate data for fine-tuning
    python Step1_generate_trajectory_reverse.py finetune --num_samples 5000 --output_dir ./finetune_data
"""

"""
Main script for generating bat-inspired trajectory data for language models.
"""

import argparse
import os
import random
import sys
from typing import List

import pandas as pd
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from utils import (
    setup_environment_data,
    generate_walk,
    rotate_trajectory,
    get_safe_start_point,
    add_azimuth,
    get_straightness,
    remove_duplicates
)

# --- Constants for Fine-tuning Segmentation ---
MODEL_MAX_LENGTH = 1024
OUTPUT_SEPARATOR = "[SEP]"
MIN_AZI_TOKENS_PER_SEGMENT = 10
LENGTH_BUFFER = 6 
SEMANTIC_BREAK_THRESHOLD_RATIO = 0.85

def find_location_hits(df_traj: pd.DataFrame, tree_loc) -> tuple:
    """Scans a trajectory DataFrame for intersections with location points."""
    loc_hits = {}
    loc_indices = []
    loc_threshold = 30  # meters

    for idx, row in df_traj.iterrows():
        x, y = row['x'], row['y']
        _, loc_idx = tree_loc.query([x, y], distance_upper_bound=loc_threshold)

        if loc_idx != tree_loc.n:
            loc_hits[idx] = loc_idx
            loc_indices.append(idx) 

    return loc_hits, loc_indices

def generate_base_trajectory(env_data: tuple) -> pd.DataFrame:
    """Generates a single raw trajectory."""
    bl_x, bl_y, tr_x, tr_y, tree_hex, tree_loc, _ = env_data

    step = random.randint(600, 2500)
    angular_error_sd = random.uniform(0.03, 0.28)
    df_traj = generate_walk(n=step + 1, angular_error_sd=angular_error_sd)

    df_traj = rotate_trajectory(df_traj, random.randint(0, 360))
    start_x, start_y = get_safe_start_point(df_traj, bl_x, bl_y, tr_x, tr_y)
    df_traj['x'] += start_x
    df_traj['y'] += start_y
    df_traj = add_azimuth(df_traj)

    return df_traj

def reverse_trajectory_dataframe(df_traj: pd.DataFrame) -> pd.DataFrame:
    """Reverses the order of points and recalculates azimuths."""
    df_rev = df_traj.iloc[::-1].reset_index(drop=True)
    df_rev = add_azimuth(df_rev)
    return df_rev

def format_for_pretraining(df_traj: pd.DataFrame, loc_hits: dict, loc_indices: list) -> str:
    """Formats a trajectory into a continuous string for pre-training."""
    if not loc_hits:
        return None

    df_working = df_traj.copy()
    df_working['loc_id'] = "-1"
    
    for idx, loc_idx in loc_hits.items():
        location_name = f"[LOC_{str(loc_idx).zfill(5)}]"
        if idx + 1 not in loc_hits:
            try:
                current_list_index = loc_indices.index(idx)
                possible_next_list_index = current_list_index + 1
                if possible_next_list_index < len(loc_indices):
                    next_hit_timestamp = loc_indices[possible_next_list_index]
                    next_loc_id = loc_hits[next_hit_timestamp]
                    location_name = f"[LOC_{str(loc_idx).zfill(5)}][LOC_{str(next_loc_id).zfill(5)}]"
                    df_working.at[idx, 'loc_id'] = location_name
                else:
                    if random.random() < 0.5:
                         df_working.at[idx, 'loc_id'] = location_name
            except ValueError:
                 if random.random() < 0.5:
                    df_working.at[idx, 'loc_id'] = location_name
        else:
            if random.random() < 0.5:
                df_working.at[idx, 'loc_id'] = location_name

    text_sequence = []
    for i in range(1, len(df_working)):
        azimuth_str = f"{str(int(df_working['azimuth'][i])).zfill(3)}_"
        text_sequence.append(azimuth_str)
        if df_working['loc_id'][i] != "-1":
            text_sequence.append(df_working['loc_id'][i])

    return "".join(text_sequence).strip()

def segment_finetune_trajectory(full_text: str, tokenizer, is_complete_trajectory: bool = True) -> List[str]:
    """
    Segments a long fine-tuning string.
    Args:
        is_complete_trajectory: 
            True -> Ends with [EOS][EOS] (We kept the end of the trajectory)
            False -> Ends with [SEP] (We kept the front, so it's incomplete)
    """
    try:
        last_pos_index = full_text.rindex("[POS]")
        context_part = full_text[:last_pos_index + len("[POS]")].strip()
        trajectory_part = full_text[last_pos_index + len("[POS]"):].strip()
    except ValueError:
        return []

    context_len = len(tokenizer.encode(context_part, add_special_tokens=False))
    max_traj_tokens = MODEL_MAX_LENGTH - context_len - LENGTH_BUFFER - 2
    semantic_break_threshold = int(max_traj_tokens * SEMANTIC_BREAK_THRESHOLD_RATIO)
    
    if max_traj_tokens < MIN_AZI_TOKENS_PER_SEGMENT:
        return []

    trajectory_tokens = trajectory_part.split()
    if not trajectory_tokens:
        return []

    segmented_texts = []
    start_idx = 0
    last_break_token = None

    # print("trajectory_tokens: ", trajectory_tokens, len(trajectory_tokens))
    # sys.exit(1)


    i = 0
    while i < len(trajectory_tokens):
        # print(i)
        current_segment_tokens = trajectory_tokens[start_idx : i + 1]
        prefix = f"{last_break_token} " if last_break_token else ""
        segment_to_check = f"{context_part} {prefix}{' '.join(current_segment_tokens)}"
        current_len = len(tokenizer.encode(segment_to_check, add_special_tokens=False))
        
        break_now = False
        is_final_segment = (i == len(trajectory_tokens) - 1)

        # print("current_len: ", current_len)
        
        # if current_len > MODEL_MAX_LENGTH - LENGTH_BUFFER:
        #     break_now = True
        #     current_segment_tokens.pop() 
        #     last_break_token = trajectory_tokens[i]
        #     i -= 1


        # --- FIXED LOGIC START ---
        if current_len > MODEL_MAX_LENGTH - LENGTH_BUFFER:
            break_now = True
            
            # Only step back if we have previous tokens in this segment to keep
            if i > start_idx:
                current_segment_tokens.pop() 
                last_break_token = trajectory_tokens[i]
                i -= 1
            else:
                # Edge Case: Context + 1st Token is ALREADY too big.
                # We cannot step back (i -= 1) or we create an infinite loop.
                # We must force a break here.
                last_break_token = trajectory_tokens[i]
                # We do NOT decrement i. We accept this token causes a break 
                # and start the next segment from i+1.
        # --- FIXED LOGIC END ---

        elif current_len >= semantic_break_threshold:
            token = trajectory_tokens[i]
            if token.startswith("LOC_"):
                break_now = True
                last_break_token = token


        # print("is_final_segment: ", is_final_segment)
        if is_final_segment:
            break_now = True
            last_break_token = None

        if break_now:
            # Determine Terminator
            if is_final_segment:
                terminator = "[EOS][EOS]" if is_complete_trajectory else OUTPUT_SEPARATOR
            else:
                # Intermediate segments (caused by length) always use SEP
                terminator = OUTPUT_SEPARATOR
                
            final_segment_str = f"{segment_to_check}{terminator}"

            if len(tokenizer.encode(final_segment_str)) <= MODEL_MAX_LENGTH:
                segmented_texts.append(final_segment_str)
            else:
                print(f"Warning: Segment still too long after formatting. Skipping.", file=sys.stderr)
            
            start_idx = i + 1

            # print("len(tokenizer.encode(final_segment_str)): ", len(tokenizer.encode(final_segment_str)), MODEL_MAX_LENGTH)
            # print("final_segment_str: ", final_segment_str)
            # print("start_idx: ", start_idx)
            # sys.exit(1)



        i += 1
        
        
    return segmented_texts

def format_for_finetuning(df_traj: pd.DataFrame, loc_hits: dict, loc_indices: list, tokenizer) -> List[str]:
    """
    Formats a trajectory into instruction-response segments.
    
    CRITICAL LOGIC:
    1. The Context (Instruction) is built from the FULL trajectory and is NEVER changed.
    2. If the tokens exceed the limit, we cut the TRAJECTORY TOKENS only.
    """
    if len(loc_indices) < 2:
        return []
    
    min_idx, max_idx = min(loc_indices), max(loc_indices)
    if (max_idx - min_idx) <= 50:
        return []

    df_sliced = df_traj.iloc[min_idx:max_idx + 1].reset_index(drop=True)

    # --- 1. Build the FIXED Context (Global Plan) ---
    path_plan_landmarks = []
    for abs_idx in range(min_idx + 1, max_idx):
        if abs_idx in loc_hits:
            path_plan_landmarks.append(f"[LOC_{str(loc_hits[abs_idx]).zfill(5)}]")
    
    unique_path_plan = remove_duplicates(path_plan_landmarks)

    start_loc = f"[LOC_{str(loc_hits[min_idx]).zfill(5)}]"
    end_loc = f"[LOC_{str(loc_hits[max_idx]).zfill(5)}]"
    straightness = get_straightness(df_sliced)
    straight_token = "[SRA]" if straightness > 0.87 else "[WID]"

    # This context string represents the FULL intent and will NOT change
    context_part = (f"{tokenizer.bos_token}{start_loc}{end_loc}{straight_token}"
                    f"[INST]{''.join(unique_path_plan)}[POS]")

    # --- 2. Build Raw Trajectory Tokens ---
    trajectory_tokens = []
    for i in range(1, len(df_sliced)):
        azimuth_val = df_sliced['azimuth'][i]
        azimuth_str = f"{str(int(azimuth_val)).zfill(3)}_"
        trajectory_tokens.append(azimuth_str)
        
        current_abs_idx = min_idx + i
        loc_hit = loc_hits.get(current_abs_idx)

        if loc_hit is not None:
            location_name = f"[LOC_{str(loc_hit).zfill(5)}]"
            is_last_in_patch = (current_abs_idx + 1) not in loc_hits

            if is_last_in_patch:
                trajectory_tokens.append(location_name)
                try:
                    current_list_pos = loc_indices.index(current_abs_idx)
                    next_list_pos = current_list_pos + 1
                    if next_list_pos < len(loc_indices):
                        next_hit_timestamp = loc_indices[next_list_pos]
                        if next_hit_timestamp <= max_idx:
                            next_loc_id = loc_hits[next_hit_timestamp]
                            trajectory_tokens.append(f"[LOC_{str(next_loc_id).zfill(5)}]")
                except ValueError:
                    pass
            else:
                if random.random() < 0.5:
                    trajectory_tokens.append(location_name)

    # --- 3. Length Check & Truncation (Trajectory Only) ---
    
    context_ids = tokenizer.encode(context_part, add_special_tokens=False)
    max_allowed_traj_ids = MODEL_MAX_LENGTH - len(context_ids) - LENGTH_BUFFER

    full_traj_str = "".join(trajectory_tokens)
    current_traj_ids = tokenizer.encode(full_traj_str, add_special_tokens=False)

    final_tokens = trajectory_tokens
    is_complete_trajectory = True

    if len(current_traj_ids) > max_allowed_traj_ids:
        # 30% chance to Keep Front (Cut End) -> INCOMPLETE
        # 70% chance to Keep End (Cut Front) -> COMPLETE (Destination reached)
        keep_front_strategy = (random.random() < 0.30)
        
        valid_cut_found = False
        # Find indices of tokens that are LOC markers
        loc_token_indices = [i for i, tok in enumerate(trajectory_tokens) if "[LOC_" in tok]
        
        if keep_front_strategy:
            # STRATEGY: KEEP FRONT (Cut Tail)
            # Trajectory: Start -> Middle ... [CUT]
            # Context:    Start -> Middle -> End
            # Result:     Ends with [SEP]
            for loc_idx in reversed(loc_token_indices):
                candidate = trajectory_tokens[:loc_idx+1]
                candidate_str = "".join(candidate)
                if len(tokenizer.encode(candidate_str, add_special_tokens=False)) <= max_allowed_traj_ids:
                    final_tokens = candidate
                    is_complete_trajectory = False
                    valid_cut_found = True
                    break
        else:
            # STRATEGY: KEEP END (Cut Front)
            # Trajectory: [CUT] ... Middle -> End
            # Context:    Start -> Middle -> End
            # Result:     Ends with [EOS][EOS]
            for loc_idx in loc_token_indices:
                candidate = trajectory_tokens[loc_idx:]
                candidate_str = "".join(candidate)
                if len(tokenizer.encode(candidate_str, add_special_tokens=False)) <= max_allowed_traj_ids:
                    final_tokens = candidate
                    is_complete_trajectory = True
                    valid_cut_found = True
                    break
        
        if not valid_cut_found:
            return [] # Could not find a valid cut point

    full_text = f"{context_part}{''.join(final_tokens)}"

    return segment_finetune_trajectory(full_text, tokenizer, is_complete_trajectory=is_complete_trajectory)


def main():
    """Main function to parse arguments and run data generation."""
    parser = argparse.ArgumentParser(description="Generate trajectory data for LLMs.")
    parser.add_argument("mode", choices=["pretrain", "finetune"],
                        help="The generation mode: 'pretrain' or 'finetune'.")
    parser.add_argument("--num_samples", type=int, required=True,
                        help="Number of trajectories to generate.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the output text file.")
    parser.add_argument("--hex_centers_path", type=str,
                        default="./ops/hex_centers_50_0_extend_no_GL_version2.csv", # default="./ops/hex_centers_50_0_extend_no_GL_version2.csv", default="./ops/hex_centers_50_0_extend_no_GL_version2_normal.csv",
                        help="Path to the hex centers CSV file.")
    parser.add_argument("--tokenizer_path", type=str,
                        default="./ops/tokenizer_v9_2.json",
                        help="Path to the tokenizer file (required for finetune mode).")
    
    args = parser.parse_args()

    # --- Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    env_data = setup_environment_data(args.hex_centers_path)
    _, _, _, _, _, tree_loc, _ = env_data 
    
    tokenizer = None
    if args.mode == "finetune":
        if not os.path.exists(args.tokenizer_path):
            print(f"Error: Tokenizer file not found at '{args.tokenizer_path}'", file=sys.stderr)
            sys.exit(1)
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_path)
        tokenizer.bos_token = "[BOS]"
        tokenizer.eos_token = "[EOS]"

    base_output_path = os.path.join(args.output_dir, f"generated_trajectories_{args.mode}")
    output_ext = ".txt"
    output_filename = f"{base_output_path}{output_ext}"
    
    counter = 1
    while os.path.exists(output_filename):
        output_filename = f"{base_output_path}_{counter}{output_ext}"
        counter += 1

    with open(output_filename, "w") as fout:
        pbar = tqdm(total=args.num_samples, desc=f"Generating for {args.mode}")
        generated_count = 0
        
        while generated_count < args.num_samples:
            df_forward = generate_base_trajectory(env_data)
            df_backward = reverse_trajectory_dataframe(df_forward)

            trajectories_to_process = [df_forward, df_backward]
            
            for df_current in trajectories_to_process:
                if generated_count >= args.num_samples:
                    break

                loc_hits, loc_indices = find_location_hits(df_current, tree_loc)

                if args.mode == "pretrain":
                    text_output = format_for_pretraining(df_current, loc_hits, loc_indices)

                    # print("text_output: ", text_output)
                    # sys.exit(1)
                    if text_output:
                        fout.write(text_output + "\n")
                        generated_count += 1
                        pbar.update(1)
                
                else: # finetune mode
                    segments = format_for_finetuning(df_current, loc_hits, loc_indices, tokenizer)
                    if segments:
                        for segment in segments:
                            fout.write(segment + "\n")
                        generated_count += 1
                        pbar.update(1)

        pbar.close()

if __name__ == "__main__":
    main()