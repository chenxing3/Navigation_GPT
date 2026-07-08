import os, sys, glob, copy, math, random, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainingArguments, GPTNeoForCausalLM, PreTrainedTokenizerFast, Trainer, AutoConfig, GPTNeoConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoPreTrainedModel, GPTNeoModel, GPTNeoBlock, _prepare_4d_causal_attention_mask
from torch.nn import CrossEntropyLoss

from datasets import load_from_disk, Dataset # Added Dataset for get_train_dataloader type hint
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, get_worker_info # Added get_worker_info
from transformers.trainer_pt_utils import ( # For get_train_dataloader
    DistributedSamplerWithLoop,
    LengthGroupedSampler,
    DistributedLengthGroupedSampler,
    SequentialDistributedSampler
)


from functools import partial
from typing import Optional, Tuple, Union, List, Dict, Callable # Added Callable
from transformers.utils import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from safetensors.torch import load_file


# --- Global dicts ---
location_map_global = {} # Maps filtered common_loc_id -> contiguous class index
echo_dataset_global = None
echo_indices_by_location_global = None # Filtered map: {location_key: [ds_index1,...]}

# --- Function to build the in-memory index ---
def build_location_index_map(dataset):
    """Builds a map from location_key to a list of dataset row indices."""
    print("Building in-memory index map (location_key -> dataset indices)...")
    index_map = defaultdict(list)
    # Ensure 'location_key' column exists
    if 'location_key' not in dataset.column_names:
        raise ValueError("Dataset loaded from disk is missing 'location_key' column.")

    # Iterate efficiently - accessing column directly is often faster
    location_keys = dataset['location_key']
    for i, key in enumerate(tqdm(location_keys, desc="Indexing Locations")):
        # Convert key to standard Python int if it's numpy int64 etc.
        index_map[int(key)].append(i)
    print(f"Built index map for {len(index_map)} unique locations.")
    return index_map
# ******MODIFICATION END******


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
    def __init__(self, input_size=15000, patch_size=150, in_channels=1, embed_dim=1024, num_encoder_layers=2, num_heads=8):
        super(BatNavEncoder_attn2, self).__init__()
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=[1, patch_size], stride=[1, patch_size]),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=[1, 3], stride=[1, 1], padding=[0, 1]),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=[1, 3], stride=[1, 1], padding=[0, 1]),
        )
        self.positional_embedding = nn.Parameter(torch.randn(int(6*input_size/patch_size), embed_dim) / (embed_dim ** 0.5))
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(embed_dim, num_heads), num_encoder_layers)
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
        # print("Input shape: ", x.shape)
        cnn_patches = self.patch_embedding(x)
        if torch.isnan(cnn_patches[0]).any():
            print('Error: cnn_patches has NaN!')
            sys.exit(1)
        cnn_patches = cnn_patches.flatten(2).permute(0, 2, 1)

        # print("cnn_patches: ",cnn_patches.shape)
        # sys.exit(1)
        cnn_patches_pos = cnn_patches + self.positional_embedding[None, :, :].to(cnn_patches.dtype)
        encoded_patches = self.transformer_encoder(cnn_patches_pos.permute(1, 0, 2)).permute(1, 0, 2)
        weights = F.softmax(self.feature_weights, dim=0)
        query = (weights * encoded_patches).sum(dim=1, keepdim=True)
        attn_output = self.echo_attn(query, encoded_patches, cnn_patches)
        enhanced_encoding = attn_output + encoded_patches[:, :1, :]
        prediction = self.output_layer(enhanced_encoding.reshape(x.size(0), -1))
        return prediction

class GPTNeoModel_NoPE(GPTNeoModel): # Inherit from original GPTNeoModel
    def __init__(self, config):
        super().__init__(config) # Initialize the parent class fully

        # Explicitly remove or disable wpe if it exists after super().__init__
        if hasattr(self, 'wpe'):
            print("Removing standard absolute positional embedding layer (wpe).")
            # Option 1: Delete it (might cause issues if other parts expect it)
            # del self.wpe
            # Option 2: Replace with an identity module or None check in forward
            self.wpe = nn.Identity() # Replace with identity, does nothing

    # Override the forward method
    # Copy the original forward method from the source code you provided
    # and REMOVE the addition of position_embeds
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None, # Still accept position_ids, but won't use wpe
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPast]: # Adjusted return type hint
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

        # --- PE Calculation Removed ---
        # We still need position_ids for the causal mask calculation if not using Flash Attention
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        # --- End PE Removal ---

        # Prepare head mask
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        # --- POSITION EMBEDDING ADDITION REMOVED ---
        # position_embeds = self.wpe(position_ids) # This line is effectively gone or wpe is Identity
        # hidden_states = inputs_embeds + position_embeds # REMOVED + position_embeds
        hidden_states = inputs_embeds # Use only token embeddings
        # --- END REMOVAL ---

        # Attention mask preparation (handle different implementations)
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
            if use_cache: logger.warning_once("use_cache=True incompatible..."); use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states: all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(block.__call__, hidden_states, None, attention_mask, head_mask[i], use_cache, output_attentions)
            else:
                outputs = block(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask[i], use_cache=use_cache, output_attentions=output_attentions)
            hidden_states = outputs[0]
            if use_cache: presents = presents + (outputs[1],)
            if output_attentions: all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states: all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict: return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)
        # Make sure return type matches parent class expectation
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=presents, hidden_states=all_hidden_states, attentions=all_self_attentions)





from dataclasses import dataclass

@dataclass
class SplitTaskOutput(CausalLMOutputWithPast):
    """
    Output object for split LM and Echo-Aux tasks.
    Inherits from CausalLMOutputWithPast for compatibility but adds echo-specific states.
    """
    loss: Optional[torch.FloatTensor] = None # Total loss computed in Trainer
    lm_logits: Optional[torch.FloatTensor] = None
    echo_hidden_states: Optional[torch.FloatTensor] = None # Hidden states from the echo path
    # Inherited: past_key_values, hidden_states (from LM path), attentions (from LM path)



# --- Model integrating the Echo Encoder ---
class GPTNeoForCausalLM_RawEcho_WithAux(GPTNeoForCausalLM):
    # __init__ remains the same as previous version (with echo_encoder)

    def __init__(self, config, num_aux_classes: int, echo_encoder_config: dict): # Pass encoder config
        super().__init__(config)
        self.transformer = GPTNeoModel_NoPE(config) # Use the NoPE base

        self.cls_token_id = 4

        # --- Instantiate Echo Encoder ---
        self.echo_encoder = BatNavEncoder_attn2(**echo_encoder_config)
        # Get the output dimension of the echo encoder
        self.echo_embed_dim = echo_encoder_config.get('embed_dim', 1024) # Default if not specified

        # --- Auxiliary Classification Head ---
        if num_aux_classes <= 0:
             raise ValueError("num_aux_classes must be positive.")
        self.aux_classifier = nn.Linear(config.hidden_size, num_aux_classes)
        print(f"Initialized RawEcho Model with Aux Classifier Head: {config.hidden_size} -> {num_aux_classes} classes")
        # self.echo_projection = nn.Linear(self.echo_embed_dim, config.hidden_size)

        self.init_weights()



    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None, # LM labels passed directly to loss calc
        # --- Custom arguments from Collator ---
        raw_echo_tensors: Optional[List[torch.Tensor]] = None, # List of [5, 1, 6, 15000]
        echo_placeholder_indices: Optional[List[Tuple[int, int, int, int]]] = None, # [(b, sep, unk, cls)]
        aux_labels_info: Optional[List[Tuple[int, int, int]]] = None, # [(b, cls, target)] - needed for loss only
        # --- Standard args (mostly for LM path) ---
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None, # Avoid using this directly now
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None, # Output states for both paths if needed
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SplitTaskOutput]: # Return custom output object

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Always output hidden states needed for loss calculation
        output_hidden_states = True

        if input_ids is None:
            raise ValueError("input_ids must be provided for this split model.")
        if inputs_embeds is not None:
            raise ValueError("inputs_embeds is not supported directly; use input_ids.")

        device = input_ids.device
        dtype = self.transformer.wte.weight.dtype # Get dtype from embeddings


        # ====== Path 1: Language Modeling ======
        # Run transformer on original input_ids
        lm_transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden_states_lm = lm_transformer_outputs.last_hidden_state
        lm_logits = self.lm_head(hidden_states_lm)


        # ====== Path 2: Echo Classification (Conditional) ======
        hidden_states_echo = None # Initialize
        if raw_echo_tensors:
            self.echo_encoder.to(device)



            # Input list has N tensors of shape [5, 1, 6, 15000]
            # Stack them to [N, 5, 1, 6, 15000] where N is num samples with echoes
            num_echo_samples = len(raw_echo_tensors)
            combined_raw_echoes = torch.stack(raw_echo_tensors).to(device, dtype=dtype) # Ensure dtype

            # Reshape for encoder: [N*5, 1, 6, 15000]
            flat_batch_echoes = combined_raw_echoes.view(
                num_echo_samples * 5,
                combined_raw_echoes.size(2), # C
                combined_raw_echoes.size(3), # H
                combined_raw_echoes.size(4)  # W
            )

            # Encode: -> [N*5, echo_embed_dim]
            encoded_flat_batch = self.echo_encoder(flat_batch_echoes)

            # Reshape back: -> [N, 5, echo_embed_dim]
            encoded_echo_groups = encoded_flat_batch.view(
                num_echo_samples, 5, self.echo_embed_dim
            )

            # print("encoded_echo_groups: ", encoded_echo_groups, encoded_echo_groups.shape)
            # sys.exit(1)

            echo_transformer_outputs = self.transformer(
                inputs_embeds=encoded_echo_groups, # Use injected embeddings
                # attention_mask=attention_mask_echo_path,         # Reuse original mask - THIS IS KEY
                # Ensure other args are None if not needed or handled carefully
                past_key_values=None, # No caching needed for this separate pass usually
                token_type_ids=token_type_ids, # Reuse if applicable
                position_ids=position_ids,     # Reuse if applicable
                head_mask=head_mask,         # Reuse if applicable
                use_cache=False,             # Disable cache for this pass
                output_attentions=False,       # Likely don't need attentions here
                output_hidden_states=True,     # Need hidden states for aux loss
                return_dict=True,
            )
            hidden_states_echo = echo_transformer_outputs.last_hidden_state
        # else: print("Warn: No successful injections, skipping echo transformer pass.")

            # print("hidden_states_echo: ", hidden_states_echo.shape)

        # ====== Combine Outputs ======
        if not return_dict:
             # Returning tuples is complicated, strongly recommend using dicts
             outputs = (lm_logits,) + lm_transformer_outputs[1:]
             # How to include echo states? Difficult with tuples.
             return outputs

        return SplitTaskOutput(
            loss=None, # Loss calculated in Trainer
            lm_logits=lm_logits,
            echo_hidden_states=hidden_states_echo,
            # Pass through standard outputs from LM path
            past_key_values=lm_transformer_outputs.past_key_values,
            hidden_states=lm_transformer_outputs.hidden_states, # Hidden states from LM path
            attentions=lm_transformer_outputs.attentions,      # Attentions from LM path
        )


# --- CustomTrainer ---
class CustomTrainer(Trainer):
    def __init__(self, *args, alpha=0.5, location_map=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.location_map = location_map
        if self.location_map is None: print("Warning: location_map not provided to CustomTrainer.")
        print(f"CustomTrainer initialized with alpha = {self.alpha}")
        if not hasattr(self, 'args') or self.args is None:
            raise ValueError("CustomTrainer initialized without TrainingArguments (self.args).")

    # --- MODIFIED compute_loss method ---
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        lm_labels = inputs.pop("labels", None)
        raw_echo_tensors = inputs.pop("raw_echo_tensors", None)
        # aux_labels_info is now a simple list of target class indices
        aux_target_class_indices = inputs.pop("aux_labels_info", None)

        model_inputs = {
            "input_ids": inputs.get("input_ids"),
            "attention_mask": inputs.get("attention_mask"),
            "raw_echo_tensors": raw_echo_tensors,
            "return_dict": True
        }
        # Filter out None values to prevent issues if a key is not present (e.g., raw_echo_tensors if no aux data for batch)
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None or k in ["input_ids", "attention_mask"]}

        # Assuming SplitTaskOutput is defined and imported correctly
        outputs: "SplitTaskOutput" = model(**model_inputs)


        # --- 1. Compute Standard LM Loss ---
        lm_loss = torch.tensor(0.0, device=outputs.lm_logits.device)
        if lm_labels is not None and outputs.lm_logits is not None:
            logits = outputs.lm_logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct_lm = CrossEntropyLoss()
            shift_labels = shift_labels.to(shift_logits.device)
            current_lm_loss = loss_fct_lm(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            if not torch.isnan(current_lm_loss):
                lm_loss = current_lm_loss
            else:
                # Add rank info to warning for better DDP debugging
                rank_info = f" (Rank {self.args.process_index})" if hasattr(self.args, 'process_index') else ""
                print(f"Warning{rank_info}: LM Loss is NaN!")




        # --- 2. Compute Auxiliary Classification Loss ---
        aux_loss = torch.tensor(0.0, device=outputs.lm_logits.device) # Default to 0
        if aux_target_class_indices and outputs.echo_hidden_states is not None and hasattr(model, "aux_classifier"):
            # print("I am here!!!")

            hidden_states_echo = outputs.echo_hidden_states
            selected_states_for_aux = []
            target_aux_labels = []
            # sequence_length = hidden_states_echo.shape[1]

            for batch_idx, aux_target_label in aux_target_class_indices:
                 # Check bounds using the echo hidden state dimensions
                #  if 0 <= batch_idx < hidden_states_echo.shape[0] and 0 <= cls_token_idx < sequence_length:
                selected_states_for_aux.append(hidden_states_echo[batch_idx, -1, :])
                target_aux_labels.append(aux_target_label)
            


            if selected_states_for_aux:
                selected_states_tensor = torch.stack(selected_states_for_aux)
                target_labels_tensor = torch.tensor(target_aux_labels, dtype=torch.long, device=selected_states_tensor.device)

                # Pass CLS states through the aux classifier
                aux_logits = model.aux_classifier(selected_states_tensor)
                loss_fct_aux = CrossEntropyLoss()
                current_aux_loss = loss_fct_aux(aux_logits, target_labels_tensor)

                if not torch.isnan(current_aux_loss):
                    aux_loss = current_aux_loss
                else:
                    print("Warning: Auxiliary Loss is NaN!")



        # # --- 2. Compute Auxiliary Classification Loss ---
        # aux_loss = torch.tensor(0.0, device=outputs.lm_logits.device)
        # # Check if there's data for the auxiliary task
        # if aux_target_class_indices and outputs.echo_hidden_states is not None and hasattr(model, "aux_classifier"):
        #     hidden_states_echo = outputs.echo_hidden_states # Shape: [N_echo, SeqLen_echo, HiddenDim]

        #     # N_echo should be len(raw_echo_tensors) and also len(aux_target_class_indices)
        #     if hidden_states_echo.shape[0] != len(aux_target_class_indices):
        #         rank_info = f" (Rank {self.args.process_index})" if hasattr(self.args, 'process_index') else ""
        #         print(f"Warning{rank_info}: Mismatch in echo hidden states batch size ({hidden_states_echo.shape[0]}) "
        #               f"and aux target labels ({len(aux_target_class_indices)}). Skipping aux loss for this batch.")
        #     else:
        #         # We need to select the correct hidden state for classification for each item in the echo path.
        #         # Assuming the last hidden state of the echo sequence is used for classification.
        #         # The model's echo path produced hidden_states_echo of shape [N_echo, SeqLen_echo, HiddenDim]
        #         # where N_echo is the number of items that had echoes processed.
                
        #         # Select the state for classification (e.g., last token's hidden state from echo path)
        #         # The echo path in your model GPTNeoForCausalLM_RawEcho_WithAux takes `encoded_echo_groups`
        #         # of shape [N_echo_samples, 5, self.echo_embed_dim] and passes it to self.transformer.
        #         # So, hidden_states_echo will be [N_echo_samples, 5, config.hidden_size].
        #         # We usually take the state of the last "token" in this sequence.
        #         states_for_classification = hidden_states_echo[:, -1, :] # Takes all N_echo samples, last token state

        #         target_labels_tensor = torch.tensor(aux_target_class_indices, dtype=torch.long, device=states_for_classification.device)
                
        #         # # Your debug print (now uses rank for clarity in DDP)
        #         # print_condition = random.random() < 0.5 # Print very infrequently
        #         # if torch.distributed.is_initialized() and print_condition:
        #         #    print(f"target_labels_tensor (Rank {self.args.process_index}): ", target_labels_tensor)

        #         aux_logits = model.aux_classifier(states_for_classification)
        #         loss_fct_aux = CrossEntropyLoss()

        #         print("aux_logits: ",aux_logits, aux_logits.shape)
        #         print("target_labels_tensor: ",target_labels_tensor, target_labels_tensor.shape)
        #         current_aux_loss = loss_fct_aux(aux_logits, target_labels_tensor)

        #         if not torch.isnan(current_aux_loss):
        #             aux_loss = current_aux_loss
        #         else:
        #             rank_info = f" (Rank {self.args.process_index})" if hasattr(self.args, 'process_index') else ""
        #             print(f"Warning{rank_info}: Auxiliary Loss is NaN!")
        
        # --- 3. Combine Losses ---
        total_loss = lm_loss + self.alpha * aux_loss

        # Optional Debug print (controlled by rank 0 and random chance)
        if random.random() < 0.01:
            lm_item = lm_loss.item() if torch.is_tensor(lm_loss) else lm_loss # lm_loss can be 0.0
            aux_item = aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss
            total_item = total_loss.item() if torch.is_tensor(total_loss) else total_loss
            print(f"compute_loss (Rank {self.args.process_index})- LM: {lm_item:.4f}, Aux: {aux_item:.4f} (alpha={self.alpha}), Total: {total_item:.4f}")
            
        return (total_loss, outputs) if return_outputs else total_loss
    


# def custom_collate_fn(batch, tokenizer, location_map, hf_echo_dataset, echo_index_map):
#     """
#     Processes batch for pre-training with a mixed strategy for echo location selection:
#     - Approx. 50% of items: Try to find LOC_XXX_ in text, fallback to random.
#     - Approx. 50% of items: Always use random location.
#     """
#     # --- Text Processing (Standard) ---
#     texts = [example["text"] for example in batch]
#     encodings = tokenizer(texts, truncation=True, max_length=1024, padding="max_length", return_tensors="pt")
#     input_ids = encodings["input_ids"]
#     attention_mask = encodings["attention_mask"]

#     # --- Standard LM Labels (Standard) ---
#     lm_labels = input_ids.clone()
#     pad_token_id = tokenizer.pad_token_id
#     for i in range(len(texts)): # Using len(texts) as batch_size for this loop
#         label_list = lm_labels[i].tolist()
#         try:
#              pad_idx = label_list.index(pad_token_id)
#              if pad_idx >= 0: label_list[pad_idx:] = [-100] * (len(label_list) - pad_idx)
#         except ValueError: pass # No padding in this sequence
#         lm_labels[i] = torch.tensor(label_list, dtype=torch.long)

#     # --- Prepare Echo Path Data ---
#     location_token_pattern = re.compile(r'LOC_(\d+)_')
#     raw_echo_groups_to_batch = [] # List to store processed echo tensors for the echo path
#     aux_target_labels = []      # List to store corresponding target labels for the aux task

#     valid_location_keys_list = list(location_map.keys())
#     if not valid_location_keys_list:
#         # This might be printed by multiple workers if not careful, but it's a one-time setup check.
#         # print("Warning (collator): No valid locations available in location_map for random selection. Aux task might be skipped.")
#         pass # Proceed, aux task will be empty if this list remains empty.

#     num_items_in_batch = len(texts)
    
#     # Determine which items use which strategy for this specific batch
#     # These are the original indices from the input `batch` (0 to num_items_in_batch-1)
#     item_indices_for_strategy_assignment = list(range(num_items_in_batch))
#     random.shuffle(item_indices_for_strategy_assignment) # Ensures random assignment of strategies

#     # Roughly half (e.g., 4 out of 9, or 5 out of 9 for text-based)
#     # Let num_text_attempt_strategy be ceil(num_items_in_batch / 2)
#     # Or, as per "another half from random (4 out of 9)", let num_always_random be floor.
#     num_always_random_strategy = num_items_in_batch // 3
    
#     indices_for_always_random_strategy = set(item_indices_for_strategy_assignment[:num_always_random_strategy])
#     # The remaining items in item_indices_for_strategy_assignment will use the "text_then_fallback"

#     for original_batch_idx in range(num_items_in_batch):
#         current_input_ids_list = input_ids[original_batch_idx].tolist() # For text parsing
        
#         echo_source_key = None
#         final_aux_target_label_for_item = None
#         label_determined_this_item = False

#         # Decide strategy for current_item (original_batch_idx)
#         use_always_random_strategy_for_this_item = original_batch_idx in indices_for_always_random_strategy

#         if use_always_random_strategy_for_this_item:
#             # --- Strategy: Always Random ---
#             if valid_location_keys_list:
#                 # This random.choice will use the worker-specific RNG
#                 random_loc_key = random.choice(valid_location_keys_list)
#                 echo_source_key = random_loc_key
#                 final_aux_target_label_for_item = location_map[random_loc_key]
#                 label_determined_this_item = True
#         else:
#             # --- Strategy: Text-based, then Fallback Random ---
#             possible_loc_candidates_from_text = []
#             for token_id in current_input_ids_list:
#                 if token_id == tokenizer.pad_token_id:
#                     break
#                 token_str = tokenizer.decode([token_id]) # Decoding single tokens can be slow; consider alternatives if performance is an issue
#                 match = location_token_pattern.fullmatch(token_str)
#                 if match:
#                     location_num_str = match.group(1)
#                     try:
#                         location_key_from_text = int(location_num_str)
#                         # Check if this location_key is valid (in filtered map and has enough samples)
#                         if location_key_from_text in location_map and \
#                            location_key_from_text in echo_index_map and \
#                            len(echo_index_map[location_key_from_text]) >= 5: # Min 5 samples needed for echo group
#                             possible_loc_candidates_from_text.append(location_key_from_text)
#                     except ValueError:
#                         pass # Invalid number in LOC token
            
#             if possible_loc_candidates_from_text:
#                 # If multiple LOC tokens found in text, pick one randomly
#                 selected_loc_key = random.choice(possible_loc_candidates_from_text)
#                 echo_source_key = selected_loc_key
#                 final_aux_target_label_for_item = location_map[selected_loc_key]
#                 label_determined_this_item = True
            
#             # Fallback for "Text-based" strategy if no suitable LOC token was found in its text
#             if not label_determined_this_item and valid_location_keys_list:
#                 random_loc_key = random.choice(valid_location_keys_list)
#                 echo_source_key = random_loc_key
#                 final_aux_target_label_for_item = location_map[random_loc_key]
#                 label_determined_this_item = True

#         # --- Fetch Echoes if a location key was successfully determined for this item ---
#         if label_determined_this_item and echo_source_key is not None:
#             available_ds_indices_for_echo = echo_index_map.get(echo_source_key, [])
            
#             if len(available_ds_indices_for_echo) >= 5: # We need 5 echoes to form a group
#                 # This random.sample will use the worker-specific RNG
#                 sampled_ds_indices = random.sample(available_ds_indices_for_echo, 5)
#                 try:
#                     # Ensure echo_array is float32 for the BatNavEncoder
#                     echo_arrays_np = [np.array(hf_echo_dataset[idx]['echo_array'], dtype=np.float32) for idx in sampled_ds_indices]
#                     # Stack to [5, 6, 15000] (assuming echo_array is [6, 15000])
#                     stacked_echo_group_np = np.stack(echo_arrays_np, axis=0) 
                    
#                     # Add channel dimension for Conv2D input: [5, 1, 6, 15000]
#                     echo_tensor_for_item = torch.tensor(stacked_echo_group_np, dtype=torch.float32).unsqueeze(1)
                    
#                     raw_echo_groups_to_batch.append(echo_tensor_for_item)
#                     aux_target_labels.append(final_aux_target_label_for_item)
#                 except Exception as e:
#                     # Log error if necessary, but don't let it crash the whole batch
#                     print(f"Error fetching/stacking echoes for loc {echo_source_key} (orig_idx {original_batch_idx}): {e}. Skipping aux for this item.")
#             # else:
#                 # Not enough samples for this chosen location_key, so skip aux for this item.
#                 # print(f"Warning (collator): Loc key {echo_source_key} (orig_idx {original_batch_idx}) selected, "
#                 #       f"but only {len(available_ds_indices_for_echo)} samples in echo_index_map (expected >= 5). Skipping aux for this item.")
#                 pass

#     # --- Prepare final batch dictionary ---
#     batch_dict = {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "labels": lm_labels, # LM labels for all items in the original batch
#     }

#     # Only add echo-related data if any echoes were successfully processed and added to the lists
#     if raw_echo_groups_to_batch: # Check if the list is not empty
#         # raw_echo_tensors will be a list of tensors, each of shape [5, 1, 6, 15000].
#         # The model's forward pass is expected to handle this list (e.g., by stacking them).
#         batch_dict["raw_echo_tensors"] = raw_echo_groups_to_batch
        
#         # aux_labels_info is now a simple list of target class indices,
#         # parallel to raw_echo_groups_to_batch.
#         # The CustomTrainer's compute_loss should be adapted if it expects a tuple.
#         # Based on your previous CustomTrainer, it iterates len(aux_labels_info)
#         # and accesses aux_labels_info[echo_sample_idx] directly for the target.
#         batch_dict["aux_labels_info"] = aux_target_labels
    
#     # The `echo_batch_indices_map` from your original snippet is removed as it wasn't
#     # clearly used by the provided model or trainer logic for loss calculation.
#     # If it's needed for something else, it would map indices from `raw_echo_groups_to_batch`
#     # back to `original_batch_idx`.

#     return batch_dict


# --- Collate Function Modified for Pre-training with Random Fallback ---
def custom_collate_fn(batch, tokenizer, location_map, hf_echo_dataset, echo_index_map):
    """
    Processes batch for pre-training with separate echo path:
    1. Tokenizes text.
    2. Prepares standard LM labels.
    3. Attempts to randomly select ONE valid 'LOC_XXX_' token within each sequence.
    4. **Fallback:** If no valid LOC token is found in the text, randomly selects
       a valid location key from the global map.
    5. If a valid location key (from text or random fallback) is determined
       AND echoes exist for it:
        - Retrieves raw echo arrays for that location.
        - Gathers auxiliary classification targets using that location.
    """
    # --- Text Processing ---
    texts = [example["text"] for example in batch]
    encodings = tokenizer(texts, truncation=True, max_length=1024, padding="max_length", return_tensors="pt")

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    # --- Standard LM Labels ---
    lm_labels = input_ids.clone()
    pad_token_id = tokenizer.pad_token_id
    for i in range(len(texts)):
        label_list = lm_labels[i].tolist()
        try: # Mask padding
             pad_idx = label_list.index(pad_token_id)
             if pad_idx >= 0: label_list[pad_idx:] = [-100] * (len(label_list) - pad_idx)
        except ValueError: pass
        lm_labels[i] = torch.tensor(label_list, dtype=torch.long)


    # --- Prepare Echo Path Data ---
    location_token_pattern_text = r'\[LOC_(\d+)\]'
    location_token_pattern = re.compile(location_token_pattern_text)

    raw_echo_groups_to_batch = []
    aux_labels_info = []
    echo_batch_indices_map = []

    # Get list of all valid location keys we can potentially sample from
    # These keys MUST exist in both location_map and echo_index_map with >= 5 samples
    # Assuming location_map's keys already represent this valid set based on data loading
    valid_location_keys_list = list(location_map.keys())
    if not valid_location_keys_list:
         print("Warning: No valid locations available in location_map for random fallback.")

    for i in range(len(texts)): # Iterate through the original batch
        ids_list = input_ids[i].tolist()

        echo_source_key = None
        final_aux_target_label = None
        label_determined = False # Flag to track if we found/assigned a label

        # --- Attempt 1: Find LOC token in text ---
        possible_loc_candidates = []
        for k, token_id in enumerate(ids_list):
             if token_id == tokenizer.pad_token_id: break
             token_str = tokenizer.decode([token_id])
             match = location_token_pattern.fullmatch(token_str)
             
             if match:
                location_num_str = match.group(1)
                # print("match: ", match, token_str)
                try:
                    location_key = int(location_num_str)
                    # Check validity
                    if location_key in location_map and location_key in echo_index_map and len(echo_index_map[location_key]) >= 5:
                        possible_loc_candidates.append(location_key) # Store only the key
                except ValueError: pass

        if possible_loc_candidates:
            # Randomly select one valid location found in the text
            selected_loc_key = random.choice(possible_loc_candidates)
            echo_source_key = selected_loc_key
            final_aux_target_label = location_map[selected_loc_key]
            label_determined = True

        # --- Attempt 2 (Fallback): Randomly assign if not found in text ---
        if not label_determined and valid_location_keys_list: # Only if fallback is possible
            random_loc_key = random.choice(valid_location_keys_list)
            echo_source_key = random_loc_key
            final_aux_target_label = location_map[random_loc_key]
            label_determined = True # Mark label as assigned (randomly)

        # --- Fetch Echoes (if a source key was determined) ---
        if label_determined and echo_source_key is not None:
            # We have a valid location key (echo_source_key) with echoes and a target label
            available_ds_indices = echo_index_map.get(echo_source_key, []) # Use .get for safety

            # Double-check we have enough indices (should be guaranteed by checks above)
            if len(available_ds_indices) >= 5:
                sampled_ds_indices = random.sample(available_ds_indices, 5)
                try:
                    echo_arrays_np = [np.array(hf_echo_dataset[idx]['echo_array'], dtype=np.float32) for idx in sampled_ds_indices]
                    stacked_echo_group_np = np.stack(echo_arrays_np, axis=0)
                    raw_echo_groups_to_batch.append(torch.tensor(stacked_echo_group_np, dtype=torch.float32).unsqueeze(1))
                    aux_labels_info.append((i, final_aux_target_label)) # Store original batch index 'i' and target
                    echo_batch_indices_map.append(i)
                except Exception as e:
                    print(f"Error fetching/stacking echoes for determined loc {echo_source_key}: {e}")
            # else: # This case should ideally not happen if valid_location_keys_list is built correctly
            #     print(f"Warning: Location key {echo_source_key} selected, but not enough samples found in echo_index_map.")


    # --- Prepare final batch dictionary ---
    batch_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": lm_labels,
    }
    if raw_echo_groups_to_batch:
        batch_dict["raw_echo_tensors"] = raw_echo_groups_to_batch
        batch_dict["aux_labels_info"] = aux_labels_info
        batch_dict["echo_batch_indices_map"] = echo_batch_indices_map # Include if needed downstream

    return batch_dict



# --- Main Fine-Tuning Code ---
def main():
 
    global echo_dataset_global, echo_indices_by_location_global, location_map_global

    # ******MODIFICATION START: Load preprocessed dataset ******
    # --- 1. Load Preprocessed Echo Dataset ---
    echo_dataset_path = "/scratch200/xingchen/processed_echo_dataset_parallel_with_coords" # Path from preprocessing script
    if not os.path.exists(echo_dataset_path):
        print(f"Error: Preprocessed echo dataset not found at {echo_dataset_path}")
        print("Please run the preprocessing script first.")
        sys.exit(1)
    print(f"Loading preprocessed echo dataset from {echo_dataset_path}...")
    # Use try-except for robust loading
    try:
        echo_dataset_global = load_from_disk(echo_dataset_path)
        # Optionally set format to numpy for faster retrieval if needed, though default should be efficient
        # echo_dataset_global.set_format("numpy")
        print(f"Loaded echo dataset with {len(echo_dataset_global)} samples.")
    except Exception as e:
        sys.exit(f"Error loading dataset from disk: {e}")

    # --- 2. Build and Filter In-Memory Index and Location Map ---
    raw_echo_indices_map = build_location_index_map(echo_dataset_global)

    # ****** MODIFICATION: Filter locations with less than 15 samples ******
    MIN_SAMPLES_PER_LOCATION = 15
    filtered_echo_indices_map = {}
    print(f"Filtering locations to have at least {MIN_SAMPLES_PER_LOCATION} samples...")
    for loc_key, indices in raw_echo_indices_map.items():
        if len(indices) >= MIN_SAMPLES_PER_LOCATION:
            filtered_echo_indices_map[loc_key] = indices
        # else:
            # print(f"  Filtering out location_key {loc_key}: only {len(indices)} samples.") # Can be verbose
    
    echo_indices_by_location_global = filtered_echo_indices_map
    num_filtered_locations = len(echo_indices_by_location_global)
    print(f"After filtering, {num_filtered_locations} unique locations remain with >= {MIN_SAMPLES_PER_LOCATION} samples each.")
    
    if num_filtered_locations == 0:
         print(f"Warning: No locations met the >= {MIN_SAMPLES_PER_LOCATION} sample threshold. Auxiliary task will not be active.")
         # num_aux_classes will be 0, model init might need handling or aux_loss will always be 0.
         # The model init checks for num_aux_classes > 0, so this path will lead to an error there, which is good.

    valid_locations_in_dataset = sorted(echo_indices_by_location_global.keys())
    location_map_global = {loc_id: i for i, loc_id in enumerate(valid_locations_in_dataset)}
    num_aux_classes = len(location_map_global)

    if num_aux_classes <= 0: # This check will catch if all locations were filtered out
        sys.exit(f"Error: No valid locations found for auxiliary task after filtering (min {MIN_SAMPLES_PER_LOCATION} samples/location).")
    print(f"Built location map for {num_aux_classes} aux classes from filtered dataset.")
    # ****** END MODIFICATION ******


    # --- 2. Initialize Tokenizer ---
    # ... (tokenizer loading and checks remain the same) ...
    tokenizer_path = "./ops/tokenizer_v9_2.json"; 
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, 
                                        max_len=1024, 
                                        pad_token='[PAD]', 
                                        unk_token='[UNK]', 
                                        sep_token='[SEP]', 
                                        cls_token='[CLS]', 
                                        bos_token='[BOS]',
                                        eos_token='[EOS]',
                                        additional_special_tokens=['[INST]', '[SRA]', '[POS]']); 
    print("--- Tokenizer IDs ---"); 
    print(f"PAD:{tokenizer.pad_token_id}, UNK:{tokenizer.unk_token_id}, SEP:{tokenizer.sep_token_id}, CLS:{tokenizer.cls_token_id}, INST:{tokenizer.convert_tokens_to_ids('[INST]')}"); print("-" * 21); assert all(x is not None for x in [tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.sep_token_id])

    # --- 3. Load Dataset ---
    # ... (dataset loading and splitting remains the same) ...
    dataset_path = '/scratch100/xingchen/navigation_code_without_GL/train_dataset_base_5400_v1_fully_cleaned_forbidden' # Make sure this text data has the 5 UNK pattern
    if not os.path.exists(dataset_path): sys.exit(f"Error: Dataset not found at {dataset_path}")
    dataset = load_from_disk(dataset_path); 
    total = len(dataset); 
    split_idx = int(total * 0.9997)
    train_dataset = dataset.select(range(0, split_idx)); 
    eval_dataset = dataset.select(range(split_idx, total))
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    # --- 4. Initialize Model (Raw Echo Version) ---
    # Define echo encoder configuration (MUST match BatNavEncoder_attn2 init args)
    echo_encoder_config = {
        'input_size': 15000,
        'patch_size': 150,
        'in_channels': 1,
        'embed_dim': 1024, # Match transformer hidden size
        'num_encoder_layers': 2,
        'num_heads': 8
    }
    # Load base transformer config
    # model_path = "/scratch200/xingchen/bat_grid_navigation/GPT_NEO_base3_v1_NO_PE"
    # model_path = "/scratch200/xingchen/bat_grid_navigation/GPT_NEO_finetune_RAW_ECHO_AUX_v1/checkpoint-1000_v1"
    # model_path ="/scratch200/xingchen/bat_grid_navigation/GPT_NEO_base_finetune_RAW_ECHO_AUX_v1_separate/checkpoint-3500"
    model_path = "/scratch100/xingchen/navigation_code_without_GL/GPT_NEO_base_AUX_v1_location_v4token/checkpoint-10"
    if not os.path.exists(model_path): 
        sys.exit(f"Error: Base model not found at {model_path}")

    # # Initialize the new model class
    model = GPTNeoForCausalLM_RawEcho_WithAux.from_pretrained( # Use from_pretrained if loading trained checkpoint
        model_path, # Load from the checkpoint directory
        num_aux_classes=num_aux_classes, # Required by __init__
        echo_encoder_config=echo_encoder_config, # Required by __init__
        torch_dtype=torch.float16,
        # ignore_mismatched_sizes=True # Probably NOT needed if loading full trained checkpoint
    )

    # # config = AutoConfig.from_pretrained(model_path) 
    # hidden_size = 1024
    # config = GPTNeoConfig(
    #     vocab_size=tokenizer.vocab_size,
    #     max_position_embeddings=1024,
    #     hidden_size=hidden_size,
    #     num_layers=24,
    #     attention_types=[[['global', 'local'], 12]],
    #     num_head=8,
    #     activation_function="gelu_new",
    #     intermediate_size=int(hidden_size * 4),
    #     resid_dropout=0.1,
    #     embed_dropout=0.1,
    #     layer_norm_epsilon=1e-5,
    #     initializer_range=0.02,
    #     scale_attn_weights=True,
    #     use_cache=True,
    #     bos_token_id=tokenizer.encode('[BOS]')[0],
    #     eos_token_id=tokenizer.encode('[EOS]')[0],
    #     pad_token_id=tokenizer.encode('[PAD]')[0],
    # )
    

    # # sys.exit(1)

    # # Initialize the new model class
    # model = GPTNeoForCausalLM_RawEcho_WithAux(
    #     config=config,
    #     num_aux_classes=num_aux_classes,
    #     echo_encoder_config=echo_encoder_config,
    #     # NOTE: Weights for echo_encoder and aux_classifier will be randomly initialized
    #     # You might want to load base GPT-Neo weights first if needed
    #     # e.g., model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin')), strict=False)
    #     # Or fine-tune everything from scratch.
    #     # torch_dtype=torch.float16,
    # )

    # --- 5. Trainer Config ---
    # ... (TrainingArguments - maybe new output dir) ...
    output_dir = "/scratch100/xingchen/navigation_code_without_GL/GPT_NEO_base_AUX_v1_location_v4token" # New output dir
    batch_size = 8#16 # Reduce batch size significantly due to echo loading/encoding
    gradient_accumulation_steps = 6 # Increase accumulation
    deepspeed_config = './ops/ds_config3_6_large_lr.json' # Check if memory still fits
    if not os.path.exists(deepspeed_config): deepspeed_config = None

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy='steps',
        logging_strategy='steps',
        logging_steps=1000,
        eval_steps=1000,
        weight_decay=0.1,
        learning_rate=3.0e-4,
        adam_beta1=0.9,
        adam_beta2=0.95,
        save_strategy='steps',
        save_steps=1000,
        save_total_limit=3,
        fp16=True,
        bf16=False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        deepspeed=deepspeed_config,
        dataloader_drop_last=True,
        dataloader_num_workers=4,
        max_grad_norm=100.0, # Adjusted from 100
        # report_to="wandb", # Optional
        # run_name="gpt-neo-aux-loss-v1", # Optional
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    # ******MODIFICATION START: Updated partial call ******
    # --- 7. Initialize Custom Trainer ---
    # Collator now needs the dataset objects and the index map
    data_collator_with_args = partial(
        custom_collate_fn,
        tokenizer=tokenizer,
        location_map=location_map_global, # Pass the map for aux labels
        hf_echo_dataset=echo_dataset_global, # Pass the loaded dataset
        echo_index_map=echo_indices_by_location_global # Pass the index map
    )
    # ******MODIFICATION END******
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator_with_args,
        tokenizer=tokenizer,
        alpha=0.1, # Tune this weight
        location_map=location_map_global
    )

    # --- 7. Train ---
    print("Starting training with Raw Echo Encoding and Auxiliary Loss...")
    # ... (Training loop remains the same) ...
    # try:
    train_result = trainer.train()
    # train_result = trainer.train(resume_from_checkpoint=True)
    trainer.save_model()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print("Training finished successfully.")
    # except Exception as e:
    #     print(f"An error occurred during training: {e}")
    #     raise e


if __name__ == '__main__':
    main()