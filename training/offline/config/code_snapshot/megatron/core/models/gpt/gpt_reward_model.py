# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import inspect
from collections import OrderedDict
from typing import Dict, Literal, Optional, List
from contextlib import nullcontext

import torch
from torch import Tensor

from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, import_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import get_linear_layer

class LinearPooler(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(
        self,
        hidden_size: int,
        init_method: callable,
        config: TransformerConfig,
        score_dimensions: int,
        sequence_parallel: bool = False,
    ):
        super().__init__(config=config)
        self.dense1 = get_linear_layer(hidden_size, hidden_size, init_method, config.perform_initialization)
        self.dense2 = get_linear_layer(hidden_size, score_dimensions, init_method, config.perform_initialization)
        self.sequence_parallel = sequence_parallel

    def forward(self, hidden_states, sequence_indices=None):
        # hidden_states: [s, b, h]
        # sequence_index: index of the token to pool.

        # gather data along sequence dimensions
        # same pooler is run on all tensor parallel nodes
        if self.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, # [s, b, h]
                tensor_parallel_output_grad=False)

        if sequence_indices is not None:
            selected_hidden = torch.index_select(hidden_states, 0, sequence_indices)
            selected_hidden = selected_hidden.diagonal(dim1=0, dim2=1).T
            pooled = self.dense2(torch.nn.functional.relu(self.dense1(selected_hidden)))
        else:
            selected_hidden = hidden_states # [s, b, h]
            pooled = self.dense2(torch.nn.functional.relu(self.dense1(selected_hidden))).squeeze() # [s, b, scoredim]

        return pooled


class RewardModel(GPTModel):
    """GPT Transformer language model.

    Args:
        config (TransformerConfig):
            Transformer config
        transformer_layer_spec (ModuleSpec):
            Specifies module to use for transformer layers
        vocab_size (int):
            Vocabulary size
        max_sequence_length (int):
            maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional):
            Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional):
            Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional):
            Defaults to False.
        parallel_output (bool, optional):
            Do not gather the outputs, keep them split across tensor
            parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional):
            When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):
            Position embedding type.. Defaults to 'learned_absolute'.
        rotary_percent (float, optional):
            Percent of rotary dimension to use for rotary position embeddings.
            Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional):
            Base period for rotary position embeddings. Ignored unless
            position_embedding_type is 'rope'.
            Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional):
            scale of linearly interpolating RoPE for longer sequences.
            The value must be a float larger than 1.0. Defaults to None.
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        seq_len_interpolation_factor: Optional[float] = None,
        score_dimension: int = 1
    ) -> None:
        super().__init__(config=config, transformer_layer_spec=transformer_layer_spec, vocab_size=vocab_size,
                         max_sequence_length=max_sequence_length, pre_process=pre_process, post_process=post_process,
                         fp16_lm_cross_entropy=fp16_lm_cross_entropy, parallel_output=parallel_output,
                         share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                         position_embedding_type=position_embedding_type, rotary_percent=rotary_percent,
                         rotary_base=rotary_base, rope_scaling=rope_scaling,
                         seq_len_interpolation_factor=seq_len_interpolation_factor)

        # Output
        if post_process:
            self.pooler_head = LinearPooler(config.hidden_size, config.init_method, config, score_dimension,
                                            config.sequence_parallel)
            self._pooler_head_key = 'pooler_head'
        else:
            self._pooler_head_key = None

        if has_config_logger_enabled(self.config):
            log_config_to_disk(
                self.config, self.state_dict(), prefix=f'{type(self).__name__}_init_ckpt'
            )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        pooling_sequence_index: int = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        if not self.post_process:
            return hidden_states


        pooled_output = self.pooler_head(hidden_states, pooling_sequence_index)
        if labels is not None:
            # logits and loss
            output_weight = None
            if self.share_embeddings_and_output_weights:
                output_weight = self.shared_embedding_or_output_weight()
            logits, _ = self.output_layer(
                hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
            )

            if has_config_logger_enabled(self.config):
                payload = OrderedDict(
                    {
                        'input_ids': input_ids,
                        'position_ids': position_ids,
                        'attention_mask': attention_mask,
                        'decoder_input': decoder_input,
                        'logits': logits,
                    }
                )
                log_config_to_disk(self.config, payload, prefix='input_and_logits')
            loss = self.compute_language_model_loss(labels, logits)

            return [loss, pooled_output]
        else:
            # inference only:
            return pooled_output

        return loss

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        state_dict_ = super().state_dict_for_save_checkpoint(prefix, keep_vars)

        if self.post_process:
            state_dict_[self._pooler_head_key] \
                = self.pooler_head.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):# pylint: disable=unused-argument
        """Customized load."""
        # Directly utilize super().load_state_dict(state_dict, strict=True) causes exceptions. This is
        # because if the base torch.nn.Module method load_state_dict is invoked, a strict key
        # matching mechanism will be enforced across all parameters. It will become
        # problematic when training a reward model derived from a SFT checkpoint, as
        # these checkpoints typically lack the state_dict for pooler_head in the reward model.
        incompatible_keys = super().load_state_dict(state_dict, strict=False)
        if len(incompatible_keys.missing_keys) == 0 and len(incompatible_keys.unexpected_keys) == 0:
            print_rank_0("load reward model pooler_head success")
            return
        elif self.post_process:
            if all(missing_key.startswith(self._pooler_head_key) for missing_key in incompatible_keys.missing_keys):
                print_rank_0("cannot load reward model pooler_head, init from random")
            if all(unexpected_key.startswith("output_layer") for unexpected_key in incompatible_keys.unexpected_keys):
                print_rank_0("neglect output_layer weight for reward model")
            return
        else:
            error_msgs: List[str] = []
            if len(incompatible_keys.unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join(f'"{k}"' for k in incompatible_keys.unexpected_keys)))
            if len(incompatible_keys.missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join(f'"{k}"' for k in incompatible_keys.missing_keys)))
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                            self.__class__.__name__, "\n\t".join(error_msgs)))


def model_provider(pre_process=True, post_process=True) -> RewardModel:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building Reward model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    # using core models
    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        if use_te:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, args.multi_latent_attention, args.fp8)
        else:
            transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, args.multi_latent_attention)

    build_model_context = nullcontext
    build_model_context_args = {}
    if args.fp8_param_gather:
        try:
            from transformer_engine.pytorch import fp8_model_init

            build_model_context = fp8_model_init
            build_model_context_args["enabled"] = True

            # Check if fp8_model_init supports preserve_high_precision_init_val
            if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                build_model_context_args["preserve_high_precision_init_val"] = True
        except:
            raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")

    with build_model_context(**build_model_context_args):
        model = RewardModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling
        )

    return model
