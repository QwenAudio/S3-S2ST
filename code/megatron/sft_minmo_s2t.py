"""SFT MinMo S2T Model."""

import os
import torch
import torch.nn.functional as F
from functools import partial
from contextlib import nullcontext
import inspect
from typing import Union, Optional, List

from megatron.core.models.multimodal.minmo_s2t_model import MinMoS2TModel
from megatron.core.parallel_state import get_tensor_model_parallel_world_size, get_tensor_model_parallel_src_rank, \
    get_tensor_model_parallel_group, get_tensor_model_parallel_rank
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.fun_asr_dataset import FunASRSFTDataset, FunASRDatasetConfig
# import megatron.legacy.model
# from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank, average_losses_across_data_parallel_group,
)
from megatron.training.arguments import (
    core_transformer_config_from_args, 
    core_audio_model_config_from_args,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.multimodal.minmo_s2t_spec import (
    get_minmo_s2t_audio_encoder_spec,
    get_minmo_s2t_audio_adaptor_spec
)

stimer = StragglerDetector()

# This is used to bypass Triton Error temporarily
# https://github.com/pytorch/pytorch/issues/124565
torch.empty(1, device='cuda', requires_grad=True).backward()

def model_provider(pre_process=True, post_process=True):
    """Builds the MinMo S2T model."""

    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building MinMo S2T model ...')
    
    # Get configurations
    config = core_transformer_config_from_args(args)
    audio_config = core_audio_model_config_from_args(args)

    if args.use_legacy_models:
        raise NotImplementedError("Legacy models not supported for MinMo S2T")
    else: # using core models
        if use_te:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, 
                args.multi_latent_attention, args.fp8
            )
            audio_encoder_spec = get_minmo_s2t_audio_encoder_spec(use_te=True)
        else:
            transformer_layer_spec = get_gpt_layer_local_spec(
                args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, 
                args.multi_latent_attention
            )
            audio_encoder_spec = get_minmo_s2t_audio_encoder_spec(use_te=False)
        
        audio_adaptor_spec = get_minmo_s2t_audio_adaptor_spec(use_te)
        
        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init
                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except:
                raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")

        with build_model_context(**build_model_context_args):
            model = MinMoS2TModel(
                config=config,
                audio_config=audio_config,
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
                rope_scaling=args.use_rope_scaling,
                seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
                length_normalized_loss=args.length_normalized_loss,
                audio_encoder_activation_checkpoint=args.audio_encoder_activation_checkpoint,
                feat_permute=args.feat_permute,
            )

    return model


def get_batch(data_iterator):
    """Generate a batch for MinMo S2T training."""
    args = get_args()
    tokenizer = get_tokenizer()

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    if data is not None and any(
        x not in data for x in ["speech", "speech_lengths", "fbank_beg", "fbank_mask", "fake_token_len"]
    ):
        data["speech"] = None
        data["speech_lengths"] = None
        data["fbank_beg"] = None
        data["fbank_mask"] = None
        data["fake_token_len"] = None

    if data is not None:
        data_list = [data["speech"], data["speech_lengths"], data["fbank_beg"], 
                    data["fbank_mask"], data["fake_token_len"]]
    else:
        data_list = [None, None, None, None, None]
    
    if get_tensor_model_parallel_world_size() > 1:
        torch.distributed.broadcast_object_list(
            data_list,
            get_tensor_model_parallel_src_rank(),
            get_tensor_model_parallel_group(),
        )
    speech, speech_lengths, fbank_beg, fbank_mask, fake_token_len = data_list

    # Unpack text data
    tokens_ = tensor_parallel.broadcast_data(['input_ids'], data, torch.int64)["input_ids"]
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    
    # Get position ids
    # print (f'position embedding type: {args.position_embedding_type}')
    position_ids = None
    if args.position_embedding_type in ['rope', 'learned_absolute']:
        position_ids = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0).expand_as(tokens)
    assert position_ids is not None

    # Get attention mask
    attention_mask = None
    if args.attention_dropout > 0:
        attention_mask = torch.ones_like(tokens, dtype=torch.bool)

    # Get loss mask
    loss_mask = data.get("loss_mask", None)
    if loss_mask is not None:
        loss_mask = loss_mask[:, 1:].contiguous()
    else:
        loss_mask = torch.ones_like(labels, dtype=torch.bool)

    return (
        tokens,
        labels,
        loss_mask,
        attention_mask,
        position_ids,
        speech,
        speech_lengths,
        fbank_beg,
        fbank_mask,
        fake_token_len,
    )


def loss_func(loss_mask, output_tensor):
    """Loss function for MinMo S2T model."""
    args = get_args()
    
    # 直接使用output_tensor作为损失，与VL模型保持一致
    losses = output_tensor.float()
    loss_mask = loss_mask.float()
    ele_loss = losses * loss_mask
    loss = ele_loss.sum() / loss_mask.sum()
    
    # 处理损失归约和日志记录
    if args.reduce_loss_per_iteration:
        averaged_loss = [loss.clone().detach().view(1)]
    else:
        averaged_loss = average_losses_across_data_parallel_group([loss])

    info = {'lm_loss': averaged_loss[0]}
    return loss, info



def forward_step(data_iterator, model: MinMoS2TModel):
    """Forward training step for MinMo S2T model.

    Args:
        data_iterator : Input data iterator
        model (MinMoS2TModel): The MinMo S2T Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            speech,
            speech_lengths,
            fbank_beg,
            fbank_mask,
            fake_token_len,
        ) = get_batch(data_iterator)
    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            speech=speech,
            speech_lengths=speech_lengths,
            fbank_beg=fbank_beg,
            fbank_mask=fbank_mask,
            labels=labels,
            fake_token_len=fake_token_len,  # 这个参数会被放入 **kwargs
        )
    
    return output_tensor, partial(loss_func, loss_mask)


def core_audio_dataset_config_from_args(args):
    """Create audio dataset configuration from arguments."""
    tokenizer = get_tokenizer()

    return FunASRDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.train_data_path),
        blend_per_split=None,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        # audio_config=args.audio_config,
        # feat_permute=args.feat_permute,
    )


def is_dataset_built_on_rank():
    """Check if dataset should be built on this rank."""
    return mpu.get_tensor_model_parallel_rank() == 0


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets for MinMo S2T."""
    args = get_args()
    config = core_audio_dataset_config_from_args(args)
    dataset_type = FunASRSFTDataset
    print_rank_0('> building train, validation, and test datasets for MinMo S2T ...')

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    if train_ds:
        print_rank_0(train_ds[0])

    print_rank_0("> finished creating MinMo S2T datasets ...")
    return train_ds, valid_ds, test_ds


def add_minmo_s2t_args(parser):
    """Add MinMo S2T specific arguments."""
    group = parser.add_argument_group(title="MinMo S2T")
    
    # Audio processing parameters
    group.add_argument('--audio-input-size', type=int, default=80,
                       help='Input feature dimension for audio')
    group.add_argument('--audio-linear-units', type=int, default=1280,
                       help='Hidden dimension for audio encoder')
    group.add_argument('--audio-attention-heads', type=int, default=16,
                       help='Number of attention heads for audio encoder')
    group.add_argument('--audio-num-blocks', type=int, default=24,
                       help='Number of encoder blocks for audio')
    group.add_argument('--audio-downsample-rate', type=int, default=4,
                       help='Downsampling rate for audio features')
    group.add_argument('--audio-use-padmask', action='store_true',
                       help='Whether to use padding mask for audio')
    
    # Audio adaptor parameters
    group.add_argument('--audio-encoder-dim', type=int, default=1280,
                       help='Output dimension of audio encoder')
    group.add_argument('--audio-llm-dim', type=int, default=4096,
                       help='Input dimension for language model')
    group.add_argument('--audio-ffn-dim', type=int, default=2048,
                       help='Hidden dimension for audio adaptor')
    group.add_argument('--audio-n-layer', type=int, default=2,
                       help='Number of layers in audio adaptor')
    group.add_argument('--audio-attention-dropout-rate', type=float, default=0.0,
                       help='Dropout rate for attention in audio adaptor')
    group.add_argument('--audio-dropout-rate', type=float, default=0.0,
                       help='General dropout rate for audio adaptor')
    
    # Attention configuration
    group.add_argument('--audio-att-type', type=str, default="self_att_fsmn_sdpa",
                       help='Attention type to use for audio')
    group.add_argument('--audio-kernel-size', type=int, default=15,
                       help='Kernel size for FSMN')
    group.add_argument('--audio-sanm-shift', type=int, default=0,
                       help='Shift parameter for SANM')
    group.add_argument('--audio-max-position-embeddings', type=int, default=2048,
                       help='Maximum position embeddings for RoPE in audio')
    group.add_argument('--audio-rope-theta', type=float, default=10000,
                       help='Theta parameter for RoPE in audio')
    
    # Training parameters
    group.add_argument('--length-normalized-loss', action='store_true',
                       help='Whether to use length normalized loss')
    group.add_argument('--audio-encoder-activation-checkpoint', action='store_true',
                       help='Whether to use activation checkpointing for audio encoder')
    group.add_argument('--feat-permute', action='store_true', default=True,
                       help='Whether to permute features for audio encoder')
    
    # Dataset parameters
    group.add_argument('--fun-audio-dataset', action='store_true',
                       help='Use FunASR audio dataset')
    group.add_argument('--fun-audio-config', type=str, default=None,
                       help='Path to FunASR audio dataset configuration')
    
    return parser



if __name__ == "__main__":
    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_minmo_s2t_args,
        args_defaults={'tokenizer_type': 'QWenVLTokenizer'},
    )