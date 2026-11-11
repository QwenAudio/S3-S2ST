"""SFT Qwen2.5-VL."""

import os
import torch
import torch.nn.functional as F
from functools import partial
from contextlib import nullcontext
import inspect

from typing import Union

from megatron.core.models.multimodal.vl2_5_model import Qwen2_5_VLModel
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
from megatron.core.datasets.qwenvl_dataset import QwenVLSFTDataset, VLSFTDatasetConfig
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank, average_losses_across_data_parallel_group,
)
from megatron.training.arguments import (
    core_transformer_config_from_args, 
    core_qwen2_5_vl_vision_model_config_from_args,
    core_mlp_projection_config_from_args
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.multimodal.qwenvl_spec import (
    get_qwen2vl_vision_model_with_transformer_engine_spec,
    get_qwen2vl_vision_model_local_spec,
    get_qwen2vl_vision_projector_spec
)

stimer = StragglerDetector()

# This is used to bypass Triton Error temporaly
# https://github.com/pytorch/pytorch/issues/124565
torch.empty(1, device='cuda', requires_grad=True).backward()

def model_provider(pre_process=True, post_process=True):
    """Builds the model."""

    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building Qwen2.5-VL model ...')
    # Experimental loading arguments from yaml
    config = core_transformer_config_from_args(args)
    vision_config = core_qwen2_5_vl_vision_model_config_from_args(args)
    projection_config = core_mlp_projection_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else: # using core models

        if use_te:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, args.multi_latent_attention, args.fp8)
            vision_model_spec = get_qwen2vl_vision_model_with_transformer_engine_spec()
        else:
            transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, args.multi_latent_attention)
            vision_model_spec = get_qwen2vl_vision_model_local_spec()
        vision_projector_spec = get_qwen2vl_vision_projector_spec(use_te)
        
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
            model = Qwen2_5_VLModel(
                config=config,
                vision_config=vision_config,
                projection_config=projection_config,
                transformer_layer_spec=transformer_layer_spec,
                vision_model_spec=vision_model_spec,
                projection_layer_spec=vision_projector_spec,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
                rotary_base=1000000,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                position_embedding_type=args.position_embedding_type,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights
            )

    return model


def get_llm_pos_ids(
    tokens: torch.Tensor,
    vision_seqlens: list[int],
    grid_ts: list[int],
    grid_hs: list[int],
    grid_ws: list[int],
    image_modality_token_id: int,
) -> torch.Tensor:
    args = get_args()
    llm_pos_ids_list = []
    idx = 0
    start = 0
    while True:
        try:
            end = tokens.index(image_modality_token_id, start)
        except:
            break

        vision_seqlen = vision_seqlens[idx]
        llm_grid_t = grid_ts[idx]
        llm_grid_h = grid_hs[idx] // args.spatial_merge_size
        llm_grid_w = grid_ws[idx] // args.spatial_merge_size
        text_len = end - start

        start_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + start_idx)
        t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
        h_index = (
            torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
        )
        w_index = (
            torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
        )
        _llm_pos_ids = torch.stack([t_index, h_index, w_index])
        llm_pos_ids_list.append(_llm_pos_ids + text_len + start_idx)
        # llm_pos_ids_list.append(
        #     torch.tensor([_llm_pos_ids.max().item() + text_len + start_idx] * 3).unsqueeze(1)
        # )

        idx += 1
        start = end + vision_seqlen

    if start < len(tokens):
        start_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
        text_len = len(tokens) - start
        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + start_idx)

    llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
    return llm_pos_ids


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    if data is not None and any(
        x not in data for x in ["grid_ts", "grid_hs", "grid_ws", "vision_seqlens"]
    ):
        data["grid_ts"] = None
        data["grid_hs"] = None
        data["grid_ws"] = None
        data["vision_seqlens"] = None

    if data is not None:
        data_list = [data["grid_ts"], data["grid_hs"], data["grid_ws"], data["vision_seqlens"]]
    else:
        data_list = [None, None, None, None]
    if get_tensor_model_parallel_world_size() > 1:
        torch.distributed.broadcast_object_list(
            data_list,
            get_tensor_model_parallel_src_rank(),
            get_tensor_model_parallel_group(),
        )
    grid_ts, grid_hs, grid_ws, vision_seqlens = data_list

    # Unpack.
    tokens_ = tensor_parallel.broadcast_data(['text'], data, torch.int64)["text"]
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    if args.use_mrope:
        llm_pos_ids_ = get_llm_pos_ids(
            tokens_.flatten().tolist(),
            vision_seqlens,
            grid_ts,
            grid_hs,
            grid_ws,
            tokenizer.image_modality_token_id,
        )
        llm_pos_ids = llm_pos_ids_[:, :-1].contiguous().unsqueeze(1)
    else:
        llm_pos_ids = None

    if data is not None and 'image' not in data:
        data["image"] = None
    data_f = tensor_parallel.broadcast_data(["image", "loss_mask"], data, torch.float32)
    images = data_f["image"]
    loss_mask = data_f["loss_mask"][:, 1:].contiguous()

    # Get the image position masks.
    image_pos_masks = tokens == tokenizer.image_modality_token_id

    if args.dynamic_flash_attn:
        clone_tokens = tokens.clone()
        clone_tokens[:, -1] = tokenizer.eod
        eod_idx = torch.nonzero(clone_tokens.eq(tokenizer.eod).flatten(), as_tuple=True)[0] + 1
        cu_seqlens = F.pad(eod_idx, (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
    else:
        cu_seqlens = None
    if grid_ts is None or grid_hs is None or grid_ws is None:
        grid_thw = None
    else:
        grid_thw = torch.stack((torch.tensor(grid_ts), torch.tensor(grid_hs), torch.tensor(grid_ws)), dim=1)
    # Get the sources of each sample
    # if args.no_log_each_dataset:
    #     source = None
    # else:
    #     if get_tensor_model_parallel_rank() == 0:
    #         source_list = [data['source']]
    #     else:
    #         source_list = [None]
    #     if get_tensor_model_parallel_world_size() > 1:
    #         source = torch.distributed.broadcast_object_list(
    #             source_list,
    #             get_tensor_model_parallel_src_rank(),
    #             get_tensor_model_parallel_group(),
    #         )
    #     source = source_list[0]

    return (
        tokens,
        labels,
        loss_mask,
        cu_seqlens,
        image_pos_masks,
        images,
        grid_thw,
        llm_pos_ids,
    )


def loss_func(loss_mask, output_tensor):
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.float()
    ele_loss = losses * loss_mask
    sample_loss_info = torch.stack([ele_loss.sum(1), loss_mask.sum(1)], axis=1).tolist()
    loss = ele_loss.sum() / loss_mask.sum()

    # valid_seq_length = loss_mask.sum() / losses.shape[0]
    # seq_length = torch.as_tensor(losses.shape[1], dtype=loss.dtype, device=loss.device)

    # Reduce loss for logging.
    if args.reduce_loss_per_iteration:
        # Put off loss allreduce for logging.
        averaged_loss = [loss.clone().detach().view(1)]
        # averaged_loss = [loss.clone().detach().view(1), valid_seq_length.float(), seq_length.float()]
    else:
        # Reduce loss for logging per gradient accumulation step.
        averaged_loss = average_losses_across_data_parallel_group([loss])
        # averaged_loss = average_losses_across_data_parallel_group([loss, valid_seq_length, seq_length])

    info = {
        'lm_loss': averaged_loss[0],
        # 'valid_seq_length': averaged_loss[1],
        # 'seq_length': averaged_loss[2],
    }

    # if source is not None:
    #     temp_source2loss = defaultdict(float)
    #     for ss, l in zip(source, sample_loss_info):
    #         if isinstance(ss, str):
    #             ss = [ss]
    #         for k in ss:
    #             temp_source2loss['loss_' + k] += l[0] / l[1]
    #     all_source2loss = [None for _ in range(get_data_parallel_world_size())]
    #     torch.distributed.all_gather_object(
    #         all_source2loss, temp_source2loss, get_data_parallel_group()
    #     )
    #     source2loss = defaultdict(float)
    #     temp_source2loss = defaultdict(list)
    #     for s in all_source2loss:
    #         for k, v in s.items():
    #             temp_source2loss[k].append(v)
    #     for k, v in temp_source2loss.items():
    #         source2loss[k] = sum(v) / len(v)
    #     info.update(source2loss)
    return loss, info


def forward_step(data_iterator, model: Qwen2_5_VLModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (QwenVL): The Qwen2VL Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        (
            tokens,
            labels,
            loss_mask,
            cu_seqlens,
            image_pos_masks,
            images,
            grid_thw,
            llm_pos_ids,
        ) = get_batch(data_iterator)
    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(
            tokens,
            images=images,
            image_input_mask=image_pos_masks,
            position_ids=llm_pos_ids,
            attention_mask=None,
            grid_thw=grid_thw,
            labels=labels,
        )

    return output_tensor, partial(loss_func, loss_mask)

def core_vl_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return VLSFTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path)
        ],
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer
    )

def is_dataset_built_on_rank():
    # dynamic-flash-attn requires build packed_seq_params on each pp rank
    # TODO: find a better way
    return mpu.get_tensor_model_parallel_rank() == 0


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    config = core_vl_dataset_config_from_args(args)
    dataset_type = QwenVLSFTDataset
    print_rank_0('> building train, validation, and test datasets ' 'for GPT ...')

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    if train_ds:
        print_rank_0(train_ds[0])

    print_rank_0("> finished creating GPT datasets ...")
    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'QWenVLTokenizer'},
    )
