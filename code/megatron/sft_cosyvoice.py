# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""SFT GPT."""

import os
import torch
from functools import partial
from contextlib import nullcontext
import inspect
from logging import getLogger
logger = getLogger(__name__)

import omegaconf
from funasr.register import tables
from megatron.core.models.multimodal.cosyvoice_compat import CosyVoiceMixedLMSimpleSysV2  # noqa: F401

from typing import Union, Dict
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.sft_dataset import SFTDatasetConfig, SFTDataset
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.core.num_microbatches_calculator import (
    get_num_microbatches
)



stimer = StragglerDetector()

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
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

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else: # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, args.multi_latent_attention, args.fp8, args.use_chunkflow)
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
            model = GPTModel(
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


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    # need to get packed_seq_params on each pp rank?
    # if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
    #     return None, None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch


def loss_func(loss_mask: torch.Tensor, kwargs: Dict, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        kwargs (Dict): Additional arguments for loss calculation
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()

    if args.use_chunkflow or args.use_smart_batching:
        assert kwargs is not None, "kwargs is required for chunkflow"
        losses = losses.view(-1)
        loss_docs = losses * loss_mask
        cu_seqlens = kwargs['cu_seqlens']
        chunkflow_total_tokens = kwargs['total_tokens']
        num_documents = cu_seqlens.shape[0] - 1
        loss = torch.tensor(0.0, device=losses.device, dtype=losses.dtype)
        for i in range(0, num_documents):
            doc_start, doc_end = cu_seqlens[i], cu_seqlens[i + 1]
            valid_token_count = loss_mask[doc_start:doc_end].sum() if num_documents != 1 else chunkflow_total_tokens[0]
            loss += torch.sum(loss_docs[doc_start:doc_end]) / valid_token_count
        # Scale loss by get_num_microbatches() * micro_batch_size
        loss = loss / (get_num_microbatches() * args.micro_batch_size)
    else:
        if args.context_parallel_size > 1:
            loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
            torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
            loss = loss[0] / loss[1]
        else:
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        # loss = loss # do not scale for non-chunkflow case
    # Reduce loss for logging.
    if args.reduce_loss_per_iteration:
        # Put off loss allreduce for logging.
        averaged_loss = [loss.clone().detach().view(1)]
    else:
        averaged_loss = average_losses_across_data_parallel_group([loss])
    loss_dict = {'lm loss': averaged_loss[0]}

    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    return (
        loss * args.context_parallel_size,
        loss_dict
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        batch = get_batch(data_iterator)
        tokens = batch['tokens']
        labels = batch['labels']
        loss_mask = batch['loss_mask']
        attention_mask = batch['attention_mask']
        position_ids = batch['position_ids']
        if args.dynamic_flash_attn or args.use_chunkflow or args.use_smart_batching:
            # TODO: Current cu_seqs will create lots of sequences (len=1) for padding tokens.
            #  Fix it or do not use --dynamic_flash_attn in sft.
            packed_seq_params = batch['packed_seq_params']
        else:
            packed_seq_params = None
    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels, packed_seq_params=packed_seq_params)

    # ChunkFlow requires additonal information for correct loss calculation.
    return output_tensor, partial(loss_func, loss_mask, batch.get("kwargs", None))


def is_dataset_built_on_rank():
    # dynamic-flash-attn requires build packed_seq_params on each pp rank
    # TODO: find a better way
    return mpu.get_tensor_model_parallel_rank() == 0


def convert_batch_to_megatron(input_data, megatron_padding_id, megatron_tokenizer_codec_offset, cosyvoice_padding_id=-1, cosyvoice_ignore_id=-100):
    # input: dict_keys(['input_ids', 'labels_ids', 'attention_mask', 'text_flag', 'codec_flag', 'codec', 'codec_len'])
    # output: dict_keys(['tokens', 'labels', 'loss_mask', 'attention_mask', 'position_ids', 'packed_seq_params'])

    input_data['input_ids'] = input_data['input_ids'][:, :-1]
    input_data['attention_mask'] = input_data['attention_mask'][:, :-1]
    input_data['labels_ids'] = input_data['labels_ids'][:, 1:]

    position_ids = torch.arange(input_data['attention_mask'].shape[1], dtype=torch.int64).unsqueeze(0).expand(input_data['attention_mask'].shape[0], -1)

    # FIXME: hard-coded
    input_mask = input_data['input_ids'] >= 0
    input_data['labels_ids'][:, :] = 0  
    labels_mask = input_data['labels_ids'] >= 0
    # MEGATRON_PADDING_VALUE = megatron_padding_id
    # MEGATRON_FILLER_VALUE = 0
    # MEGATRON_FILLER_VALUE = MEGATRON_PADDING_VALUE

    # input_data['input_ids'][input_data['input_ids'] == cosyvoice_padding_id] = megatron_padding_id
    # input_data['input_ids'][input_data['input_ids'] == cosyvoice_ignore_id] = MEGATRON_FILLER_VALUE
    # input_data['labels_ids'][input_data['labels_ids'] == cosyvoice_padding_id] = megatron_padding_id
    # input_data['labels_ids'][input_data['labels_ids'] == cosyvoice_ignore_id] = MEGATRON_FILLER_VALUE

    # # offset token ids by megatron_tokenizer_codec_offset
    # input_data['input_ids'][input_data['codec_flag'] == 1] += megatron_tokenizer_codec_offset
    # input_data['labels_ids'][input_data['codec_flag'] == 1] += megatron_tokenizer_codec_offset

    input_data['input_ids'][~input_mask] = megatron_padding_id
    input_data['labels_ids'][~labels_mask] = megatron_padding_id

    loss_mask = torch.ones_like(input_data['attention_mask'], dtype=torch.float32)
    loss_mask[~labels_mask] = 0.0

    output_data = {
        'tokens': input_data['input_ids'],
        'labels': input_data['labels_ids'],
        'loss_mask': loss_mask,
        # 'attention_mask': None,
        'position_ids': position_ids
    }
    return output_data


def get_megatron_tokenizer_info():
    print(" ===== get_megatron_tokenizer_info =====, pid: ", os.getpid())
    megatron_tokenizer = get_tokenizer()
    # megatron_padding_id = megatron_tokenizer.eod
    megatron_padding_id = 151645 # qwen use 151645
    megatron_tokenizer_codec_offset = megatron_tokenizer._tokenizer('<|c0|>')['input_ids'][0]
    del megatron_tokenizer
    # megatron_tokenizer_codec_offset = 151692
    return megatron_padding_id, megatron_tokenizer_codec_offset



class TrainIterator:
    def __init__(self, dataloader, dataset_conf, train_conf, **kwargs):
        self.kwargs = kwargs
        self.dataloader = dataloader
        self.dataset_conf = dataset_conf
        self.train_conf = train_conf
        self.epoch = 0
        self.total_steps = 0
        self.step_in_epoch = 0
        self.step_in_split = 0
        self.max_epoch = train_conf.get('max_epoch', 1)
        self.data_split_i = 0
        self.dataloader_tr = None
        self.dataloader_tr_iterator = None
        self.dataloader_val = None
        self.data_split_num = self.dataloader.data_split_num
        self.data_split_size = self.train_conf.get('data_split_size', -1)
        if self.data_split_size > 0:
            self.data_split_num = 999999

    def load_state(self, state_dict):
        logger.info(f"loading dataloader state: {state_dict}")
        self.epoch = state_dict['epoch']
        self.data_split_i = state_dict['data_split_i']
        self.step_in_split = state_dict['step_in_split']
        self.step_in_epoch = state_dict['step_in_epoch']
        self.total_steps = state_dict['total_steps']
        self.data_split_num = state_dict['data_split_num']
        self.data_split_size = state_dict['data_split_size']

    def save_state(self):
        state_dict = {
            'epoch': self.epoch,
            'data_split_i': self.data_split_i,
            # 'dataloader_tr_iterator': self.dataloader_tr_iterator,
            'step_in_split': self.step_in_split,
            'step_in_epoch': self.step_in_epoch,
            'total_steps': self.total_steps,
            'dataloader_iterator_num_yielded': self.dataloader_tr_iterator._num_yielded if self.dataloader_tr_iterator is not None else -1,
            'data_split_num': self.data_split_num,
            'data_split_size': self.data_split_size,
            'max_epoch': self.max_epoch,
        }
        logger.info(f"saving dataloader state: {state_dict}")
        return state_dict

    def __next__(self):
        while True:
            if self.dataloader_tr is None:
                self.dataloader_tr, _ = self.dataloader.build_iter(self.epoch, data_split_i=self.data_split_i, start_step=self.step_in_split)
                self.dataloader_tr_iterator = iter(self.dataloader_tr)
            try:
                data = next(self.dataloader_tr_iterator)
                break
            except StopIteration:
                self.dataloader_tr_iterator = None
                self.dataloader_tr = None
                self.data_split_i += 1
                self.step_in_split = 0
                if self.data_split_i >= self.data_split_num:
                    self.epoch += 1
                    self.data_split_i = 0
                    self.step_in_epoch = 0
                if self.epoch >= self.max_epoch:
                    raise StopIteration
        self.total_steps += 1
        self.step_in_split += 1
        self.step_in_epoch += 1
        return convert_batch_to_megatron(data, **self.kwargs)

    def __iter__(self):
        return self


def train_iterator(dataloader, **kwargs):
    for epoch in range(0, 10):
        for data_split_i in range(0, dataloader.data_split_num):
            dataloader_tr, _ = dataloader.build_iter(
                epoch, data_split_i=data_split_i, start_step=0
            )
            for data in dataloader_tr:
                data = convert_batch_to_megatron(data, **kwargs)
                yield data


def val_iterator(dataloader, **kwargs):
    for data in dataloader:
        data = convert_batch_to_megatron(data, **kwargs)
        yield data


def build_cosyvoice_data_iterators(train_val_test_num_samples):
    args = get_args()
    config_path = args.cosyvoice_dataset_config_path
    print_rank_0(f"Loading cosyvoice dataset config from {config_path}")
    kwargs = omegaconf.OmegaConf.load(config_path)
    kwargs = dict(kwargs)

    # build tokenizer
    print_rank_0("Build tokenizer")
    tokenizer = kwargs.get("tokenizer", None)
    if tokenizer is not None:
        tokenizer_class = tables.tokenizer_classes.get(tokenizer)
        tokenizer = tokenizer_class(**kwargs.get("tokenizer_conf", {}))
        kwargs["token_list"] = (
            tokenizer.token_list if hasattr(tokenizer, "token_list") else None
        )
        kwargs["token_list"] = (
            tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else kwargs["token_list"]
        )
        vocab_size = len(kwargs["token_list"]) if kwargs["token_list"] is not None else -1
        if vocab_size == -1 and hasattr(tokenizer, "get_vocab_size"):
            vocab_size = tokenizer.get_vocab_size()
    else:
        vocab_size = -1
    kwargs["tokenizer"] = tokenizer

    # build frontend
    frontend = kwargs.get("frontend", None)
    kwargs["input_size"] = None
    if frontend is not None:
        frontend_class = tables.frontend_classes.get(frontend)
        frontend = frontend_class(**kwargs.get("frontend_conf", {}))
        kwargs["input_size"] = (
            frontend.output_size() if hasattr(frontend, "output_size") else None
        )
    kwargs["frontend"] = frontend

    # dataset
    print_rank_0("Build dataloader")
    dataloader_class = tables.dataloader_classes.get(
        kwargs["dataset_conf"].get("dataloader", "DataloaderMapStyle")
    )
    dataloader = dataloader_class(**kwargs)
    _, dataloader_val = dataloader.build_iter(
            0, data_split_i=0, start_step=0
        )

    megatron_padding_id, megatron_tokenizer_codec_offset = get_megatron_tokenizer_info()
    fill_token = kwargs['dataset_conf'].get('fill_token')
    pad_id = kwargs['dataset_conf'].get('int_pad_value', -1)
    ignore_id = kwargs['dataset_conf'].get('ignore_id', -100)
    specaug_fill_value = kwargs['dataset_conf'].get('specaug_conf', {}).get('fill_value', -100)

    train_iterator = TrainIterator(dataloader,
                                   megatron_padding_id=megatron_padding_id,
                                   megatron_tokenizer_codec_offset=megatron_tokenizer_codec_offset,
                                   dataset_conf=kwargs["dataset_conf"],
                                   train_conf=kwargs["train_conf"])

    return train_iterator, None, None


if __name__ == "__main__":
    # Temporary for transition to core datasets
    build_cosyvoice_data_iterators.is_distributed = True

    pretrain(
        build_cosyvoice_data_iterators,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        use_train_valid_test_dataset_provider_as_data_iterators=True
    )
