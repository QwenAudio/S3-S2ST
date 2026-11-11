# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import contextlib
import dataclasses
from typing import Iterator, List, Union

import torch
from torch.autograd.variable import Variable

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.utils import (
    drain_embedding_wgrad_compute,
    get_attr_wrapped_model,
    get_model_config,
    get_model_type,
    get_model_xattn,
    get_model_extra_output_specs,
)
from megatron.core.chunkflow.chunkflow_state_store import state_store_manager

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_src_rank,
)

from megatron.core.transformer.moe.flashmoe_router import (
    set_lbl_model_chunk_id,
    set_lbl_num_microbatches,
    get_lbl_num_microbatches,
    post_finish_lbl_loss_calculate
)

# Types
Shape = Union[List[int], torch.Size]


def get_forward_backward_func():
    """Retrieves the appropriate forward_backward function given the
    configuration of parallel_state.

    Returns a function that will perform all of the forward and
    backward passes of the model given the pipeline model parallel
    world size and virtual pipeline model parallel world size in the
    global parallel_state.

    Note that if using sequence parallelism, the sequence length component of
    the tensor shape is updated to original_sequence_length /
    tensor_model_parallel_world_size.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one torch.Tensor and return a torch.Tensor of loss and a
        dictionary of string -> torch.Tensor.

        A third argument, checkpoint_activations_microbatch, indicates
        that the activations for this microbatch should be
        checkpointed. A None value for this argument indicates that
        the default from the configuration should be used. This is
        used when the
        num_microbatches_with_partial_activation_checkpoints is used.

        For example:

        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        def forward_step(data_iterator, model):
            data, loss_mask = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, loss_mask)


        forward_backward_func(forward_step_func=forward_step, ...)


    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func. Expected to be a list of
        iterators in the case of interleaved pipeline parallelism.

    model (required): the actual model. Expected to be a list of modules in the case of interleaved
        pipeline parallelism. Must be a (potentially wrapped) megatron.core.models.MegatronModule.

    num_microbatches (int, required):
        The number of microbatches to go through

    seq_length (int, required): Sequence length of the current global batch. If this is a dual-stack
        transformer, this is the encoder's sequence length. This is ignored if variable_seq_lengths
        in the config is True. Otherwise, each microbatch in the current global batch size must use
        this sequence length.

    micro_batch_size (int, required): The number of sequences in a microbatch.

    decoder_seq_length (int, optional): The sequence length for the decoder in a dual-stack
        transformer. This is ignored for a single-stack transformer.

    forward_only (optional, default = False): Perform only the forward step

    collect_non_loss_data (optional, bool, default=False): TODO

    first_val_step (bool, optional): Is the first step of the validation phase. Used by
        Transformer Engine modules to only update their fp8 weights only on the first validation
        step.

    """
    from megatron.training import get_args
    args = get_args()
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    if pipeline_model_parallel_size > 1:
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            assert not (args.use_chunkflow or args.use_smart_batching), "ChunkFlow smart batching is not supported in virtual pipeline parallelism for now"
            forward_backward_func = forward_backward_pipelining_with_interleaving
        else:
            use_dynamic_collate = (args.use_dynamic_collate or args.use_smart_batching) and (not args.dynamic_padding_full)
            use_chunkflow = args.use_chunkflow
            if use_chunkflow:
                forward_backward_func = forward_backward_pipelining_without_interleaving_with_chunkflow
            elif use_dynamic_collate:
                from functools import partial
                forward_backward_func = partial(forward_backward_pipelining_without_interleaving, use_dynamic_collate=True)
            else:
                forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        if args.use_chunkflow:
            forward_backward_func = forward_backward_no_pipelining_with_chunkflow
        else:
            forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func


def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, "counter-productive to free a view of another tensor."
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype)


def custom_backward(output, grad_output, lbl_loss=None):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    assert output.numel() == 1, "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), (
        "grad_output == '%s'." % type(grad_output).__name__
    )

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(output, memory_format=torch.preserve_format)
    grad_lbl_loss = None
    if lbl_loss is not None:
        assert lbl_loss.numel() == 1, "implicit grad requires scalar output."
        grad_lbl_loss = torch.ones_like(
            lbl_loss,
            memory_format=torch.preserve_format,
        )
        tensors = (output, lbl_loss)
        grad_tensors = (grad_output, grad_lbl_loss)
    else:
        tensors = (output,)
        grad_tensors = (grad_output,)

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors=tensors,
        grad_tensors=grad_tensors,
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def set_current_microbatch(model, microbatch_id):
    decoder_exists = True
    decoder = None
    try:
        decoder = get_attr_wrapped_model(model, "decoder")
    except RuntimeError:
        decoder_exists = False
    if decoder_exists and decoder is not None:
        decoder.current_microbatch = microbatch_id


def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
    encoder_decoder_xattn=False,
):
    """Forward step for passed-in model.

    If it is the first stage, the input tensor is obtained from the data_iterator.
    Otherwise, the passed-in input_tensor is used.

    Args:
        forward_step_func (callable):
            The forward step function for the model that takes the
            data iterator as the first argument, and model as the second.
            This user's forward step is expected to output a tuple of two elements:

                1. The output object from the forward step. This output object needs to be a
                    tensor or some kind of collection of tensors. The only hard requirement
                    for this object is that it needs to be acceptible as input into the second
                    function.
                2. A function to reduce (optionally) the output from the forward step. This
                    could be a reduction over the loss from the model, it could be a function that
                    grabs the output from the model and reformats, it could be a function that just
                    passes through the model output. This function must have one of the following
                    patterns, and depending on the pattern different things happen internally:

                        a. A tuple of reduced loss and some other data. Note that in this case
                            the first argument is divided by the number of global microbatches,
                            assuming it is a loss, so that the loss is stable as a function of
                            the number of devices the step is split across.
                        b. A triple of reduced loss, number of tokens, and some other data. This
                            is similar to case (a), but the loss is further averaged across the
                            number of tokens in the batch. If the user is not already averaging
                            across the number of tokens, this pattern is useful to use.
                        c. Any arbitrary data the user wants (eg a dictionary of tensors, a list
                            of tensors, etc in the case of inference). To trigger case 3 you need
                            to specify `collect_non_loss_data=True` and you may also want to
                            specify `forward_only=True` in the call to the parent forward_backward
                            function.
        data_iterator (iterator):
            The data iterator.
        model (nn.Module):
            The model to perform the forward step on.
        num_microbatches (int):
            The number of microbatches.
        input_tensor (Tensor or list[Tensor]):
            The input tensor(s) for the forward step.
        forward_data_store (list):
            The list to store the forward data. If you go down path 2.a or
            2.b for the return of your forward reduction function then this will store only the
            final dimension of the output, for example the metadata output by the loss function.
            If you go down the path of 2.c then this will store the entire output of the forward
            reduction function applied to the model output.
        config (object):
            The configuration object.
        collect_non_loss_data (bool, optional):
            Whether to collect non-loss data. Defaults to False.
            This is the path to use if you want to collect arbitrary output from the model forward,
            such as with inference use cases. Defaults to False.
        checkpoint_activations_microbatch (int, optional):
            The microbatch to checkpoint activations.
            Defaults to None.
        is_first_microbatch (bool, optional):
            Whether it is the first microbatch. Defaults to False.
        current_microbatch (int, optional):
            The current microbatch. Defaults to None.

    Returns:
        Tensor or list[Tensor]: The output object(s) from the forward step.
        Tensor: The number of tokens.
    """
    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    if is_first_microbatch and hasattr(model, 'set_is_first_microbatch'):
        model.set_is_first_microbatch()
    if current_microbatch is not None:
        set_current_microbatch(model, current_microbatch)

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        if checkpoint_activations_microbatch is None:
            output_tensor, loss_func = forward_step_func(data_iterator, model)
        else:
            output_tensor, loss_func = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch
            )
    
    if isinstance(output_tensor, tuple):
        output_tensor, extra_outputs = output_tensor
    else:
        extra_outputs = []

    num_tokens = torch.tensor(0, dtype=torch.int)
    lbl_loss = None
    if parallel_state.is_pipeline_last_stage():
        if not collect_non_loss_data:
            outputs = loss_func(output_tensor, *extra_outputs)
            packed_batch = config.use_chunkflow or config.use_smart_batching
            if len(outputs) == 3:
                output_tensor, num_tokens, loss_reduced = outputs
                # We dont do loss scaling when ChunkFlow or smartbatching is enabled
                if not config.calculate_per_token_loss and not packed_batch:
                    output_tensor /= num_tokens
                    output_tensor /= num_microbatches
            else:
                # preserve legacy loss averaging behavior (ie, over the number of microbatches)
                assert len(outputs) == 2
                output_tensor, loss_reduced = outputs
                # We dont do loss scaling when ChunkFlow or smartbatching is enabled
                if not packed_batch:
                    output_tensor /= num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, *extra_outputs, non_loss_data=True)
            forward_data_store.append(data)
    elif config.use_flashmoe_router:
        lbl_loss, num_tokens, lbl_loss_reduced = loss_func(None, *extra_outputs)
        if lbl_loss is not None:
            lbl_loss = lbl_loss / num_microbatches
            forward_data_store.append(lbl_loss_reduced)

    if config.timers is not None:
        config.timers('forward-compute').stop()

    # Set the loss scale for the auxiliary loss of the MoE layer.
    # Since we use a trick to do backward on the auxiliary loss, we need to set the scale
    # explicitly.
    if hasattr(config, 'num_moe_experts') and config.num_moe_experts is not None and not config.use_flashmoe_router:
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.tensor(1.0)
        )
        # Set the loss scale
        MoEAuxLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    # If T5 model and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    model_type = get_model_type(model)
    if (
        model_type == ModelType.encoder_and_decoder
        and encoder_decoder_xattn
        and parallel_state.is_inside_decoder()
    ):
        return [output_tensor, input_tensor[-1]], num_tokens, extra_outputs

    if lbl_loss is not None:
      return [output_tensor, lbl_loss], num_tokens, extra_outputs
    if lbl_loss is None and not parallel_state.is_pipeline_last_stage():
      return [output_tensor, None], num_tokens, extra_outputs

    if unwrap_output_tensor:
        return output_tensor, num_tokens, extra_outputs
    return [output_tensor], num_tokens, extra_outputs


def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    if config.timers is not None:
        config.timers('backward-compute', log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None and x.requires_grad:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    lbl_loss = 0
    # args = get_args()
    # num_layers_per_pipeline_stage, _ = \
    #     parallel_state.get_pipeline_num_layers_per_stage_and_offset(args)
    num_layers_per_pipeline_stage = config.num_layers_per_pipeline_stage
    if config.moe_lbl_all_reduce_overlap and num_layers_per_pipeline_stage > 0:
        lbl_loss = post_finish_lbl_loss_calculate()
        lbl_loss = lbl_loss / get_lbl_num_microbatches()
    if output_tensor_grad[0] is None and config.grad_scale_func is not None:
        output_tensor[0] = config.grad_scale_func(output_tensor[0])
    if config.deallocate_pipeline_outputs:
        if parallel_state.is_pipeline_last_stage():
            custom_backward(output_tensor[0]+lbl_loss, output_tensor_grad[0])
        elif config.use_flashmoe_router:
            if num_layers_per_pipeline_stage > 0:
                custom_backward(output_tensor[0], output_tensor_grad[0], lbl_loss=output_tensor[1]+lbl_loss)
            else:
                custom_backward(output_tensor[0], output_tensor_grad[0], lbl_loss=output_tensor[1])
        else:
            custom_backward(output_tensor[0], output_tensor_grad[0])
    else:
        torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if (
        parallel_state.get_pipeline_model_parallel_world_size() > 1
        and model_type == ModelType.encoder_and_decoder
        and len(output_tensor_grad) > 1  # excludes models that lack a skip connection.
    ):
        if output_tensor_grad[1] is not None:
            assert input_tensor_grad[-1] is not None
            input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if config.timers is not None:
        config.timers('backward-compute').stop()

    return input_tensor_grad


def check_first_val_step(first_val_step, forward_only, cond):
    if (first_val_step is not None) and forward_only:
        return first_val_step and cond
    else:
        return cond


def forward_backward_no_pipelining(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,  # unused
    micro_batch_size: int,  # unused
    decoder_seq_length: int = None,  # unused
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses.


    See get_forward_backward_func() for argument details
    """

    if isinstance(model, list):
        assert len(model) == 1, "non-pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    model_type = get_model_type(model)

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    if config.use_smart_batching:
        num_microbatches_tensor = torch.zeros(1, device=torch.cuda.current_device(), dtype=torch.int64)
        if data_iterator is not None:
            data_iterator = PrefetchBatches(data_iterator, num_microbatches,config,micro_batch_size)
            num_microbatches_tensor[0] = len(data_iterator)
        # Broadcast num_micro_batches information when smart_batching is enabled
        torch.distributed.broadcast(num_microbatches_tensor, get_tensor_model_parallel_src_rank(),
                                group=get_tensor_model_parallel_group())
        num_microbatches = num_microbatches_tensor[0].item()

    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    set_lbl_num_microbatches(num_microbatches)

    with no_sync_func():
        for i in range(num_microbatches - 1):
            output_tensor, num_tokens, _ = forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
                current_microbatch=i,
            )
            if num_tokens is not None:
              total_num_tokens += num_tokens.item()
            if not forward_only:
                backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor, num_tokens, _ = forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data,
        is_first_microbatch=check_first_val_step(
            first_val_step, forward_only, num_microbatches == 1
        ),
        current_microbatch=num_microbatches - 1,
    )
    if num_tokens is not None:
      total_num_tokens += num_tokens.item()

    if not forward_only:
        backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism and layernorm all-reduce for sequence parallelism).
        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    return forward_data_store


class PrefetchBatches:
    def __init__(self, data_iterator, num_microbatches: int, config, micro_batch_size):
        assert config.use_chunkflow or config.use_smart_batching , "PrefetchBatches should be only used when chunkflow  or smartbatching is enabled"
        self.prefetch_batches = []
        expected_total_samples = num_microbatches * micro_batch_size
        chunk_left = 0
        use_chunkflow = config.use_chunkflow
        while expected_total_samples > 0 or (use_chunkflow and chunk_left > 0):
            batch = next(data_iterator)
            self.prefetch_batches.append(batch)
            expected_total_samples -= torch.eq(batch["position_ids"], 0).sum().item()
            if use_chunkflow:
                seq_length = batch['seq_length'].item()
                chunk_idx = batch['position_ids'][0][-1].item() // config.chunk_size
                chunk_count = (seq_length +  config.chunk_size - 1) // config.chunk_size
                if chunk_idx == 0 and chunk_left == 0:
                    if chunk_count == 1:
                        chunk_left = 1
                    else:
                        if chunk_count > config.chunks_kept:
                            expected_total_samples = expected_total_samples + 1
                            chunk_left = (2 * chunk_count - config.chunks_kept)
                        else:
                            chunk_left = chunk_count
                chunk_left -= 1
        if expected_total_samples != 0:
            print("Expected to prefetch {} batches, but got {}".format(num_microbatches, num_microbatches - expected_total_samples))
            raise Exception("Expected to prefetch {} batches, but got {}".format(num_microbatches, num_microbatches - expected_total_samples))
        self.index = 0

    def __next__(self):
        if self.index < len(self.prefetch_batches):
            data = self.prefetch_batches[self.index]
            self.index += 1
            return data
        else:
            raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.prefetch_batches)

    def current_size(self):
        if self.index < len(self.prefetch_batches):
            return self.get_size(self.index)
        return None

    def get_size(self, index: int):
        return self.prefetch_batches[index]['text'].size()[1]

    def __getitem__(self, index):
        return self.prefetch_batches[index]



def forward_backward_no_pipelining_with_chunkflow(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,  # unused
    micro_batch_size: int,  # unused
    decoder_seq_length: int = None,  # unused
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses.


    See get_forward_backward_func() for argument details
    """

    if isinstance(model, list):
        assert len(model) == 1, "non-pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    model_type = get_model_type(model)

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    grad_compute_queue = []
    # Just for logging
    grad_compute_chunk_idx = []
    batch_idx_info = {}

    num_microbatches_tensor = torch.zeros(1, device=torch.cuda.current_device(), dtype=torch.int64)
    if data_iterator is not None:
        data_iterator = PrefetchBatches(data_iterator, num_microbatches,config,micro_batch_size)
        num_microbatches_tensor[0] = len(data_iterator)
    # Broadcast num_micro_batches information when chunkflow is enabled
    torch.distributed.broadcast(num_microbatches_tensor, get_tensor_model_parallel_src_rank(),
                            group=get_tensor_model_parallel_group())
    num_microbatches = num_microbatches_tensor[0].item()

    set_lbl_num_microbatches(num_microbatches)

    # ChunkFlow: By default, chunk_idx is 0 and chunk_count is 1
    chunk_idx_tensor = torch.zeros(num_microbatches, device=torch.cuda.current_device(), dtype=torch.int64)
    chunk_count_tensor = torch.ones(num_microbatches, device=torch.cuda.current_device(), dtype=torch.int64)

    if data_iterator is not None:
        for i in range(len(data_iterator)):
            # Calculate chunk_idx and chunk_count
            batch = data_iterator[i]
            seq_length = batch['seq_length'].item()
            chunk_idx_val = batch['position_ids'][0][-1].item() // config.chunk_size
            chunk_count_val = (seq_length +  config.chunk_size - 1) // config.chunk_size
            chunk_idx_tensor[i] = chunk_idx_val
            chunk_count_tensor[i] = chunk_count_val

    torch.distributed.broadcast(chunk_count_tensor, get_tensor_model_parallel_src_rank(),
                            group=get_tensor_model_parallel_group())
    torch.distributed.broadcast(chunk_idx_tensor, get_tensor_model_parallel_src_rank(),
                                group=get_tensor_model_parallel_group())
    chunk_idx_list = chunk_idx_tensor.tolist()
    chunk_count_list = chunk_count_tensor.tolist()

    def is_the_sequence_tail_chunk(chunk_idx, chunk_count):
        # sequence_tail_chunk is the chunk containing the last token of the sequence
        return chunk_idx == chunk_count - 1

    # Real batch information, since multiple sequences could belong to one batch
    batch_idx = 0
    with no_sync_func():
        for i in range(num_microbatches - 1):
             # Record correct batch index information
            batch_idx_info[i] = batch_idx

            output_tensor, num_tokens = forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
                current_microbatch=i,
            )

            # some batches are only secheduled to generate correct states.
            inner_forward_only = True
            # Execute forward-backward for chunks with idx in [N-K, N]
            # Execute forward-backward for the second-time forward passes for chunks with idx in [0, N-K]
            if chunk_idx_list[i] >= (chunk_count_list[i] -  config.chunks_kept) or \
                state_store_manager.get_state_store().get_reuse_state():
                inner_forward_only = forward_only

            # Start to reuse kv state after the last chunk, i.e. chunks[N-1]
            if is_the_sequence_tail_chunk(chunk_idx_list[i],chunk_count_list[i]):
                state_store_manager.get_state_store().set_reuse_state()

            # Update batch_idx for logging
            if chunk_count_list[i] <= config.chunks_kept and \
                is_the_sequence_tail_chunk(chunk_idx_list[i],chunk_count_list[i]):
                batch_idx += 1
            elif chunk_count_list[i] > config.chunks_kept and \
                 state_store_manager.get_state_store().get_reuse_state() and \
                chunk_idx_list[i] == 0:
                batch_idx += 1
            # For those forward-only-schedules, we dont count there loss information into total loss
            if inner_forward_only:
                forward_data_store.pop()
            else:
                total_num_tokens += num_tokens.item()
            if not inner_forward_only:
                grad_compute_queue.append(output_tensor)
                grad_compute_chunk_idx.append(i)
                # We add losses for backwards for:
                # 1. The second-time forward passes for chunks with idx in [0, N-K]
                # 2. The first-time forward passes and the last chunk is executed
                if state_store_manager.get_state_store().get_reuse_state() \
                or is_the_sequence_tail_chunk(chunk_idx_list[i],chunk_count_list[i]):
                    for chunk_output_tensor, chunk_idx in zip(reversed(grad_compute_queue), reversed(grad_compute_chunk_idx)):
                        backward_step(input_tensor, chunk_output_tensor, output_tensor_grad, model_type, config)
                    grad_compute_queue.clear()
                    grad_compute_chunk_idx.clear()

        # Run computation for last microbatch out of context handler (want to
        # synchronize gradients).
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            is_first_microbatch=check_first_val_step(
                first_val_step, forward_only, num_microbatches == 1
            ),
            current_microbatch=num_microbatches - 1,
        )
        batch_idx_info[num_microbatches - 1] = batch_idx
        total_num_tokens += num_tokens.item()

        # Run backward passes for remaining chunks
        if not forward_only:
            grad_compute_queue.append(output_tensor)
            grad_compute_chunk_idx.append(num_microbatches - 1)

            for idx, chunk_output_tensor, chunk_idx in zip(range(len(grad_compute_queue) - 1), reversed(grad_compute_queue[1:]), reversed(grad_compute_chunk_idx[1:])):
                backward_step(input_tensor, chunk_output_tensor, output_tensor_grad, model_type, config)

    # Backward for the last chunk here
    if not forward_only:
        chunk_output_tensor = grad_compute_queue[0]
        chunk_idx = grad_compute_chunk_idx[0]
        backward_step(input_tensor, chunk_output_tensor, output_tensor_grad, model_type, config)
    grad_compute_queue.clear()
    grad_compute_chunk_idx.clear()

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism and layernorm all-reduce for sequence parallelism).
        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    return forward_data_store


def clear_embedding_activation_buffer(config, model):

    if (
        parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        and config.defer_embedding_wgrad_compute
    ):
        if isinstance(model, list):
            embedding_module = get_attr_wrapped_model(
                model[-1], 'post_process', return_model_obj=True
            )
        else:
            embedding_module = get_attr_wrapped_model(model, 'post_process', return_model_obj=True)

        # Need to ensure no stray activations exists in this buffer
        embedding_module.embedding_activation_buffer.clear()

        return embedding_module
    else:
        return None


def finish_embedding_wgrad_compute(config, embedding_module):
    if (
        parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        and config.defer_embedding_wgrad_compute
    ):
        embedding_activation_buffer = embedding_module.embedding_activation_buffer
        grad_output_buffer = embedding_module.grad_output_buffer
        weight = (
            embedding_module.output_layer.weight
            if embedding_module.share_embeddings_and_output_weights
            else embedding_module.shared_embedding_or_output_weight()
        )

        drain_embedding_wgrad_compute(
            config, embedding_activation_buffer, grad_output_buffer, weight
        )


def forward_backward_pipelining_with_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    assert isinstance(model, list), "interleaved pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid model chunking"
    assert isinstance(
        data_iterator, list
    ), "interleaved pipeline parallelism expected each model chunk to have a data iterator"

    config = get_model_config(model[0])
    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model)

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if isinstance(no_sync_func, list):

        def multi_no_sync():
            stack = contextlib.ExitStack()
            for model_chunk_no_sync_func in config.no_sync_func:
                stack.enter_context(model_chunk_no_sync_func())
            return stack

        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    if config.grad_sync_func is not None and not isinstance(config.grad_sync_func, list):
        config.grad_sync_func = [config.grad_sync_func for _ in model]

    if config.param_sync_func is not None and not isinstance(config.param_sync_func, list):
        config.param_sync_func = [config.param_sync_func for _ in model]

    # Disable config.grad_sync_func and config.param_sync_func if only running forward passes.
    # They will be re-enabled at the end of this function.
    grad_sync_func, param_sync_func = None, None
    if forward_only:
        grad_sync_func, param_sync_func = config.grad_sync_func, config.param_sync_func
        config.grad_sync_func, config.param_sync_func = None, None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    forward_data_store = []
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

    if num_microbatches % pipeline_parallel_size != 0:
        msg = f'number of microbatches ({num_microbatches}) is not divisible by '
        msg += f'pipeline-model-parallel-size ({pipeline_parallel_size}) '
        msg += 'when using interleaved schedule'
        raise RuntimeError(msg)

    model_type = get_model_type(model[0])
    if model_type == ModelType.encoder_and_decoder:
        raise RuntimeError("Interleaving is not supported with an encoder and decoder model.")

    if decoder_seq_length is not None and decoder_seq_length != seq_length:
        raise RuntimeError(
            "Interleaving is not supported with a different decoder sequence length."
        )

    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
    tensor_shape[0] = tensor_shape[0] // parallel_state.get_context_parallel_world_size()
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    total_num_microbatches = num_microbatches * num_model_chunks
    all_warmup_microbatches = False
    if forward_only:
        num_warmup_microbatches = total_num_microbatches
    else:
        # Run all forward passes and then all backward passes if number of
        # microbatches is just the number of pipeline stages.
        # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 1F1B).
        if num_microbatches == pipeline_parallel_size:
            num_warmup_microbatches = total_num_microbatches
            all_warmup_microbatches = True
        else:
            num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            num_warmup_microbatches += (num_model_chunks - 1) * pipeline_parallel_size
            num_warmup_microbatches = min(num_warmup_microbatches, total_num_microbatches)
    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    # Synchronize params for first two model chunks
    if config.param_sync_func is not None:
        config.param_sync_func[0](model[0].parameters())
        config.param_sync_func[1](model[1].parameters())

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (pipeline_parallel_size * num_model_chunks)
        model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def get_microbatch_id_in_model_chunk(iteration_id, forward):
        """Helper method to get the microbatch_id within model chunk given the iteration number."""
        assert forward
        iteration_group_id = iteration_id // (pipeline_parallel_size * num_model_chunks)
        microbatch_id_in_model_chunk = (iteration_group_id * pipeline_parallel_size) + (
            iteration_id % pipeline_parallel_size
        )
        return microbatch_id_in_model_chunk

    def is_first_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == 0:
            return microbatch_id_in_group % pipeline_parallel_size == 0
        else:
            return False

    def is_last_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == num_microbatch_groups - 1:
            return microbatch_id_in_group % pipeline_parallel_size == pipeline_parallel_size - 1
        else:
            return False

    def forward_step_helper(microbatch_id, current_microbatch, checkpoint_activations_microbatch):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        set_lbl_model_chunk_id(model_chunk_id)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch param synchronization for next model chunk
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.param_sync_func is not None:
            param_sync_microbatch_id = microbatch_id + pipeline_parallel_rank
            if (
                param_sync_microbatch_id < total_num_microbatches
                and is_first_microbatch_for_model_chunk(param_sync_microbatch_id)
            ):
                param_sync_chunk_id = get_model_chunk_id(param_sync_microbatch_id, forward=True) + 1
                if 1 < param_sync_chunk_id < num_model_chunks:
                    config.param_sync_func[param_sync_chunk_id](
                        model[param_sync_chunk_id].parameters()
                    )

        # forward step
        if parallel_state.is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step, forward_only, is_first_microbatch_for_model_chunk(microbatch_id)
            ),
            current_microbatch=current_microbatch,
        )
        output_tensors[model_chunk_id].append(output_tensor)

        nonlocal total_num_tokens
        if num_tokens is not None:
          total_num_tokens += num_tokens.item()

        # if forward-only, no need to save tensors for a backward pass
        if forward_only:
            input_tensors[model_chunk_id].pop()
            output_tensors[model_chunk_id].pop()

        return output_tensor

    def backward_step_helper(microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        set_lbl_model_chunk_id(model_chunk_id)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch grad synchronization (default)
        if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(microbatch_id):
            enable_grad_sync()
            synchronized_model_chunks.add(model_chunk_id)

        if parallel_state.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )

        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.grad_sync_func is not None:
            grad_sync_microbatch_id = microbatch_id - pipeline_parallel_rank
            if grad_sync_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(
                grad_sync_microbatch_id
            ):
                grad_sync_chunk_id = get_model_chunk_id(grad_sync_microbatch_id, forward=False)
                enable_grad_sync()
                config.grad_sync_func[grad_sync_chunk_id](model[grad_sync_chunk_id].parameters())
                synchronized_model_chunks.add(grad_sync_chunk_id)
        disable_grad_sync()

        return input_tensor_grad

    set_lbl_num_microbatches(num_microbatches)
    # Run warmup forward passes.
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    input_tensors[0].append(p2p_communication.recv_forward(tensor_shape, config))

    fwd_wait_handles = None
    bwd_wait_handles = None

    for k in range(num_warmup_microbatches):

        if fwd_wait_handles is not None:
            for req in fwd_wait_handles:
                req.wait()

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        current_microbatch = get_microbatch_id_in_model_chunk(k, forward=True)
        output_tensor = forward_step_helper(
            k, current_microbatch, checkpoint_activations_microbatch
        )
        if isinstance(output_tensor, list):
          output_tensor = output_tensor[0]

        # Determine if tensor should be received from previous stage.
        next_forward_model_chunk_id = get_model_chunk_id(k + 1, forward=True)
        recv_prev = True
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            if next_forward_model_chunk_id == 0:
                recv_prev = False
        if k == (total_num_microbatches - 1):
            recv_prev = False

        # Don't send tensor downstream if on last stage.
        if parallel_state.is_pipeline_last_stage():
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if not config.overlap_p2p_comm:
            if (
                k == (num_warmup_microbatches - 1)
                and not forward_only
                and not all_warmup_microbatches
            ):
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False
                (input_tensor, output_tensor_grad) = (
                    p2p_communication.send_forward_backward_recv_forward_backward(
                        output_tensor,
                        input_tensor_grad,
                        recv_prev=recv_prev,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        config=config,
                    )
                )
                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            else:
                input_tensor = p2p_communication.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape, config=config
                )
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        else:
            input_tensor, fwd_wait_handles = p2p_communication.send_forward_recv_forward(
                output_tensor,
                recv_prev=recv_prev,
                tensor_shape=tensor_shape,
                config=config,
                overlap_p2p_comm=True,
            )

            if (
                k == (num_warmup_microbatches - 1)
                and not forward_only
                and not all_warmup_microbatches
            ):
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False

                (output_tensor_grad, bwd_wait_handles) = (
                    p2p_communication.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        config=config,
                        overlap_p2p_comm=True,
                    )
                )

                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            input_tensors[next_forward_model_chunk_id].append(input_tensor)

        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Run 1F1B in steady state.
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                forward_k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        current_microbatch = get_microbatch_id_in_model_chunk(forward_k, forward=True)
        if config.overlap_p2p_comm:
            if fwd_wait_handles is not None:
                for req in fwd_wait_handles:
                    req.wait()

            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            output_tensor = forward_step_helper(
                forward_k, current_microbatch, checkpoint_activations_microbatch
            )
            if isinstance(output_tensor, list):
              output_tensor = output_tensor[0]

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)

            # Last virtual stage no activation tensor to send
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            # Determine if peers are sending, and where in data structure to put
            # received tensors.
            recv_prev = True
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k - (pipeline_parallel_size - 1), forward=True
                )
                if next_forward_model_chunk_id == (num_model_chunks - 1):
                    recv_prev = False
                next_forward_model_chunk_id += 1
            else:
                next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1, forward=True)

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Send activation tensor to the next stage and receive activation tensor from the
            # previous stage
            input_tensor, fwd_wait_handles = p2p_communication.send_forward_recv_forward(
                output_tensor,
                recv_prev=recv_prev,
                tensor_shape=tensor_shape,
                config=config,
                overlap_p2p_comm=True,
            )
            # assert fwd_wait_handles is not None

            if bwd_wait_handles is not None:
                for req in bwd_wait_handles:
                    req.wait()

            # Backward pass.
            backward_k = k
            input_tensor_grad = backward_step_helper(backward_k)

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)

            # First virtual stage no activation gradient tensor to send
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            # Determine if the current virtual stage has an activation gradient tensor to receive
            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k - (pipeline_parallel_size - 1), forward=False
                )
                if next_backward_model_chunk_id == 0:
                    recv_next = False
                next_backward_model_chunk_id -= 1
            else:
                next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1, forward=False)

            output_tensor_grad, bwd_wait_handles = p2p_communication.send_backward_recv_backward(
                input_tensor_grad,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                config=config,
                overlap_p2p_comm=True,
            )

        else:  # no p2p overlap
            output_tensor = forward_step_helper(
                forward_k, current_microbatch, checkpoint_activations_microbatch
            )
            if isinstance(output_tensor, list):
              output_tensor = output_tensor[0]

            # Backward pass.
            backward_k = k
            input_tensor_grad = backward_step_helper(backward_k)

            # Send output_tensor and input_tensor_grad, receive input_tensor
            # and output_tensor_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            # Determine if peers are sending, and where in data structure to put
            # received tensors.
            recv_prev = True
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k - (pipeline_parallel_size - 1), forward=True
                )
                if next_forward_model_chunk_id == (num_model_chunks - 1):
                    recv_prev = False
                next_forward_model_chunk_id += 1
            else:
                next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1, forward=True)

            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k - (pipeline_parallel_size - 1), forward=False
                )
                if next_backward_model_chunk_id == 0:
                    recv_next = False
                next_backward_model_chunk_id -= 1
            else:
                next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1, forward=False)

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Communicate tensors.
            (input_tensor, output_tensor_grad) = (
                p2p_communication.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    config=config,
                )
            )
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

        # Put input_tensor and output_tensor_grad in data structures in the
        # right location.
        if recv_prev:
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        if recv_next:
            output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Run cooldown backward passes (flush out pipeline).
    if not forward_only:
        if config.overlap_p2p_comm and bwd_wait_handles is not None:
            for wait_handle in bwd_wait_handles:
                wait_handle.wait()

        if all_warmup_microbatches:
            output_tensor_grads[num_model_chunks - 1].append(
                p2p_communication.recv_backward(tensor_shape, config=config)
            )
        for k in range(num_microbatches_remaining, total_num_microbatches):
            input_tensor_grad = backward_step_helper(k)
            next_backward_model_chunk_id = get_model_chunk_id(k + 1, forward=False)
            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                if next_backward_model_chunk_id == (num_model_chunks - 1):
                    recv_next = False
            if k == (total_num_microbatches - 1):
                recv_next = False
            output_tensor_grads[next_backward_model_chunk_id].append(
                p2p_communication.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next, tensor_shape=tensor_shape, config=config
                )
            )

        # Launch any remaining grad reductions.
        enable_grad_sync()
        if config.grad_sync_func is not None:
            for model_chunk_id in range(num_model_chunks):
                if model_chunk_id not in synchronized_model_chunks:
                    config.grad_sync_func[model_chunk_id](model[model_chunk_id].parameters())
                    synchronized_model_chunks.add(model_chunk_id)

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(config, embedding_module)

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            model, total_num_tokens if config.calculate_per_token_loss else None
        )

    # Restore config.grad_sync_func and config.param_sync_func.
    if forward_only:
        config.grad_sync_func, config.param_sync_func = grad_sync_func, param_sync_func

    if config.timers is not None:
        config.timers('forward-backward').stop()

    return forward_data_store


def get_tensor_shapes(
    *,
    rank: int,
    model_type: ModelType,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int,
    config,
    encoder_decoder_xattn: bool,
    model_extra_output_specs: list[tuple[list[Union[int, str]], torch.dtype]],
):
    # Determine right tensor sizes (based on position of rank with
    # respect to split rank) and model size.
    # Send two tensors if model decoder requires the encoder's output
    # (via cross-attention) and rank is in decoder stage.
    #     first tensor is decoder.
    #     second tensor is encoder.
    # If model has an encoder & decoder and rank is at the boundary:
    #     send one tensor.
    # Otherwise, send one tensor.
    tensor_shapes = []

    seq_length = seq_length // parallel_state.get_context_parallel_world_size()
    if model_type == ModelType.encoder_and_decoder:
        decoder_seq_length = decoder_seq_length // parallel_state.get_context_parallel_world_size()

    if config.sequence_parallel:
        seq_length = seq_length // parallel_state.get_tensor_model_parallel_world_size()
        if model_type == ModelType.encoder_and_decoder:
            decoder_seq_length = (
                decoder_seq_length // parallel_state.get_tensor_model_parallel_world_size()
            )

    if model_type == ModelType.encoder_and_decoder:
        if parallel_state.is_inside_encoder(rank):
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
        elif encoder_decoder_xattn:
            tensor_shapes.append((decoder_seq_length, micro_batch_size, config.hidden_size))
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
        else:
            tensor_shapes.append((decoder_seq_length, micro_batch_size, config.hidden_size))
    else:  # model_type == ModelType.encoder_or_decoder
        tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
    
    extra_specs = []
    for spec in model_extra_output_specs:
        shape, dtype = spec
        if shape == p2p_communication.TensorSpec.DYNAMIC_SHAPE:
            parsed_shape = shape
        else:
            parsed_shape = []
            for dim in shape:
                if dim == "b":
                    parsed_shape.append(micro_batch_size)
                elif dim == "l":
                    parsed_shape.append(seq_length)
                else:
                    parsed_shape.append(dim)
            parsed_shape = tuple(parsed_shape)
        extra_specs.append(p2p_communication.TensorSpec(parsed_shape, dtype))

    return tensor_shapes, extra_specs


def recv_forward(tensor_shapes, extra_specs, config):
    input_tensors = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensors.append(p2p_communication.recv_forward(tensor_shape, config))
    for tensor_spec in extra_specs:
        if tensor_spec is None:
            input_tensors.append(None)
        else:
            input_tensors.append(p2p_communication.recv_forward(tensor_spec, config))
    return input_tensors


def recv_backward(tensor_shapes, config):
    output_tensor_grads = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grads.append(p2p_communication.recv_backward(tensor_shape, config))
    return output_tensor_grads


def send_forward(output_tensors, tensor_shapes, extra_outputs, extra_specs, config):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for output_tensor, tensor_shape in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_forward(output_tensor, config)
    for extra_tensor, tensor_spec in zip(extra_outputs, extra_specs):
        if tensor_spec is None:
            continue
        p2p_communication.send_forward(extra_tensor, config)


def send_backward(input_tensor_grads, tensor_shapes, config):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_backward(input_tensor_grad, config)


def send_forward_recv_backward(output_tensors, tensor_shapes, extra_outputs, extra_specs, config):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    output_tensor_grads = []
    for output_tensor, tensor_shape in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            output_tensor_grads.append(None)
            continue
        output_tensor_grad = p2p_communication.send_forward_recv_backward(
            output_tensor, tensor_shape, config
        )
        output_tensor_grads.append(output_tensor_grad)
    for extra_tensor, tensor_spec in zip(extra_outputs, extra_specs):
        if tensor_spec is None:
            continue
        p2p_communication.send_forward(extra_tensor, config)
    return output_tensor_grads


def send_backward_recv_forward(input_tensor_grads, tensor_shapes, extra_specs, config):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    input_tensors = []
    for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            input_tensors.append(None)
            continue
        input_tensor = p2p_communication.send_backward_recv_forward(
            input_tensor_grad, tensor_shape, config
        )
        input_tensors.append(input_tensor)
    for tensor_spec in extra_specs:
        if tensor_spec is None:
            input_tensors.append(None)
        else:
            input_tensors.append(p2p_communication.recv_forward(tensor_spec, config))
    return input_tensors


def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
    use_dynamic_collate: bool = False,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages. Returns dictionary with losses if the last stage, empty dict otherwise."""

    if isinstance(model, list):
        assert (
            len(model) == 1
        ), "non-interleaved pipeline parallelism does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)

    if use_dynamic_collate:
        config = dataclasses.replace(config, variable_seq_lengths=True)

    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model)

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()
    if config.use_smart_batching:
        num_microbatches_tensor = torch.zeros(1, device=torch.cuda.current_device(), dtype=torch.int64)
        # Dynamic collate: prefetch batch data to get dynamic seq_length.
        if data_iterator is not None:
            data_iterator = PrefetchBatches(data_iterator, num_microbatches,config,micro_batch_size)
            # Get number of microbatches, When smart_batching is disabled,
            # len(data_iterator) is always the same with user's settings.
            num_microbatches_tensor[0] = len(data_iterator)

            # Broadcast num_micro_batches information when smart_batching is enabled.
            torch.distributed.broadcast(num_microbatches_tensor, get_tensor_model_parallel_src_rank(),
                                        group=get_tensor_model_parallel_group())
            num_microbatches = num_microbatches_tensor[0]

        else:
            # Broadcast num_micro_batches information when smart_batching is enabled.
            torch.distributed.broadcast(num_microbatches_tensor, get_tensor_model_parallel_src_rank(),
                                        group=get_tensor_model_parallel_group())
            num_microbatches = num_microbatches_tensor[0]

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
        - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    model_type = get_model_type(model)
    encoder_decoder_xattn = get_model_xattn(model)
    model_extra_output_specs = get_model_extra_output_specs(model)

    rank = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes, recv_extra_specs = get_tensor_shapes(
        rank=rank - 1,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
        model_extra_output_specs=model_extra_output_specs,
    )
    send_tensor_shapes, send_extra_specs = get_tensor_shapes(
        rank=rank,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
        model_extra_output_specs=model_extra_output_specs,
    )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []

    set_lbl_num_microbatches(num_microbatches)
    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = recv_forward(recv_tensor_shapes, recv_extra_specs, config)
        output_tensor, num_tokens, extra_outputs = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )
        send_forward(output_tensor, send_tensor_shapes, extra_outputs, send_extra_specs, config)
        if num_tokens is not None:
          total_num_tokens += num_tokens.item()

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = recv_forward(recv_tensor_shapes, recv_extra_specs, config)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None

        output_tensor, num_tokens, extra_outputs = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=i + num_warmup_microbatches,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )
        if num_tokens is not None:
          total_num_tokens += num_tokens.item()

        if forward_only:
            send_forward(output_tensor, send_tensor_shapes, extra_outputs, send_extra_specs, config)

            if not last_iteration:
                input_tensor = recv_forward(recv_tensor_shapes, recv_extra_specs, config)

        else:
            output_tensor_grad = send_forward_recv_backward(
                output_tensor, send_tensor_shapes, extra_outputs, send_extra_specs, config
            )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            if last_iteration:
                input_tensor = None
                send_backward(input_tensor_grad, recv_tensor_shapes, config)
            else:
                input_tensor = send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, recv_extra_specs, config
                )

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):

            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = recv_backward(send_tensor_shapes, config)

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            send_backward(input_tensor_grad, recv_tensor_shapes, config)

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(config, embedding_module)

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    return forward_data_store


def forward_backward_pipelining_without_interleaving_with_chunkflow(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages. Returns dictionary with losses if the last stage, empty dict otherwise."""

    if isinstance(model, list):
        assert (
            len(model) == 1
        ), "non-interleaved pipeline parallelism does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if not config.dynamic_padding_full:
        config = dataclasses.replace(config, variable_seq_lengths=True)
    full_seq_length = seq_length

    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model)

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    ################################Chunkflow Start#############################################################
    # In ChunkFlow, we need to get the actual number of microbatches from the data iterator.
    num_microbatches_tensor = torch.zeros(1, device=torch.cuda.current_device(), dtype=torch.int64)
    chunk_idx_tensor = None
    chunk_count_tensor = None
    if data_iterator is not None:
        data_iterator = PrefetchBatches(data_iterator, num_microbatches,config,micro_batch_size)
        # Get number of microbatches, When chunkflow is disabled,
        # len(data_iterator) is always the same with user's settings.
        num_microbatches_tensor[0] = len(data_iterator)
        torch.distributed.broadcast(num_microbatches_tensor, get_tensor_model_parallel_src_rank(),
                                    group=get_tensor_model_parallel_group())
        num_microbatches = num_microbatches_tensor[0]
        # Construct chunk_idx_tensor and chunk_count_tensor. By default, chunk_idx_tensor is all zeros and chunk_count_tensor is all ones.
        chunk_idx_tensor = torch.zeros(num_microbatches, device=torch.cuda.current_device(), dtype=torch.int64)
        chunk_count_tensor = torch.ones(num_microbatches, device=torch.cuda.current_device(), dtype=torch.int64)
        # Calculate the chunk_idx_tensor and chunk_count_tensor.
        for i in range(len(data_iterator)):
            batch = data_iterator[i]
            seq_length = batch['seq_length'].item()
            chunk_idx_value = batch['position_ids'][0][-1].item() // config.chunk_size
            chunk_count_value = (seq_length +  config.chunk_size - 1) // config.chunk_size
            chunk_idx_tensor[i] = chunk_idx_value
            chunk_count_tensor[i] = chunk_count_value
        torch.distributed.broadcast(chunk_idx_tensor, get_tensor_model_parallel_src_rank(),
                                group=get_tensor_model_parallel_group())
        torch.distributed.broadcast(chunk_count_tensor, get_tensor_model_parallel_src_rank(),
                                group=get_tensor_model_parallel_group())

    else:
        # Broadcast num_micro_batches information when chunkflow is enabled.
        torch.distributed.broadcast(num_microbatches_tensor, get_tensor_model_parallel_src_rank(),
                                    group=get_tensor_model_parallel_group())
        num_microbatches = num_microbatches_tensor[0]

        chunk_idx_tensor = torch.zeros(num_microbatches, device=torch.cuda.current_device(), dtype=torch.int64)
        chunk_count_tensor = torch.ones(num_microbatches, device=torch.cuda.current_device(), dtype=torch.int64)

        torch.distributed.broadcast(chunk_idx_tensor, get_tensor_model_parallel_src_rank(),
                                group=get_tensor_model_parallel_group())
        torch.distributed.broadcast(chunk_count_tensor, get_tensor_model_parallel_src_rank(),
                                group=get_tensor_model_parallel_group())

    # Tensor to list for latter use
    chunk_idx_list = chunk_idx_tensor.tolist()
    chunk_count_list = chunk_count_tensor.tolist()
    world_size = parallel_state.get_pipeline_model_parallel_world_size()
    rank = parallel_state.get_pipeline_model_parallel_rank()
    ################################Chunkflow End#############################################################

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
        - 2
        + chunk_count_list[0]
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    model_type = get_model_type(model)
    encoder_decoder_xattn = get_model_xattn(model)
    recv_tensor_shapes = get_tensor_shapes(
        rank=rank - 1,
        model_type=model_type,
        seq_length=full_seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )
    send_tensor_shapes = get_tensor_shapes(
        rank=rank,
        model_type=model_type,
        seq_length=full_seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    batch_indices = None
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    if not forward_only:
        input_tensors = []
        output_tensors = []
        batch_indices = []
    forward_data_store = []

    # batch_idx for one long sequence
    batch_indices_for_single_sequence = []
    input_tensors_for_single_sequence = []
    output_tensors_for_single_sequence = []

    # Help to track the batch_idx for each chunk.
    batch_idx = 0
    batch_idx_info = {}


    inner_forward_only = forward_only
    def is_the_sequence_tail_chunk(chunk_idx, chunk_count):
        # sequence_tail_chunk is the chunk containing the last token of the sequence.
        return chunk_idx == chunk_count - 1

    # Run warmup forward passes.
    _batch_idx = -1
    for i in range(num_warmup_microbatches):
        # Decide to checkpoint all layers' activations of the current micro-batch
        _batch_idx += 1
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = recv_forward(recv_tensor_shapes, config)
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )

        batch_idx_info[_batch_idx] = batch_idx
        inner_forward_only = True
        # Start reuse state after the execution of the last chunk of the sequence.
        if is_the_sequence_tail_chunk(chunk_idx_list[_batch_idx], chunk_count_list[_batch_idx]):
                state_store_manager.get_state_store().set_reuse_state()

        # 1. During the first-time forward passes, we execute forward-backward for chunks with idx in [N-K, N]
        # 2. During the second-time forward passes, we execute forward-backward for chunks with idx in [0, N-K]
        if chunk_idx_list[_batch_idx] >= (chunk_count_list[_batch_idx] - config.chunks_kept) or \
                state_store_manager.get_state_store().get_reuse_state():
            inner_forward_only = forward_only

        # When the next chunk will be the first chunk of a new sequence, we need to update batch_idx.
        if chunk_count_list[_batch_idx] <= config.chunks_kept and \
            is_the_sequence_tail_chunk(chunk_idx_list[_batch_idx], chunk_count_list[_batch_idx]):
            batch_idx += 1
        elif chunk_count_list[_batch_idx] > config.chunks_kept and \
                state_store_manager.get_state_store().get_reuse_state() and \
            chunk_idx_list[_batch_idx] == 0:
            batch_idx += 1

        # For those forward-only schedules, we dont count there loss data.
        if parallel_state.is_pipeline_last_stage() and inner_forward_only:
            forward_data_store.pop()

        send_forward(output_tensor, send_tensor_shapes, config)
        # Only count num_tokens from those micro-batches who will execute both forwards and backwards
        if not inner_forward_only:
            total_num_tokens += num_tokens.item()

        if not inner_forward_only:
            input_tensors_for_single_sequence.append(input_tensor)
            output_tensors_for_single_sequence.append(output_tensor)
            batch_indices_for_single_sequence.append(_batch_idx)
            # 1. For the first-time forward passes, we start backward when the last chunk is executed
            # 2. For the second-time forward passes, we start backward for each of them
            if state_store_manager.get_state_store().get_reuse_state() \
                or is_the_sequence_tail_chunk(chunk_idx_list[_batch_idx], chunk_count_list[_batch_idx]):
                input_tensors.extend(reversed(input_tensors_for_single_sequence))
                output_tensors.extend(reversed(output_tensors_for_single_sequence))
                batch_indices.extend(reversed(batch_indices_for_single_sequence))

                # Reset
                input_tensors_for_single_sequence = []
                output_tensors_for_single_sequence = []
                batch_indices_for_single_sequence = []
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = recv_forward(recv_tensor_shapes, config)
    # Run 1F1B in steady state.
    _batch_idx = num_warmup_microbatches - 1
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)
        _batch_idx += 1
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None
        batch_idx_info[_batch_idx] = batch_idx
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=i + num_warmup_microbatches,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )

        inner_forward_only = True

        # Start reuse state after the execution of the last chunk of the sequence.
        if is_the_sequence_tail_chunk(chunk_idx_list[_batch_idx], chunk_count_list[_batch_idx]):
                state_store_manager.get_state_store().set_reuse_state()

        # 1. During the first-time forward passes, we execute forward-backward for chunks with idx in [N-K, N]
        # 2. During the second-time forward passes, we execute forward-backward for chunks with idx in [0, N-K]
        if chunk_idx_list[_batch_idx] >= (chunk_count_list[_batch_idx] - config.chunks_kept) or \
                state_store_manager.get_state_store().get_reuse_state():
            inner_forward_only = forward_only

        # When the next chunk will be the first chunk of a new sequence, we need to update batch_idx.
        if chunk_count_list[_batch_idx] <= config.chunks_kept and \
            is_the_sequence_tail_chunk(chunk_idx_list[_batch_idx], chunk_count_list[_batch_idx]):
            batch_idx += 1
        elif chunk_count_list[_batch_idx] > config.chunks_kept and \
                state_store_manager.get_state_store().get_reuse_state() and \
            chunk_idx_list[_batch_idx] == 0:
            batch_idx += 1

        # For those forward-only schedules, we dont count there loss data.
        if parallel_state.is_pipeline_last_stage() and inner_forward_only:
            forward_data_store.pop()

        if not inner_forward_only:
            total_num_tokens += num_tokens.item()

        if forward_only:
            send_forward(output_tensor, send_tensor_shapes, config)

            if not last_iteration:
                input_tensor = recv_forward(recv_tensor_shapes, config)
        else:
            if not inner_forward_only:
                input_tensors_for_single_sequence.append(input_tensor)
                output_tensors_for_single_sequence.append(output_tensor)
                batch_indices_for_single_sequence.append(_batch_idx)
                # 1. For the first-time forward passes, we start backward when the sequence tail chunk is executed
                # 2. For the second-time forward passes, we start backward for each of them, i.e. state_store_manager.get_state_store().get_reuse_state() = True
                if state_store_manager.get_state_store().get_reuse_state() \
                or is_the_sequence_tail_chunk(chunk_idx_list[_batch_idx], chunk_count_list[_batch_idx]):
                    input_tensors.extend(reversed(input_tensors_for_single_sequence))
                    output_tensors.extend(reversed(output_tensors_for_single_sequence))
                    batch_indices.extend(reversed(batch_indices_for_single_sequence))

                    # Reset storage for single sequences
                    input_tensors_for_single_sequence = []
                    output_tensors_for_single_sequence = []
                    batch_indices_for_single_sequence = []

            if not batch_indices or _batch_idx - batch_indices[0] < (world_size - rank - 1):
                send_forward(output_tensor, send_tensor_shapes, config)
                if not last_iteration:
                    input_tensor = recv_forward(recv_tensor_shapes, config)
                deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)
                continue

            output_tensor_grad = send_forward_recv_backward(
                output_tensor, send_tensor_shapes, config
            )
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            grad_chunk_idx = batch_indices.pop(0)

            # TODO: We will enable this feature in next version
            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            #if num_warmup_microbatches == 0 and last_iteration:
            #    if config.grad_sync_func is None or rank == 0:
            #        enable_grad_sync()

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )
            if last_iteration:
                input_tensor = None
                send_backward(input_tensor_grad, recv_tensor_shapes, config)
            else:
                input_tensor = send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, config
                )

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(len(batch_indices)):

            # TODO: We will enable this feature in next version
            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            #if i == len(batch_indices) - 1:
            #    if config.grad_sync_func is None or rank == 0:
            #        enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)
            grad_chunk_idx = batch_indices.pop(0)


            output_tensor_grad = recv_backward(send_tensor_shapes, config)

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )
            send_backward(input_tensor_grad, recv_tensor_shapes, config)

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(config, embedding_module)

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    return forward_data_store
