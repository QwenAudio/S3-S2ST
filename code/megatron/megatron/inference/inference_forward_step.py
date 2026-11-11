# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Forward step utilities."""

from collections.abc import Iterable

from megatron.inference.text_generation.forward_step import _allocate_recv_buffer
from megatron.training import get_args
from megatron.core import mpu, parallel_state

from megatron.inference.text_generation.communication import (
    send_to_next_pipeline_rank,
    recv_from_prev_pipeline_rank_)


class InferenceForwardStep:
    """Forward step function with all the communications.
    We use a class here to hide the inference parameters
    from the outside caller."""

    def __init__(self, model):
        """Set values so we don't need to do it multiple times."""
        # Make sure model is in eval mode.
        assert not isinstance(model, Iterable), \
            'interleaving schedule is not supported for inference'
        model.eval()
        self.model = model

    def __call__(self, tokens, position_ids, attention_mask, labels):
        """Invocation of the forward methods. Note that self.inference_params
        is being modified by the forward step."""
        return _no_pipelining_forward_step(self.model,
                                           tokens,
                                           position_ids,
                                           attention_mask,
                                           labels)
def _forward_step_helper(model, tokens, position_ids, attention_mask, labels, recv_buffer=None):
    """Single forward step. Update the allocate memory flag so
    only the first time the memory is allocated."""
    batch_size = tokens.size(0)
    sequence_length = tokens.size(1)
    args = get_args()
    if args.sequence_parallel:
        sequence_length = sequence_length // parallel_state.get_tensor_model_parallel_world_size()
    if recv_buffer is None:
        recv_buffer = _allocate_recv_buffer(batch_size, sequence_length)

    # Receive from previous stage.
    recv_from_prev_pipeline_rank_(recv_buffer)

    # Forward pass through the model.
    model.set_input_tensor(recv_buffer)
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    # Send output to the next stage.
    send_to_next_pipeline_rank(output_tensor)

    return output_tensor

def _no_pipelining_forward_step(model, tokens, position_ids, attention_mask, labels, recv_buffer=None):
    """If recv_buffer is none, we will allocate one on the fly."""
    # Run a simple forward pass.
    output_tensor = _forward_step_helper(model, tokens, position_ids,
                                         attention_mask, labels,
                                         recv_buffer=recv_buffer)

    logits = None
    if mpu.is_pipeline_last_stage():
        logits = output_tensor

    return logits
