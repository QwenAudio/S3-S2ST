# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import torch

from megatron.core.tensor_parallel import (
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
)

class SharedParam(torch.nn.Parameter):
    """
    A class to represent shared parameters between TP rank in the transformer layer.
    The weights need 'sequence_parallel' attribute for gradient all reduce.
    """

    def __new__(cls, config=None, shape=None, dtype=torch.float32, requires_grad=True,
                init_method=None, force_fp32=False):
        instance = super(SharedParam, cls).__new__(cls, torch.empty(shape, dtype=dtype), requires_grad)
        if config.perform_initialization and init_method:
            with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
                init_method(instance.data)
        instance.force_fp32 = force_fp32
        if config.sequence_parallel:
            setattr(instance, 'sequence_parallel', config.sequence_parallel)
        return instance

    def bfloat16(self):
        if self.force_fp32:
            return self
        else:
            return super().bfloat16()