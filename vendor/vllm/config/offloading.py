import hashlib
from typing import Any

from pydantic.dataclasses import dataclass

from .utils import config


@config
@dataclass
class CpuOffloadingConfig:
    offloading_method: str = "v1"
    """
    v1 config, how many GBs of RAM to reserve for CPU offloading.
    """
    cpu_offload_gb: int = 0
    """
    offloading config for CPUOffloaderV2
    """
    offload_group_size: int = 0
    """
    Number of layers per group in offloading.
    """
    offload_num_in_group: int = 0
    """Number of layers to be offloaded within a group."""
    offload_prefetch_step: int = 0
    """Steps to prefetch in offloading."""
    
    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        factors.append(self.offloading_method)
        factors.append(self.cpu_offload_gb)
        factors.append(self.offload_group_size)
        factors.append(self.offload_num_in_group)
        factors.append(self.offload_prefetch_step)
        return hashlib.sha256(str(factors).encode()).hexdigest()
    
    def __post_init__(self):
        if self.offloading_method == "v1":
            assert self.cpu_offload_gb >= 0, (
                "cpu_offload_gb must be greater than or equal to 0 for "
                "cpu offloading v1 method."
            )
        elif self.offloading_method == "v2":
            assert (
                self.cpu_offload_gb == 0
            ), "V2 offload does not support cpu_offload_gb yet"
            assert self.offload_group_size > 0, (
                    "V2 offload needs a valid offload_group_size, "
                    f"found {self.offload_group_size}"
            )
        else:
            raise ValueError(
                "Unsupported cpu offloading method %s.", self.offloading_method
            )

    @classmethod
    def from_dict(cls, dict_value: dict) -> "CpuOffloadingConfig":
        """Parse the CLI value for the config."""
        return cls(**dict_value)