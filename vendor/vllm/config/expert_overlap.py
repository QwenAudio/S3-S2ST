import hashlib
from typing import Any

from pydantic.dataclasses import dataclass

from .utils import config


@config
@dataclass
class ExpertCommOverlapConfig:
    overlap: bool = False
    """Whether to use overlap for moe and combine communication."""
    reduce_hidden: int = 4096
    """Combine send a portion of hidden once the gemm computes the
    results for these blocks.
    """
    combine_sm_num: int = 6
    """Number of SM to use for combine send kernel."""
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
        factors.append(self.overlap)
        factors.append(self.reduce_hidden)
        factors.append(self.combine_sm_num)
        return hashlib.sha256(str(factors).encode()).hexdigest()
    @classmethod
    def from_dict(cls, dict_value: dict) -> "ExpertCommOverlapConfig":
        """Parse the CLI value for the config."""
        return cls(**dict_value)