import json
import os
from typing import Any

from loguru import logger


class EPGroups:
    """
    A class to handle expert groups from a JSON or JSON file.
    The JSON should contain a key 'ep_groups' which maps to a 2D list of expert groups.
    Each expert group is a list of integers representing expert IDs.
    Example JSON structure:
    {
        "ep_groups": [
            [0, 1, 2],
            [3, 4, 5],
            ...
        ],
    }
    """

    def __init__(self, ep_groups, fname=None):
        # TODO(yancey.yx): using this class in other modules, .e.g. ffn.py
        self.ratios: list[Any] | None = None
        self.fname = fname
        if isinstance(ep_groups, list):
            self.data: list[Any] = ep_groups
        elif isinstance(ep_groups, dict):
            assert "ep_groups" in ep_groups, "ep_groups key not found in dict"
            self.data = ep_groups.get("ep_groups", [])
            self.ratios = ep_groups.get("ratio", None)
        elif isinstance(ep_groups, str):
            try:
                logger.info("ep_groups: {}", ep_groups)
                self.data = json.loads(ep_groups)
            except Exception:
                logger.error(f"Failed to load ep_groups from string: {ep_groups}")
        else:
            raise ValueError(
                f"Invalid ep_groups, should be a JSON object "
                f"or a path to a JSON file: {ep_groups}"
            )

    @staticmethod
    def from_file(fn):
        """
        Create an EPGroups instance from a file.
        :param fn: Path to the JSON file containing expert groups.
        :return: EPGroups instance.
        """
        if not os.path.exists(fn):
            raise FileNotFoundError(f"EPGroups file {fn} does not exist")
        with open(fn) as f:
            json_data = json.loads(f.read())
            return EPGroups(json_data, fn)

    def __getitem__(self, item):
        return self.data[item]

    def get_num_layers(self):
        return len(self.data)

    def get_ranks(self):
        return len(self.data[0])

    def get_json(self):
        return self.data

    def get_num_local_experts(self):
        return len(self.data[0][0]) if self.get_num_layers() > 0 else 0

    def num_logical_experts(self):
        return len(set([element for sublist in self.data[0] for element in sublist]))

    def get_num_replicas(self):
        return len([item for sublist in self.data for item in sublist])

    def get_imbalance_ratio(self, layer_id: int):
        if self.ratios is None:
            raise ValueError("ratios not available")
        return self.ratios[layer_id]

    def find_expert_location(self, layer_id: int, expert_id: int):
        for rank_id, rank_experts in enumerate(self.data[layer_id]):
            if expert_id in rank_experts:
                return rank_id, rank_experts.index(expert_id)
        raise RuntimeError(f"expert{expert_id} not found")

    def get_fname(self):
        return self.fname
