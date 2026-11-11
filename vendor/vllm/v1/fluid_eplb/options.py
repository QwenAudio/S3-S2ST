import dataclasses
import os


def get_bool_env(key: str, default: bool = False) -> bool:
    if not os.getenv(key):
        return default
    return os.getenv(key, "0").lower() in ["1", True, "true", 1, "on"]


def get_int_env(key: str, default: int = 0) -> int:
    try:
        return int(os.getenv(key, default))
    except ValueError:
        return default


def get_float_env(key: str, default: float = 0.0) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_str_env(key: str, default: str = "") -> str:
    if not os.getenv(key):
        return default
    return os.getenv(key, default)


@dataclasses.dataclass
class EPLBOptions:
    """
    EPLB options for configuring the expert parallel load balancing system.
    """

    log_dir: str = get_str_env("EPLB_LOG_DIR")
    expert_cache_dir: str = get_str_env(
        "EPLB_EXPERT_CACHE_DIR", default="/dev/shm/fluid-expert-cache"
    )
    redundants: int = get_int_env("EPLB_REDUNDANTS", default=0)
    rebalance_threshold: float = get_float_env(
        "EPLB_REBALANCE_THRESHOLD", default=1.5
    )
    dump_batch_size: int = get_int_env("EPLB_DUMP_BATCH_SIZE", default=100)
    enable_expert_cnt_dump: bool = get_bool_env(
        "ENABLE_EXPERT_CNT_DUMP", default=False
    )
    enable_expert_logging: bool = get_bool_env(
        "ENABLE_EXPERT_LOGGING", default=False
    )
    rebalance_interval: int = get_int_env(
        "EPLB_REBALANCE_INTERVAL", default=200
    )
    debug_flag: bool = get_bool_env("EPLB_DEBUG_FLAG", default=False)
    enable_online_eplb: bool = get_bool_env(
        "ENABLE_ONLINE_EPLB", default=False
    )
    bs_threshold: int = get_int_env("EPLB_BS_THRESHOLD", default=32)
    debug_rebalance_epgroups_fn: str = get_str_env(
        "EPLB_DEBUG_REBALANCE_EPGROUPS_FN", default=""
    )
    enable_mows: bool = get_bool_env("EPLB_ENABLE_MOWS", default=False)

    def __init__(self):
        if self.enable_online_eplb:
            self.enable_expert_logging = True

    def get_expert_replicas(self, num_logical_experts: int) -> int:
        """
        Get the number of expert replicas based on the number of logical experts.
        """
        return num_logical_experts + self.redundants

    def enable(self):
        return self.enable_online_eplb

    def get_expert_cache_dir(self):
        if self.expert_cache_dir == "":
            return self.log_dir
        return self.expert_cache_dir


eplb_options = EPLBOptions()
