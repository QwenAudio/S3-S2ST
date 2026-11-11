import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

# TODO(erke): Add fluid to dependency and remove condition
_FLUID_EPLB_AVAILABLE = True
try:
    from fluid import ep_groups, expert_logging, fluid_eplb
    from fluid.fluid_eplb import FluidMoELayer
    from fluid.options import eplb_options
    from fluid.utils import get_epgroups_fn
except ImportError:
    logger.error("Fluid import failed, EPLB features will be disabled")
    _FLUID_EPLB_AVAILABLE = False

def enable(vllm_config: VllmConfig):
    expert_rebalance_interval = (
        vllm_config.scheduler_config.moe_expert_rebalance_interval
    )
    return _FLUID_EPLB_AVAILABLE and expert_rebalance_interval > 0

def enable_eplb_remapping(vllm_config: VllmConfig):
    # Enable remapping only when needed to avoid unnecessary loading overhead.
    return _FLUID_EPLB_AVAILABLE and (
        envs.VLLM_MOE_EXPERT_REMAPPING or enable(vllm_config)
    )

def get_instance():
    return fluid_eplb.get_instance()


def init_once(n_experts, ep_rank, ep_size, vllm_config):
    if fluid_eplb.get_instance() is not None:
        return fluid_eplb.get_instance()  # Already initialized

    from vllm.model_executor.layers.fused_moe.layer import REMAP_WEIGHTS, FusedMoE

    layers = vllm_config.compilation_config.static_forward_context
    target_layers = []
    num_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
    for _, layer in layers.items():
        if isinstance(layer, FusedMoE) and layer.layer_index < num_layers:
            parameters = {}
            for param in REMAP_WEIGHTS:
                if hasattr(layer, param):
                    parameters[param] = getattr(layer, param)
            target_layers.append(
                FluidMoELayer(
                    parameters,
                    layer.expert_mapping_func,
                    layer.ep_groups,
                    layer.local_experts,
                )
            )
    fluid_eplb.init_once(target_layers, ep_rank, ep_size, n_experts)
    logger.info(
        "fluid EPLB init success, layers %d, n_experts %d",
        len(target_layers),
        n_experts,
    )
    return fluid_eplb.get_instance()
