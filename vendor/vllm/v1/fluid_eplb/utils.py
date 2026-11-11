import json
import os

import torch

from vllm.logger import init_logger

from .options import eplb_options

logger = init_logger(__name__)


def imbalance_ratio_per_layer(expert_cnts, nodes):
    group_cnts = expert_cnts.view(expert_cnts.shape[0], nodes, -1).sum(dim=2)
    max = torch.max(group_cnts, dim=1).values
    mean = torch.mean(group_cnts.to(dtype=torch.float), dim=1)
    return torch.div(max, mean)

def imbalance_ratio_per_layer_per_rank(expert_cnts, nodes):
    rank_cnts = expert_cnts.view(expert_cnts.shape[0], nodes, -1)
    rank_max = torch.max(rank_cnts, dim=2).values
    rank_mean = torch.mean(rank_cnts.float(), dim=2)
    return torch.div(rank_max, rank_mean)

def remove_zeros_row(tensor):
    mask = (tensor != 0).any(dim=1)
    return tensor[mask]


def load_and_check_imbalance_ratio(expert_cnts, latest_step, ep_ranks):
    expert_cnts = remove_zeros_row(expert_cnts)
    if expert_cnts.shape[0] == 0:
        logger.info("check step{}, invalid expert cnts, skip rebalance", latest_step)
        return False, None
    layers_ratio = imbalance_ratio_per_layer(expert_cnts, ep_ranks)
    layers_rank_ratio = imbalance_ratio_per_layer_per_rank(expert_cnts, ep_ranks)
    model_ratio = float(layers_ratio.mean())
    model_rank_ratio = float(layers_rank_ratio.mean())
    global_bs = expert_cnts[0].sum() // eplb_options.dump_batch_size // 8 // ep_ranks
    need_rebalance = (
        model_ratio > eplb_options.rebalance_threshold
        and global_bs > eplb_options.bs_threshold
    )
    logger.info(
            f"[EPLB] step={latest_step}, "
            f"imbalance_ratio={model_ratio:.4f}"
            f"(thres={eplb_options.rebalance_threshold}), "
            f"intra_rank_imbalance_ratio={model_rank_ratio:.4f}, "
            f"global_bs={global_bs.item()}(thres={eplb_options.bs_threshold}), "
            f"need_rebalance={need_rebalance}."
    )
    return need_rebalance, model_ratio, model_rank_ratio


def generate_eplb_json(expert_cnts, num_replicas, ratio, ranks):
    from fluid import eplb  # lazy import

    num_nodes = ranks // 8
    num_gpus = ranks
    # TODO(yancey.yx): num_groups is a hypothetical value, need to be calculated
    # at generating EPLB configuration file
    num_groups = num_nodes * 2

    expert_cnts = remove_zeros_row(expert_cnts)
    phy2log, _, _ = eplb.rebalance_experts(
        expert_cnts, num_replicas, num_groups, num_nodes, num_gpus
    )
    ep_groups = {
        "ep_groups": phy2log.view(phy2log.shape[0], num_gpus, -1).tolist(),
        "ratio": ratio.tolist(),
    }
    return ep_groups


def check_and_gen_eplb(origin_expert_cnts, expert_cnts, step, rank, ranks):
    _, num_experts = expert_cnts.shape
    need_rebalance, \
    ratio, \
    rank_ratio = load_and_check_imbalance_ratio(
        expert_cnts, step, ranks
    )
    if need_rebalance:
        fn = get_epgroups_fn(step)
        ep_groups = generate_eplb_json(origin_expert_cnts, num_experts, ratio, ranks)
        if rank % 8 == 0:
            with open(fn, "w") as f:
                json.dump(ep_groups, f)
            logger.info(f"rank{rank}| generated EPLB file: {fn}")  # noqa: G004
        return True, ratio, rank_ratio
    return False, ratio, rank_ratio


def get_epgroups_fn(step):
    return os.path.join(eplb_options.log_dir, f"eplb-groups-steps{step}.json")
