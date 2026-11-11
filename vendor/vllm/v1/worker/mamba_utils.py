from typing import Any, Optional

import torch

from vllm.config import CacheConfig
from vllm.config.vllm import KVCacheConfig
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec
from vllm.v1.worker.gpu_input_batch import CachedRequestState


def get_mamba_groups(kv_cache_config: KVCacheConfig) -> tuple[list[int], MambaSpec]:
    mamba_group_ids: list[int] = []
    mamba_specs: list[MambaSpec] = []
    for i in range(len(kv_cache_config.kv_cache_groups)):
        kv_cache_spec = kv_cache_config.kv_cache_groups[i].kv_cache_spec
        if isinstance(kv_cache_spec, MambaSpec):
            mamba_group_ids.append(i)
            mamba_specs.append(kv_cache_spec)
    assert len(mamba_group_ids) > 0, "no mamba layers in the model"
    assert all(mamba_specs[0] == spec for spec in mamba_specs)
    return mamba_group_ids, mamba_specs[0]


def mamba_copy_block_for_light_mode(
    layer_names: list[str],
    forward_context: dict[str, Any],
    src_block_id: int,
    dest_block_id: int,
):
    for layer_name in layer_names:
        kv_caches: list[list[torch.Tensor]] = forward_context[layer_name].kv_cache
        for kv_cache in kv_caches:
            if isinstance(kv_cache, torch.Tensor):
                kv_cache[dest_block_id].copy_(kv_cache[src_block_id])
            elif isinstance(kv_cache, list):
                for kv_cache_part in kv_cache:
                    kv_cache_part[dest_block_id].copy_(kv_cache_part[src_block_id])


# TODO: batch copy state like upstream PR#30877
def preprocess_mamba_for_light_mode(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    forward_context: dict[str, Any],
):
    assert cache_config.enable_prefix_caching

    mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)
    new_reqs: list[NewRequestData] = scheduler_output.scheduled_new_reqs
    cached_reqs: CachedRequestData = scheduler_output.scheduled_cached_reqs

    for kv_cache_group_id in mamba_group_ids:
        layer_names: list[str] = kv_cache_config.kv_cache_groups[
            kv_cache_group_id
        ].layer_names
        for new_req in new_reqs:
            if new_req.num_computed_tokens == 0:
                continue
            block_ids: list[int] = new_req.block_ids[kv_cache_group_id]
            assert block_ids[0] != 0, f"{block_ids=}"
            prefix_block_id, dest_block_id = block_ids[0], block_ids[1]
            mamba_copy_block_for_light_mode(
                layer_names, forward_context, prefix_block_id, dest_block_id
            )
        for i, resumed in enumerate(cached_reqs.resumed_from_preemption):
            if not resumed:
                continue
            group_block_ids: Optional[tuple[list[int], ...]] = (
                cached_reqs.new_block_ids[i]
            )
            assert group_block_ids is not None
            new_block_ids: list[int] = group_block_ids[kv_cache_group_id]
            assert len(new_block_ids) >= 2 + mamba_spec.num_speculative_blocks
            if cached_reqs.num_computed_tokens[i] == 0:
                continue
            assert new_block_ids[0] != 0, f"{new_block_ids=}"
            prefix_block_id, dest_block_id = new_block_ids[0], new_block_ids[1]
            mamba_copy_block_for_light_mode(
                layer_names, forward_context, prefix_block_id, dest_block_id
            )


def postprocess_mamba_for_light_mode(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    requests: dict[str, CachedRequestState],
    forward_context: dict[str, Any],
):
    assert cache_config.enable_prefix_caching

    mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    new_reqs: list[NewRequestData] = scheduler_output.scheduled_new_reqs
    cached_reqs: CachedRequestData = scheduler_output.scheduled_cached_reqs
    
    for kv_cache_group_id in mamba_group_ids:
        layer_names: list[str] = kv_cache_config.kv_cache_groups[
            kv_cache_group_id
        ].layer_names
        for new_req in new_reqs:
            block_ids: list[int] = new_req.block_ids[kv_cache_group_id]
            if len(block_ids) <= 2 + num_speculative_blocks:
                continue
            assert len(block_ids) == 3 + num_speculative_blocks
            src_block_id, dest_block_id = block_ids[1], block_ids[-1]
            mamba_copy_block_for_light_mode(
                layer_names, forward_context, src_block_id, dest_block_id
            )
        for i, req_id in enumerate(cached_reqs.req_ids):
            group_block_ids = cached_reqs.new_block_ids[i]
            if group_block_ids is None:
                assert not cached_reqs.resumed_from_preemption[i]
                continue
            new_block_ids: list[int] = group_block_ids[kv_cache_group_id]
            if not new_block_ids:
                assert not cached_reqs.resumed_from_preemption[i]
                continue
            if not cached_reqs.resumed_from_preemption[i]:
                assert len(new_block_ids) == 1
                block_ids = requests[req_id].block_ids[kv_cache_group_id]
                src_block_id, dest_block_id = block_ids[1], new_block_ids[0]
            else:
                if len(new_block_ids) == 2 + num_speculative_blocks:
                    continue
                assert len(new_block_ids) == 3 + num_speculative_blocks
                src_block_id, dest_block_id = new_block_ids[1], new_block_ids[-1]
            mamba_copy_block_for_light_mode(
                layer_names, forward_context, src_block_id, dest_block_id
            )
