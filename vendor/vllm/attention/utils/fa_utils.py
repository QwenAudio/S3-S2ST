# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import importlib
import importlib.util
import inspect
from typing import Optional

import torch
from torch.library import CustomOpDef

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_ppu

logger = init_logger(__name__)

if current_platform.is_cuda():
    from vllm import _custom_ops as ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash

if (
    current_platform.is_cuda()
    # The performance gain on PPU is not fully verified, don't enable by default
    and not is_ppu()
    and envs.VLLM_FLASH_ATTN_USE_UPSTREAM is None
    and importlib.util.find_spec("flash_attn_3") is not None
):
    device_capability = current_platform.get_device_capability()
    assert device_capability is not None

    # Note: fa3 is not supported with sm100.
    if (device_capability.major == 9 or is_ppu()):
        logger.info_once(
            "Upstream flash_attn_3 is detected. "
            "Enable upstream flash_attn_3 by default by setting "
            "VLLM_FLASH_ATTN_USE_UPSTREAM to True. To disable, set "
            "VLLM_FLASH_ATTN_USE_UPSTREAM=0 in the environment variables."
        )
        envs.VLLM_FLASH_ATTN_USE_UPSTREAM = True
    else:
        assert not envs.VLLM_FLASH_ATTN_USE_UPSTREAM, (
            "Upstream flash_attn_3 requires hopper architecture "
            "(compute capability >= 9.0) or PPU."
        )


if current_platform.is_cuda():
    if envs.VLLM_FLASH_ATTN_USE_UPSTREAM and not is_ppu():
        logger.info_once("Using upstream flash_attn_3 for flash attention backend.")

        from flash_attn_interface import _flash_attn_forward, get_scheduler_metadata

        # Recent versions of upstream FA3 has upgraded to support torch compile,
        # where the out argument is no longer supported.
        if isinstance(_flash_attn_forward, CustomOpDef):
            def flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q,
                cu_seqlens_q,
                max_seqlen_k,
                cu_seqlens_k=None,  # only used for non-paged prefill
                seqused_k=None,
                qv=None,
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
                window_size: Optional[list[int]] = None,
                softcap=0.0,  # 0.0 means deactivated
                alibi_slopes=None,
                deterministic=False,
                block_table=None,
                return_attn_probs=False,
                # FA3 Only
                scheduler_metadata=None,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                num_splits: int = 0,
                # Version selector
                fa_version: int = 3,
            ):
                assert fa_version == 3, (
                    "fa_version=3 is required when using upstream FA3"
                )
                if softmax_scale is None:
                    softmax_scale = (
                        q.shape[-1] + (qv.shape[-1] if qv is not None else 0)
                    ) ** (-0.5)
                out, softmax_lse, *rest = _flash_attn_forward(
                    q=q,
                    k=k,
                    v=v,
                    k_new=None,
                    v_new=None,
                    qv=qv,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    cu_seqlens_k_new=None,
                    seqused_q=None,
                    seqused_k=seqused_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    page_table=block_table,
                    kv_batch_idx=None,
                    leftpad_k=None,
                    rotary_cos=None,
                    rotary_sin=None,
                    seqlens_rotary=None,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size_left=window_size[0] if window_size is not None else -1,
                    window_size_right=window_size[1] if window_size is not None else -1,
                    attention_chunk=0,
                    softcap=softcap,
                    rotary_interleaved=True,
                    scheduler_metadata=scheduler_metadata,
                    num_splits=num_splits,
                    pack_gqa=None,
                    sm_margin=0,
                )
                return (out, softmax_lse) if return_attn_probs else out
        else:
            def flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q,
                cu_seqlens_q,
                max_seqlen_k,
                cu_seqlens_k=None,  # only used for non-paged prefill
                seqused_k=None,
                qv=None,
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
                window_size: Optional[list[int]] = None,
                softcap=0.0,  # 0.0 means deactivated
                alibi_slopes=None,
                deterministic=False,
                block_table=None,
                return_attn_probs=False,
                out=None,
                # FA3 Only
                scheduler_metadata=None,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                num_splits: int = 0,
                # Version selector
                fa_version: int = 3,
            ):
                assert fa_version == 3, (
                    "fa_version=3 is required when using upstream FA3"
                )
                if softmax_scale is None:
                    softmax_scale = (
                        q.shape[-1] + (qv.shape[-1] if qv is not None else 0)
                    ) ** (-0.5)
                out, softmax_lse, *rest = _flash_attn_forward(
                    q=q,
                    k=k,
                    v=v,
                    k_new=None,
                    v_new=None,
                    qv=qv,
                    out=out,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    cu_seqlens_k_new=None,
                    seqused_q=None,
                    seqused_k=seqused_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    page_table=block_table,
                    kv_batch_idx=None,
                    leftpad_k=None,
                    rotary_cos=None,
                    rotary_sin=None,
                    seqlens_rotary=None,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size if window_size is not None else (-1, -1),
                    attention_chunk=0,
                    softcap=softcap,
                    rotary_interleaved=True,
                    scheduler_metadata=scheduler_metadata,
                    num_splits=num_splits,
                    pack_gqa=None,
                    sm_margin=0,
                )
                return (out, softmax_lse) if return_attn_probs else out

        # GPU fa3: cannot be used as vllm-flash-attn as it doesn't support cuda graph.
        is_vllm_fa = False
        scheduler_meta_supported = True

    elif envs.VLLM_FLASH_ATTN_USE_UPSTREAM and is_ppu():
        logger.info_once("Using upstream flash_attn_3 for flash attention backend.")

        import flash_attn_3._C

        def maybe_contiguous(x):
            return x.contiguous() if x is not None and x.stride(-1) != 1 else x

        def get_scheduler_metadata(
            batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_kv, headdim,
            cache_seqlens: torch.Tensor,
            qkv_dtype=torch.bfloat16,
            headdim_v=None,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_k_new: Optional[torch.Tensor] = None,
            cache_leftpad: Optional[torch.Tensor] = None,
            page_size: Optional[int] = None,
            max_seqlen_k_new=0,
            causal=False,
            window_size=(-1, -1),  # -1 means infinite context window
            has_softcap=False,
            num_splits=0,    # Can be tuned for speed
            pack_gqa=None,   # Can be tuned for speed
            sm_margin=0,     # Can be tuned if some SMs are used for communication
        ):
            cache_seqlens = maybe_contiguous(cache_seqlens)
            if headdim_v is None:
                headdim_v = headdim
            scheduler_metadata = torch.ops.flash_attn_3.get_scheduler_metadata(
                batch_size, max_seqlen_q, max_seqlen_k,
                num_heads_q, num_heads_kv, headdim, headdim_v,
                qkv_dtype,
                cache_seqlens,
                cu_seqlens_q,
                None,  # cu_seqlens_k
                cu_seqlens_k_new,
                None,  # seqused_q
                cache_leftpad,
                page_size,
                max_seqlen_k_new,
                causal,
                window_size[0], window_size[1],
                0, # attention_chunk
                has_softcap,
                num_splits,
                pack_gqa,
                sm_margin,
            )

            return scheduler_metadata

        def flash_attn_varlen_func(
            q,
            k,
            v,
            max_seqlen_q,
            cu_seqlens_q,
            max_seqlen_k,
            cu_seqlens_k=None, # only used for non-paged prefill
            seqused_k=None,
            q_v=None,
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
            window_size: Optional[list[int]] = None,
            softcap=0.0, # 0.0 means deactivated
            alibi_slopes=None,
            deterministic=False,
            return_attn_probs=False,
            block_table=None,
            return_softmax_lse=False,
            out=None,
            # FA3 Only
            scheduler_metadata=None,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            num_splits: int = 0,
            # Version selector
            fa_version: int = 3, # DEFAULT_FA_VERSION,
            s_aux=None,
            # FA2 PD split
            num_prefill: int = -1,
            max_seqlen_k_decode: int = 0,
            max_seqlen_k_prefill: int = 0,
        ):
            assert fa_version == 3, "This function only supports FA version 3"

            assert cu_seqlens_k is not None or seqused_k is not None, \
                "cu_seqlens_k or seqused_k must be provided"
            assert cu_seqlens_k is None or seqused_k is None, \
                "cu_seqlens_k and seqused_k cannot be provided at the same time"
            assert block_table is None or seqused_k is not None, \
                "seqused_k must be provided if block_table is provided"

            if s_aux is not None:
                raise NotImplementedError("PPU FA does not support s_aux")
            if softmax_scale is None:
                softmax_scale = q.shape[-1] ** (-0.5)
            # custom op does not support non-tuple input
            real_window_size: tuple[int, int]
            if window_size is None:
                real_window_size = (-1, -1)
            else:
                assert len(window_size) == 2
                real_window_size = (window_size[0], window_size[1])
            q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

            dummy_cu_seqlens_k = torch.empty_like(cu_seqlens_q)  # noqa

            assert alibi_slopes is None, "Alibi is not supported in FA3"

            out, softmax_lse, _, _ = torch.ops.flash_attn_3.fwd(
                q, k, v,
                None, None,       # k_new, v_new
                q_v,
                out,
                cu_seqlens_q,
                cu_seqlens_k,     # cu_seqlens_k
                None,             # cu_seqlens_k_new
                None, seqused_k,  # seqused_q, seqused_k
                max_seqlen_q, max_seqlen_k,
                block_table,
                None,             # kv_batch_idx
                None,             # leftpad_k
                None, None, None, # rotary_cos, rotary_sin, seqlens_rotary
                q_descale, k_descale, v_descale,
                softmax_scale,
                causal,
                real_window_size[0], real_window_size[1],
                0,                # attention chunk
                softcap,
                True,             # rotary_interleaved
                scheduler_metadata,
                num_splits,
                None,             # pack_gqa
                0                 # sm_margin
            )
            return (out, softmax_lse) if return_softmax_lse else out

        # PPU fa3: can be used as vllm-flash-attn as it supports cuda graph.
        is_vllm_fa = True
        scheduler_meta_supported = True

    else:
        try:
            from vllm.vllm_flash_attn import flash_attn_varlen_func

            is_vllm_fa = True
        except (ImportError, ModuleNotFoundError):
            try:
                from vllm_flash_attn import flash_attn_varlen_func

                is_vllm_fa = True
            except (ImportError, ModuleNotFoundError):
                from flash_attn import flash_attn_varlen_func

                is_vllm_fa = False
                fa_cuda_graph_supported = False

        try:
            from vllm.vllm_flash_attn import get_scheduler_metadata

            scheduler_meta_supported = True
        except (ImportError, ModuleNotFoundError):
            try:
                from vllm_flash_attn import get_scheduler_metadata

                scheduler_meta_supported = True
            except (ImportError, ModuleNotFoundError):
                get_scheduler_metadata = lambda *args, **kwargs: None
                scheduler_meta_supported = False

elif current_platform.is_xpu():
    from vllm._ipex_ops import ipex_ops as ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash
    flash_attn_varlen_func = ops.flash_attn_varlen_func
    get_scheduler_metadata = ops.get_scheduler_metadata
    is_vllm_fa = True
    scheduler_meta_supported = True

elif current_platform.is_rocm():
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Rocm platform requires upstream flash-attn "
            "to be installed. Please install flash-attn first."
        ) from e


def get_flash_attn_version(requires_alibi: bool = False) -> int | None:
    # import here to avoid circular dependencies
    from vllm.platforms import current_platform

    if envs.VLLM_USE_BLADNN_FLASH_ATTN:
        try:
            from bladnn import flash_attn_varlen_func, flash_attn_with_kvcache

            logger.info_once(
                "Set flash-attn version to 2, as bladnn's flash-attn is used."
            )
            return 2
        except (ImportError, ModuleNotFoundError):
            return None

    if current_platform.is_xpu():
        logger.info_once("Set flash-attn version to 2, as XPU platform is used.")
        return 2

    if envs.VLLM_FLASH_ATTN_USE_UPSTREAM:
        logger.info_once(
            "Set flash-attn version to 3, as upstream flash_attn_3 is used."
        )
        return 3

    try:
        try:
            from vllm.vllm_flash_attn.flash_attn_interface import (
                fa_version_unsupported_reason,
                is_fa_version_supported,
            )
        except (ImportError, ModuleNotFoundError):
            from vllm_flash_attn.flash_attn_interface import (
                fa_version_unsupported_reason,
                is_fa_version_supported,
            )
        device_capability = current_platform.get_device_capability()

        assert device_capability is not None

        # 1. default version depending on platform
        fa_version = (
            3 if (device_capability.major >= 9 and is_fa_version_supported(3)) else 2
        )

        # 2. override if passed by environment or config
        from vllm.config import get_current_vllm_config, has_current_vllm_config

        if has_current_vllm_config():
            vllm_config = get_current_vllm_config()
            if vllm_config.attention_config.flash_attn_version is not None:
                fa_version = vllm_config.attention_config.flash_attn_version

        assert fa_version == 3 or not envs.VLLM_FLASH_ATTN_USE_UPSTREAM, (
            "Upstream flash_attn_3 cannot be used with fa_version != 3, set "
            "fa_version to 2 with VLLM_FLASH_ATTN_VERSION=2, or disabling upstream "
            "flash_attn_3 with VLLM_FLASH_ATTN_USE_UPSTREAM=0."
        )

        if requires_alibi and fa_version == 3:
            logger.warning_once(
                "Cannot use FA version 3 with ALiBi, defaulting to FA version 2."
            )
            fa_version = 2

        if not is_fa_version_supported(fa_version):
            logger.error(
                "Cannot use FA version %d is not supported due to %s",
                fa_version,
                fa_version_unsupported_reason(fa_version),
            )

        assert is_fa_version_supported(fa_version)
        logger.info_once("Set flash-attn version to %d", fa_version)
        return fa_version
    except (ImportError, AssertionError):
        logger.info_once(
            "Set flash-attn version to 2, as is_fa_version_supported cannot be imported"
        )
        return 2


def flash_attn_supports_fp8() -> bool:
    return (
        get_flash_attn_version() == 3
        and current_platform.get_device_capability().major == 9
    )


def flash_attn_supports_sinks() -> bool:
    if current_platform.is_xpu():
        return True
    else:
        return get_flash_attn_version() == 3


def flash_attn_supports_mla():
    from vllm.platforms import current_platform

    if current_platform.is_cuda():
        if envs.VLLM_FLASH_ATTN_USE_UPSTREAM:
            return (
                (current_platform.get_device_capability()[0] == 9
                 and "qv" in inspect.signature(flash_attn_varlen_func).parameters)
                or
                (is_ppu()
                 and "q_v" in inspect.signature(flash_attn_varlen_func).parameters)
            )
        else:
            with contextlib.suppress(ImportError, ModuleNotFoundError):
                try:
                    from vllm.vllm_flash_attn.flash_attn_interface import (
                        is_fa_version_supported,
                    )
                except (ImportError, ModuleNotFoundError):
                    from vllm_flash_attn.flash_attn_interface import (
                        is_fa_version_supported,
                    )

                return (
                    is_fa_version_supported(3)
                    and current_platform.get_device_capability()[0] == 9
                )
    return False


def is_flash_attn_varlen_func_available() -> bool:
    return current_platform.is_cuda() or current_platform.is_xpu()
