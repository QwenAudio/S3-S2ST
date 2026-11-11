import os
import socket
from typing import TYPE_CHECKING, Any, Callable

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, get_ip_address

if TYPE_CHECKING:
    from vllm.worker.worker import Worker

logger = init_logger(__name__)

# ============== Environment Variable Utilities ==============

_RUND_TEMPLATE_ENV_FIXERS: list[tuple[Callable[[], None], Callable[[], bool]]] = []
_RUND_TEMPLATE_ENV_CHANGES: dict[str, str] = {}

def env_fixer(
    func: Callable[[], None] | None = None,
    *,
    when: Callable[[], bool] | str | None = None,
) -> Callable[[], None]:
    """
    Decorator to register an environment fixer function.
    
    Args:
        when: Condition to run the fixer. Can be:
            - None: always run
            - str: env var name (runs if set and non-empty)
            - callable: custom condition function
    
    Example::
    
        @env_fixer
        def fix_always() -> None: ...
        
        @env_fixer(when="DS_LLM_MODE")
        def fix_if_mode_set() -> None: ...
        
        @env_fixer(when=lambda: os.getenv("MODE") in ("prefill", "decode"))
        def fix_pd() -> None: ...
    """
    if isinstance(when, str):
        env_name = when
        condition = lambda: bool(os.getenv(env_name))
    elif callable(when):
        condition = when
    else:
        condition = lambda: True
    
    def decorator(fn: Callable[[], None]) -> Callable[[], None]:
        _RUND_TEMPLATE_ENV_FIXERS.append((fn, condition))
        return fn
    
    if func is not None:
        return decorator(func)
    else:
        return decorator  # type: ignore[return-value]

def update_os_environ(name: str, value: Any) -> None:
    if value is None:
        os.environ.pop(name, None)
        str_value = ""
    elif isinstance(value, bool):
        str_value = "1" if value else "0"
    elif isinstance(value, (list, set)):
        str_value = ",".join(str(v) for v in value)
    else:
        str_value = str(value)

    if value is not None:
        os.environ[name] = str_value
        _RUND_TEMPLATE_ENV_CHANGES[name] = str_value

def update_vllm_env(name: str, value: Any) -> None:
    """Update a vLLM environment variable."""
    if name not in envs.environment_variables:
        logger.warning("Unknown vLLM environment variable: %s", name)
        return

    setattr(envs, name, value)


# ============== VLLM Env Fixers ==============

@env_fixer(when=lambda: os.getenv("DASHGEN_DEPLOYMENT_ROLE") in ("prefill", "decode"))  # type: ignore[call-arg]
def fix_pd_env() -> None:
    """Fix PD disaggregation environment for P/D RPC server."""
    update_vllm_env("VLLM_HOST_IP", "")

    interface = os.getenv("DS_LLM_DISAGGREGATED_PREFILL_IF_NAME", "eth1")
    host_ip = get_ip_address(interface) or get_ip()

    logger.info("KV_Connector: Using IP %s", host_ip)
    update_vllm_env("VLLM_HOST_IP", host_ip)
    update_os_environ("VLLM_HOST_IP", host_ip)

    host_name = socket.gethostname()

    # hostname is always the same as the POD_NAME env
    # make sure it has the prefix "ds-" in DashScope
    if not host_name.startswith("ds-"):
        raise ValueError(f"Expected hostname to start with 'ds-', but got: {host_name}")

    update_os_environ("POD_NAME", host_name)

def reset_vllm_envs_after_checkpoint() -> dict[str, str]:
    """Run all enabled environment fixers after checkpoint restore."""
    for fixer, condition in _RUND_TEMPLATE_ENV_FIXERS:
        if not condition():
            continue
        try:
            fixer()
        except Exception as e:
            logger.warning("Env fixer %s failed: %s", fixer.__name__, e)

    result = dict(_RUND_TEMPLATE_ENV_CHANGES)
    _RUND_TEMPLATE_ENV_CHANGES.clear()
    return result


# ============== Epsilon Integration ==============

def is_rund_template_enabled() -> bool:
    """Check if RUND application template is enabled."""
    if not (envs.VLLM_ENABLE_RUND_APPLICATION_TEMPLATE
            or os.getenv("DS_LLM_ENABLE_RUND_APPLICATION_TEMPLATE") == "1"):
        return False
    try:
        from epsilon import is_snapstart_enable
        snapstart_enable = is_snapstart_enable()
        if not snapstart_enable:
            logger.warning(
                "[RUND_TEMPLATE] Enable Rund application template, "
                "but snapstart is not enabled."
            )
        else:
            logger.info("[RUND_TEMPLATE] Enable Rund application template.")
        return snapstart_enable
    except ImportError:
        return False
    except Exception as e:
        logger.error("[RUND_TEMPLATE] Snapstart check failed: %s", e)
        return False


def register_model_and_kv_caches(worker: "Worker") -> None:
    """Register model and KV caches with epsilon if enabled."""
    try:
        from epsilon import register_kv_caches, register_model
        register_model(worker.model_runner.model)
        register_kv_caches(worker.model_runner.kv_caches)
        logger.info("[RUND_TEMPLATE] Registered model and KV caches with epsilon")
    except ImportError:
        logger.debug("[RUND_TEMPLATE] Epsilon not found, skipping registration")
    except Exception as e:
        logger.error("[RUND_TEMPLATE] Epsilon registration failed: %s", e)


def start_rund_checkpoint() -> None:
    """Start RUND checkpoint if enabled."""
    try:
        from epsilon import snapstart_checkpoint
        engines_num = int(os.getenv("DS_LLM_MULTI_ENGINE_NUM", "1"))
        logger.info("[RUND_TEMPLATE] Starting snapstart checkpoint")
        snapstart_checkpoint(wait_mode=1, worker_num=engines_num, timeout=300)
    except ImportError:
        logger.debug("[RUND_TEMPLATE] Epsilon not found, skipping checkpoint")
    except Exception as e:
        logger.error("[RUND_TEMPLATE] Snapstart checkpoint failed: %s", e)
