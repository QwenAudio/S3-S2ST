import os
import threading
import time
from multiprocessing import Manager, Queue
from typing import Any, Dict, List, Optional, Union

import vllm.envs as envs
from vllm.logger import init_logger

_GB = 1024 * 1024 * 1024
ENABLE_QUICK_MODEL_LOADER: bool = False
HAS_QUICK_MODEL_LOADER: bool = False
LOAD_THREAD = None
CACHE_THREAD = None
EVICT_THREAD = None

try:
    from quick_model_loader.handle import ModelHandle
    from quick_model_loader.meta import CachingMode, CheckPointConfig, ModelContent
    from quick_model_loader.model_loader import QuickModelLoader

    MODEL_LOADER: Optional[QuickModelLoader] = None
    CHECKPOINT_CONFIG: Optional[Dict[str, CheckPointConfig]] = None
    MODEL_HANDLE: Optional[ModelHandle] = None
    LOADED_MODEL: Optional[ModelContent] = None
    HAS_QUICK_MODEL_LOADER = True
except ImportError:
    HAS_QUICK_MODEL_LOADER = False

MODEL_HANDLE_QUEUE: Optional[Queue] = None
DRIVER_HANDLE_QUEUE: Optional[Any] = None

logger = init_logger(__name__)


# Init quick model loader.
def init_model_loader(
    model_paths: List[str],
    cuda_storage_sizes: List[int] = None,  # Without cuda storage pre-allocated.
    pinned_buffer_size: int = 4 * _GB,
    models_storage_size: int = 48 * _GB,
    loaded_model_path: Optional[str] = None,
    checkpoint_config: Union[
        None, CheckPointConfig, Dict[str, CheckPointConfig]
    ] = None,
    caching_mode: CachingMode = CachingMode.LOCAL,
) -> None:
    global MODEL_LOADER, ENABLE_QUICK_MODEL_LOADER, CHECKPOINT_CONFIG
    logger.info("Start initializing quick model loader...")

    if cuda_storage_sizes is None:
        cuda_storage_sizes = [1]
    try:
        MODEL_LOADER = QuickModelLoader.with_models(
            cuda_storage_sizes=cuda_storage_sizes,
            pinned_buffer_size=pinned_buffer_size,
            models_storage_size=models_storage_size,
            cached_models_paths=model_paths,  # Cached models in CPU.
            loaded_model_path=loaded_model_path,  # Without default model in GPU.
            checkpoint_config=checkpoint_config,
            caching_mode=caching_mode,
        )
        # Create a Queue through Manager for sharing betwenn multiple processes.
        create_driver_handle_queue()
        logger.info("Quick model loader initialized.")
        # Print Cached Models.
        for model in MODEL_LOADER.cached_models:
            logger.info("Model (%s) cached.", model)
        ENABLE_QUICK_MODEL_LOADER = True
        CHECKPOINT_CONFIG = checkpoint_config
    except Exception as e:
        logger.warning("Quick model loader initialization fails: %s", e)
        MODEL_LOADER = None
        ENABLE_QUICK_MODEL_LOADER = False
        CHECKPOINT_CONFIG = None


def get_quick_model_loader() -> QuickModelLoader:
    global MODEL_LOADER
    return MODEL_LOADER


def is_quick_model_loader_enabled() -> bool:
    global ENABLE_QUICK_MODEL_LOADER
    return ENABLE_QUICK_MODEL_LOADER


def clear_model_loader():
    global MODEL_LOADER
    if MODEL_LOADER is not None:
        MODEL_LOADER.clear()
        logger.info("Clear all models in quick model loader.")
    else:
        logger.warning("Quick model loader is not initialized, cannot clear models.")


def load_model_in_thread(
    model: str,
    model_loader: QuickModelLoader,
    checkpoint_config: Union[
        None, CheckPointConfig, Dict[str, CheckPointConfig]
    ] = None,
) -> None:
    try:
        start_time = time.perf_counter()
        if isinstance(checkpoint_config, dict):
            if model not in checkpoint_config:
                raise ValueError(f"Checkpoint config for model {model} is not found.")
            loaded_model = model_loader.load_model(model, checkpoint_config[model])
        else:
            loaded_model = model_loader.load_model(model, checkpoint_config)
        set_loaded_model(loaded_model)
        set_model_handle(ModelHandle(loaded_model))
        # put model handle to driver handle queue if using v1 engine
        driver_handle_queue = get_driver_handle_queue()
        if driver_handle_queue is None:
            logger.error("Driver handle queue is not initialized.")
        else:
            driver_handle_queue.put(get_model_handle())
        end_time = time.perf_counter()
        logger.info(
            "Model %s loaded in thread in %.2f seconds.", model, (end_time - start_time)
        )
    except Exception as e:
        logger.error("Error in loading model thread %s: %s", model, e)


def start_thread_for_loading_model(model: str) -> None:
    global MODEL_LOADER, LOAD_THREAD, CHECKPOINT_CONFIG
    if LOAD_THREAD is not None:
        logger.error("Loading thread is running, cannot start another one.")
        return

    if MODEL_LOADER is not None and CHECKPOINT_CONFIG is not None:
        LOAD_THREAD = threading.Thread(
            target=load_model_in_thread, args=(model, MODEL_LOADER, CHECKPOINT_CONFIG)
        )
        logger.info("Start loading model %s in thread.", model)
        LOAD_THREAD.start()  # Start the thread.
    else:
        logger.warning(
            "Model loader or checkpoint config is not initialized, "
            "cannot start loading thread."
        )


def join_thread_for_loading_model() -> None:
    global LOAD_THREAD
    if LOAD_THREAD is not None:
        LOAD_THREAD.join()
        LOAD_THREAD = None
        logger.info("Loading thread joined.")
    else:
        logger.warning("No loading thread to join.")


def cache_model_in_thread(
    model: str,
    model_loader: QuickModelLoader,
    checkpoint_config: Union[
        None, CheckPointConfig, Dict[str, CheckPointConfig]
    ] = None,
) -> None:
    try:
        model_loader.cache_model(model, checkpoint_config)
        logger.info("Model %s cached in thread.", model)
    except Exception as e:
        logger.error("Error in caching model thread %s: %s", model, e)


def start_thread_for_caching_model(model: str) -> None:
    global MODEL_LOADER, CACHE_THREAD, CHECKPOINT_CONFIG
    if CACHE_THREAD is not None:
        logger.error("Caching thread is running, cannot start another one.")
        return

    if MODEL_LOADER is not None and CHECKPOINT_CONFIG is not None:
        CACHE_THREAD = threading.Thread(
            target=cache_model_in_thread, args=(model, MODEL_LOADER, CHECKPOINT_CONFIG)
        )
        logger.info("Start caching model %s in thread.", model)
        CACHE_THREAD.start()  # Start the thread.
    else:
        logger.warning(
            "Model loader or checkpoint config is not initialized, "
            "cannot start caching thread."
        )


def join_thread_for_caching_model() -> None:
    global CACHE_THREAD
    if CACHE_THREAD is not None:
        CACHE_THREAD.join()
        CACHE_THREAD = None
        logger.info("Caching thread joined.")
    else:
        logger.warning("No caching thread to join.")


def evict_model_in_thread(model: str, model_loader: QuickModelLoader) -> None:
    try:
        model_loader.evict_cached_model(model)
        logger.info("Model %s evicted in thread.", model)
    except Exception as e:
        logger.error("Error in evicting model thread %s: %s", model, e)


def start_thread_for_evicting_model(model: str) -> None:
    global MODEL_LOADER, EVICT_THREAD
    if EVICT_THREAD is not None:
        logger.error("Evicting thread is running, cannot start another one.")
        return

    if MODEL_LOADER is not None:
        EVICT_THREAD = threading.Thread(
            target=evict_model_in_thread, args=(model, MODEL_LOADER)
        )
        logger.info("Start evicting model %s in thread.", model)
        EVICT_THREAD.start()  # Start the thread.
    else:
        logger.warning("Model loader is not initialized, cannot start evicting thread.")


def join_thread_for_evicting_model() -> None:
    global EVICT_THREAD
    if EVICT_THREAD is not None:
        EVICT_THREAD.join()
        EVICT_THREAD = None
        logger.info("Evicting thread joined.")
    else:
        logger.warning("No evicting thread to join.")


def set_loaded_model(loaded_model: ModelContent):
    global LOADED_MODEL
    LOADED_MODEL = loaded_model


def get_loaded_model() -> Optional[ModelContent]:
    global LOADED_MODEL
    return LOADED_MODEL


def set_model_handle(model_handle: ModelHandle):
    global MODEL_HANDLE
    MODEL_HANDLE = model_handle


def get_model_handle() -> Optional[ModelHandle]:
    global MODEL_HANDLE
    return MODEL_HANDLE


def set_model_handle_queue(queue: Queue):
    global MODEL_HANDLE_QUEUE
    MODEL_HANDLE_QUEUE = queue


def get_model_handle_queue() -> Queue:
    global MODEL_HANDLE_QUEUE
    return MODEL_HANDLE_QUEUE


def delete_model_handle_queue():
    global MODEL_HANDLE_QUEUE
    if MODEL_HANDLE_QUEUE is not None:
        while not MODEL_HANDLE_QUEUE.empty():
            _ = MODEL_HANDLE_QUEUE.get()
        MODEL_HANDLE_QUEUE.close()
        MODEL_HANDLE_QUEUE.join_thread()
        MODEL_HANDLE_QUEUE = None
    logger.info("Model handle queue has been deleted.")


def create_driver_handle_queue():
    global DRIVER_HANDLE_QUEUE
    if DRIVER_HANDLE_QUEUE is not None:
        logger.warning("Driver handle queue already exists, skipping creation.")
        return
    DRIVER_HANDLE_QUEUE = Manager().Queue()
    logger.info(
        "create DRIVER_HANDLE_QUEUE=%s, PID=%d", DRIVER_HANDLE_QUEUE, os.getpid()
    )


def set_driver_handle_queue(queue: Any):
    global DRIVER_HANDLE_QUEUE
    if DRIVER_HANDLE_QUEUE is not None:
        logger.warning("Driver handle queue already exists, overwriting it.")
    DRIVER_HANDLE_QUEUE = queue
    logger.info("Set DRIVER_HANDLE_QUEUE=%s, PID=%d", DRIVER_HANDLE_QUEUE, os.getpid())


def get_driver_handle_queue() -> Optional[Queue]:
    global DRIVER_HANDLE_QUEUE
    return DRIVER_HANDLE_QUEUE


def delete_driver_handle_queue():
    global DRIVER_HANDLE_QUEUE
    if DRIVER_HANDLE_QUEUE is not None:
        while not DRIVER_HANDLE_QUEUE.empty():
            DRIVER_HANDLE_QUEUE.get()
        DRIVER_HANDLE_QUEUE = None
    logger.info("Driver handle queue has been deleted.")
