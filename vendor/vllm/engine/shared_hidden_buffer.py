"""
Shared buffer for prefill hidden states transfer between thinker worker 
process and main process.

Supports two backends:
1. SharedMemory (default): Uses named shared memory for cross-process communication
2. Mooncake: Uses Mooncake distributed store for remote/distributed scenarios

Design (simplified for prefill-only):
- Producer (thinker worker): writes prefill hidden states once per request
- Consumer (main process): reads prefill hidden states once per request
- Each request has exactly one write and one read
"""

import contextlib
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Shared memory name prefix
SHM_NAME_PREFIX = "vllm_accept_hidden_"

# Backend types
BACKEND_SHM = "shm"
BACKEND_MOONCAKE = "mooncake"

# Mooncake default configurations
DEFAULT_GLOBAL_SEGMENT_SIZE = 3355443200  # 3.125 GiB
DEFAULT_LOCAL_BUFFER_SIZE = 1073741824  # 1.0 GiB
DEFAULT_EVICTION_FACTOR = 1.2  # Extra 20% space for eviction buffer

# Fast transfer configuration
DEFAULT_FAST_TRANSFER_BUFFER_SIZE = 1073741824  # 1 GiB
FAST_TRANSFER_MAGIC = 0xFA57BEEF  # Magic number for fast transfer SHM format
# Default concurrent factor: pool only needs to hold a few requests at a time
# because buffers are allocated and freed per-transfer (not held long-term).
DEFAULT_FAST_TRANSFER_CONCURRENT_FACTOR = 2


def calculate_fast_transfer_buffer_size(
    max_model_len: int,
    hidden_size: int,
    dtype_size: int = 2,
    concurrent_factor: int = DEFAULT_FAST_TRANSFER_CONCURRENT_FACTOR,
) -> int:
    """Calculate the fast-transfer pinned-memory pool size.

    The pool is used as a *transient* buffer: each write/read allocates,
    transfers, then immediately frees.  So the pool only needs to hold
    ``concurrent_factor`` requests worth of data at most (typically 1-2).

    The result is rounded up to the next power of two because
    ``TensorMemoryPool`` uses a buddy allocator internally.

    Args:
        max_model_len: Maximum context length (tokens).
        hidden_size:   Hidden dimension of the model.
        dtype_size:    Bytes per element (default 2 for bfloat16).
        concurrent_factor: How many concurrent in-flight buffers to
            support (default 2 – one active + alignment headroom).

    Returns:
        Pool size in bytes (power of two, >= 1 GiB minimum).
    """
    single_request = max_model_len * hidden_size * dtype_size
    raw = concurrent_factor * single_request
    # Round up to the next power of two
    pool_size = 1 << (raw - 1).bit_length() if raw > 0 else 0
    # Enforce a reasonable minimum (1 GiB)
    pool_size = max(pool_size, DEFAULT_FAST_TRANSFER_BUFFER_SIZE)
    logger.info(
        "Calculated fast_transfer_buffer_size: %d bytes (%.2f GiB) "
        "[max_model_len=%d, hidden_size=%d, dtype_size=%d, "
        "concurrent_factor=%d, single_request=%d bytes]",
        pool_size, pool_size / (1024**3),
        max_model_len, hidden_size, dtype_size,
        concurrent_factor, single_request,
    )
    return pool_size


# Default Mooncake configuration values
# These are set by omni main process and shared with workers via environment variables
DEFAULT_MOONCAKE_LOCAL_HOSTNAME = "127.0.0.1"
DEFAULT_MOONCAKE_MASTER_HOST = "127.0.0.1"
# Note: mooncake_master has two ports:
# - HTTP metadata server port (default 8080) - for metadata queries
# - RPC service port (default 50051) - for actual data transfer
# master_server_address should use the RPC port
DEFAULT_MOONCAKE_MASTER_RPC_PORT = 50051
DEFAULT_MOONCAKE_PROTOCOL = "tcp"


@dataclass
class MooncakeStoreConfig:
    """Configuration for Mooncake hidden states transfer backend.
    
    Configuration is loaded from environment variables with defaults.
    The omni main process sets these environment variables after starting
    the mooncake master, so workers can reuse the same configuration.
    
    Environment variables:
        - MOONCAKE_LOCAL_HOSTNAME: Local hostname for this node (default: "127.0.0.1")
        - MOONCAKE_MASTER_SERVER_ADDRESS: Master RPC server address
          (default: "127.0.0.1:50051")
        - MOONCAKE_PROTOCOL: Network protocol (default: "tcp")
        - MOONCAKE_DEVICE_NAME: RDMA device name (default: "")
        - MOONCAKE_GLOBAL_SEGMENT_SIZE: Global segment size in bytes
        - MOONCAKE_LOCAL_BUFFER_SIZE: Local buffer size in bytes
        - MOONCAKE_EVICTION_FACTOR: Eviction factor for sizing (default: 1.2)
    """
    local_hostname: str
    metadata_server: str  # Always "P2PHANDSHAKE" for our use case
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    device_name: str
    master_server_address: str
    eviction_factor: float = DEFAULT_EVICTION_FACTOR

    @staticmethod
    def load_from_env() -> "MooncakeStoreConfig":
        """Load config from environment variables with defaults.
        
        The omni main process sets MOONCAKE_MASTER_SERVER_ADDRESS after
        starting the mooncake master, so workers automatically use the
        correct master address.
        """
        local_hostname = os.getenv(
            "MOONCAKE_LOCAL_HOSTNAME", DEFAULT_MOONCAKE_LOCAL_HOSTNAME
        )
        # Master RPC server address is set by omni main process
        # Note: This is the RPC port (50051), not the HTTP metadata port (8080)
        master_server_address = os.getenv(
            "MOONCAKE_MASTER_SERVER_ADDRESS",
            f"{DEFAULT_MOONCAKE_MASTER_HOST}:{DEFAULT_MOONCAKE_MASTER_RPC_PORT}"
        )
        protocol = os.getenv("MOONCAKE_PROTOCOL", DEFAULT_MOONCAKE_PROTOCOL)
        device_name = os.getenv("MOONCAKE_DEVICE_NAME", "")
        
        global_segment_size = int(os.getenv(
            "MOONCAKE_GLOBAL_SEGMENT_SIZE", str(DEFAULT_GLOBAL_SEGMENT_SIZE)
        ))
        local_buffer_size = int(os.getenv(
            "MOONCAKE_LOCAL_BUFFER_SIZE", str(DEFAULT_LOCAL_BUFFER_SIZE)
        ))
        eviction_factor = float(os.getenv(
            "MOONCAKE_EVICTION_FACTOR", str(DEFAULT_EVICTION_FACTOR)
        ))
        
        logger.info(
            "Mooncake config: local_hostname=%s, master_server_address=%s, "
            "protocol=%s, device_name=%s",
            local_hostname, master_server_address, protocol, device_name
        )
        
        return MooncakeStoreConfig(
            local_hostname=local_hostname,
            metadata_server="P2PHANDSHAKE",  # Always use P2PHANDSHAKE mode
            global_segment_size=global_segment_size,
            local_buffer_size=local_buffer_size,
            protocol=protocol,
            device_name=device_name,
            master_server_address=master_server_address,
            eviction_factor=eviction_factor,
        )

    @staticmethod
    def calculate_global_segment_size(
        max_num_seqs: int,
        max_model_len: int,
        hidden_size: int,
        dtype_size: int = 2,  # bfloat16 = 2 bytes
        eviction_factor: float = DEFAULT_EVICTION_FACTOR,
    ) -> int:
        """
        Calculate the global segment size based on service configuration.
        
        Since we only read/write prefill hidden states, the max shm size is:
        max_num_seqs * max_model_len * hidden_size * dtype_size * eviction_factor
        
        Args:
            max_num_seqs: Maximum number of concurrent requests
            max_model_len: Maximum context length
            hidden_size: Hidden dimension size of the model
            dtype_size: Size of each element in bytes (default: 2 for bfloat16)
            eviction_factor: Extra factor to account for eviction (default: 1.2)
        
        Returns:
            Calculated global segment size in bytes
        """
        base_size = max_num_seqs * max_model_len * hidden_size * dtype_size
        total_size = int(base_size * eviction_factor)
        logger.info(
            "Calculated global_segment_size: %d bytes (%.2f GiB) "
            "[max_num_seqs=%d, max_model_len=%d, hidden_size=%d, "
            "dtype_size=%d, eviction_factor=%.2f]",
            total_size, total_size / (1024**3),
            max_num_seqs, max_model_len, hidden_size, dtype_size, eviction_factor
        )
        return total_size


class HiddenBufferBackend(ABC):
    """Abstract base class for hidden buffer backends."""
    
    @abstractmethod
    def write(
        self,
        tensor: torch.Tensor,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> bool:
        """Write prefill hidden states for a request."""
        pass
    
    @abstractmethod
    def read_by_request_id(
        self,
        request_id: str,
        device: str = "cuda",
        timeout: Optional[float] = None,
        wait: bool = True,
    ) -> Optional[torch.Tensor]:
        """Read prefill hidden states for a specific request_id."""
        pass
    
    @abstractmethod
    def close(self):
        """Close the backend."""
        pass


class MooncakeHiddenBuffer(HiddenBufferBackend):
    """
    Mooncake-based backend for hidden states transfer.
    Uses MooncakeDistributedStore for key-value storage.
    """
    
    def __init__(
        self,
        dtype: torch.dtype = torch.bfloat16,
        is_creator: bool = False,
        max_num_seqs: Optional[int] = None,
        max_model_len: Optional[int] = None,
        hidden_size: Optional[int] = None,
    ):
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
                "to run vLLM with Mooncake backend."
            ) from e
        
        self.dtype = dtype
        self.is_creator = is_creator
        self.store: Optional[MooncakeDistributedStore] = None
        self._initialized = False
        
        try:
            self.config = MooncakeStoreConfig.load_from_env()
            
            logger.info(
                "Mooncake using P2PHANDSHAKE mode with master: %s",
                self.config.master_server_address
            )
            
            if is_creator:
                # Calculate global_segment_size based on service configuration
                # if max_num_seqs, max_model_len, and hidden_size are provided
                if (max_num_seqs is not None and 
                    max_model_len is not None and 
                    hidden_size is not None):
                    # Determine dtype size
                    dtype_size = 2  # default for bfloat16/float16
                    if dtype == torch.float32:
                        dtype_size = 4
                    
                    global_segment_size = \
                        MooncakeStoreConfig.calculate_global_segment_size(
                        max_num_seqs=max_num_seqs,
                        max_model_len=max_model_len,
                        hidden_size=hidden_size,
                        dtype_size=dtype_size,
                        eviction_factor=self.config.eviction_factor,
                    )
                else:
                    # Fall back to config file value
                    global_segment_size = self.config.global_segment_size
                    logger.info(
                        "Using global_segment_size from config: %d bytes (%.2f GiB)",
                        global_segment_size, global_segment_size / (1024**3)
                    )
                local_buffer_size = self.config.local_buffer_size
            else:
                global_segment_size = 0
                local_buffer_size = self.config.local_buffer_size
            
            # Ensure all parameters have valid string values (not None)
            protocol = self.config.protocol or "tcp"
            device_name = self.config.device_name or ""
            master_server = self.config.master_server_address or ""
            
            logger.info(
                "Initializing MooncakeDistributedStore with: "
                "global_segment_size=%d, local_buffer_size=%d, "
                "protocol=%s, device_name='%s', master_server='%s'",
                global_segment_size, local_buffer_size,
                protocol, device_name, master_server
            )
            
            # Create store after config validation
            self.store = MooncakeDistributedStore()
            
            # Setup with validated config - all params must be valid strings
            self.store.setup(
                str(self.config.local_hostname),
                str(self.config.metadata_server),
                int(global_segment_size),
                int(local_buffer_size),
                str(protocol),
                str(device_name),
                str(master_server),
            )
            self._initialized = True
            
        except Exception as e:
            logger.error("Mooncake configuration/initialization failed: %s", e)
            import traceback
            logger.error("Traceback: %s", traceback.format_exc())
            raise
        
        self.key_prefix = "accept_hidden_"
        
        # Fast transfer initialization (zero-copy via registered pinned memory)
        self.fast_transfer = os.getenv(
            "VLLM_HIDDEN_BUFFER_FAST_TRANSFER", "0"
        ).lower() in ("1", "true")
        
        if self.fast_transfer:
            from mooncake.store import ReplicateConfig
            
            from vllm.distributed.kv_transfer.kv_connector.v1.p2p.tensor_memory_pool import (  # noqa: E501
                TensorMemoryPool,
            )
            
            fast_buffer_size = int(os.getenv(
                "VLLM_HIDDEN_BUFFER_FAST_TRANSFER_SIZE",
                str(DEFAULT_FAST_TRANSFER_BUFFER_SIZE),
            ))
            self._tensor_pool = TensorMemoryPool(
                max_block_size=fast_buffer_size
            )
            self._pool_lock = threading.Lock()
            self.store.register_buffer(
                self._tensor_pool.base_address, fast_buffer_size
            )
            self._fast_buffer_size = fast_buffer_size
            self._replica_config = ReplicateConfig()
            self._replica_config.replica_num = 1
            logger.info(
                "MooncakeHiddenBuffer: fast_transfer enabled, "
                "buffer_size=%d bytes (%.2f GiB)",
                fast_buffer_size, fast_buffer_size / (1024**3),
            )
        
        logger.info("MooncakeHiddenBuffer initialized (is_creator=%s)", is_creator)
    
    def _make_key(self, request_id: str) -> str:
        return f"{self.key_prefix}{request_id}"
    
    def write(
        self,
        tensor: torch.Tensor,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> bool:
        """Write prefill hidden states to Mooncake store."""
        if not self._initialized or self.store is None:
            logger.error(
                "MooncakeHiddenBuffer.write: store not initialized"
            )
            return False
        
        if self.fast_transfer:
            return self._fast_write(tensor, request_id, timeout)
        
        from safetensors.torch import save as safetensors_save
        
        key = self._make_key(request_id)
        
        try:
            cpu_tensor = tensor.cpu()
            if self.dtype == torch.bfloat16:
                cpu_tensor = cpu_tensor.to(torch.float16)
            
            value_bytes = safetensors_save({"tensor": cpu_tensor})
            self.store.put(key, value_bytes)
            
            logger.debug(
                "MooncakeHiddenBuffer.write: request_id=%s, shape=%s",
                request_id, tensor.shape
            )
            return True
            
        except Exception as e:
            logger.error(
                "MooncakeHiddenBuffer.write FAILED: request_id=%s, error=%s",
                request_id, e
            )
            import traceback
            logger.error("Traceback: %s", traceback.format_exc())
            return False
    
    def _fast_write(
        self,
        tensor: torch.Tensor,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> bool:
        """Write hidden states using zero-copy transfer via registered pinned
        memory.  Data is written first, then metadata (as a 'ready' signal)."""
        import json
        
        key = self._make_key(request_id)
        meta_key = key + "_meta"
        
        try:
            # Ensure tensor is on CUDA for pinned-memory DMA
            if not tensor.is_cuda:
                tensor = tensor.cuda()
            
            # GPU → registered pinned memory (single fast DMA copy)
            with self._pool_lock:
                buffer_addr = self._tensor_pool.store_tensor(tensor)
            buffer_size = tensor.numel() * tensor.element_size()
            
            try:
                # Zero-copy put from registered memory (data first)
                self.store.batch_put_from(
                    [key], [buffer_addr], [buffer_size],
                    self._replica_config,
                )
                
                # Metadata stored separately (acts as "ready" signal)
                meta = json.dumps({
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }).encode("utf-8")
                self.store.put(meta_key, meta)
                
                logger.debug(
                    "MooncakeHiddenBuffer._fast_write: request_id=%s, "
                    "shape=%s, size=%d",
                    request_id, tensor.shape, buffer_size,
                )
                return True
            finally:
                with self._pool_lock:
                    self._tensor_pool.free(buffer_addr)
        
        except Exception as e:
            logger.error(
                "MooncakeHiddenBuffer._fast_write FAILED: "
                "request_id=%s, error=%s",
                request_id, e,
            )
            import traceback
            logger.error("Traceback: %s", traceback.format_exc())
            return False
    
    def read_by_request_id(
        self,
        request_id: str,
        device: str = "cuda",
        timeout: Optional[float] = None,
        wait: bool = True,
    ) -> Optional[torch.Tensor]:
        """Read prefill hidden states from Mooncake store."""
        if not self._initialized or self.store is None:
            logger.error(
                "MooncakeHiddenBuffer.read: store not initialized"
            )
            return None
        
        if self.fast_transfer:
            return self._fast_read_by_request_id(
                request_id, device, timeout, wait
            )
        
        from safetensors.torch import load as safetensors_load
        
        key = self._make_key(request_id)
        start_time = time.time()
        
        while True:
            try:
                data = self.store.get(key)
                
                if data:
                    loaded_tensors = safetensors_load(data)
                    tensor = loaded_tensors["tensor"]
                    
                    if self.dtype == torch.bfloat16:
                        tensor = tensor.to(torch.bfloat16)
                    
                    result = tensor.to(device)
                    
                    # Delete the key after reading to release shm
                    # Note: MooncakeDistributedStore may not have delete method,
                    # in which case we rely on eviction mechanism
                    try:
                        if hasattr(self.store, 'delete'):
                            self.store.delete(key)
                            logger.debug(
                                "MooncakeHiddenBuffer: deleted key %s after read",
                                key
                            )
                        else:
                            # No explicit delete, rely on eviction
                            logger.debug(
                                "MooncakeHiddenBuffer: no delete method, "
                                "relying on eviction for key %s", key
                            )
                    except Exception as e:
                        logger.warning(
                            "MooncakeHiddenBuffer: failed to delete key %s: %s",
                            key, e
                        )
                    
                    logger.debug(
                        "MooncakeHiddenBuffer.read: request_id=%s, shape=%s",
                        request_id, result.shape
                    )
                    return result
                    
            except Exception as e:
                logger.warning("Mooncake get failed for key %s: %s", key, e)
                import traceback
                logger.debug("Traceback: %s", traceback.format_exc())
            
            if not wait:
                return None
            
            if timeout is not None and (time.time() - start_time) > timeout:
                logger.warning(
                    "MooncakeHiddenBuffer.read TIMEOUT: request_id=%s",
                    request_id
                )
                return None
            
            time.sleep(0.001)
    
    def _fast_read_by_request_id(
        self,
        request_id: str,
        device: str = "cuda",
        timeout: Optional[float] = None,
        wait: bool = True,
    ) -> Optional[torch.Tensor]:
        """Read hidden states using zero-copy transfer via registered pinned
        memory.  Metadata is checked first as the 'ready' signal."""
        import json
        import math
        
        key = self._make_key(request_id)
        meta_key = key + "_meta"
        start_time = time.time()
        
        while True:
            try:
                # Metadata was written *after* data, so its presence
                # guarantees the data blob is already available.
                meta_bytes = self.store.get(meta_key)
                
                if meta_bytes:
                    meta = json.loads(meta_bytes.decode("utf-8"))
                    dtype = getattr(torch, meta["dtype"].split(".")[-1])
                    shape = tuple(meta["shape"])
                    element_size = torch.tensor(
                        [], dtype=dtype
                    ).element_size()
                    num_elem = math.prod(shape)
                    buffer_size = num_elem * element_size
                    
                    # Allocate from registered pinned-memory pool
                    with self._pool_lock:
                        buffer_addr = self._tensor_pool.allocate(buffer_size)
                    
                    try:
                        # Zero-copy get into registered memory
                        read_results = self.store.batch_get_into(
                            [key], [buffer_addr], [buffer_size],
                        )
                        
                        if read_results and read_results[0] > 0:
                            # Pinned memory → target device
                            result = self._tensor_pool.load_tensor(
                                buffer_addr, dtype, shape,
                                torch.device(device),
                            )
                            
                            # Best-effort cleanup of store keys
                            try:
                                if hasattr(self.store, 'delete'):
                                    self.store.delete(key)
                                    self.store.delete(meta_key)
                            except Exception:
                                pass
                            
                            logger.debug(
                                "MooncakeHiddenBuffer._fast_read: "
                                "request_id=%s, shape=%s",
                                request_id, result.shape,
                            )
                            return result
                    finally:
                        with self._pool_lock:
                            self._tensor_pool.free(buffer_addr)
            
            except Exception as e:
                logger.warning(
                    "Mooncake fast_read failed for %s: %s",
                    request_id, e,
                )
                import traceback
                logger.debug("Traceback: %s", traceback.format_exc())
            
            if not wait:
                return None
            
            if timeout is not None and (time.time() - start_time) > timeout:
                logger.warning(
                    "MooncakeHiddenBuffer._fast_read TIMEOUT: "
                    "request_id=%s",
                    request_id,
                )
                return None
            
            time.sleep(0.001)
    
    def close(self):
        """Close the Mooncake store.
        
        Note: MooncakeDistributedStore handles cleanup in its destructor,
        so we just mark as not initialized.
        """
        # Clean up fast-transfer resources first
        if self.fast_transfer and hasattr(self, '_tensor_pool'):
            try:
                self.store.unregister_buffer(
                    self._tensor_pool.base_address,
                    self._fast_buffer_size,
                )
                self._tensor_pool.cleanup()
                logger.debug(
                    "MooncakeHiddenBuffer: fast_transfer resources cleaned up"
                )
            except Exception as e:
                logger.debug(
                    "MooncakeHiddenBuffer: fast_transfer cleanup error: %s", e
                )
        
        self._initialized = False
        # Store cleanup is handled by MooncakeDistributedStore destructor
        logger.debug("MooncakeHiddenBuffer closed")


class SharedMemoryHiddenBuffer(HiddenBufferBackend):
    """
    SharedMemory-based backend using named shared memory for cross-process
    communication. Uses safetensors for tensor serialization.
    
    Each request creates a named shared memory segment that can be accessed
    by any process knowing the request_id.
    
    IMPORTANT: Writer creates shm and keeps it alive until reader reads and 
    unlinks it. Writer should NOT unlink on close() - only reader should unlink
    after successful read.
    """
    
    # Maximum length for shared memory name (POSIX limit)
    MAX_SHM_NAME_LEN = 250
    
    def __init__(
        self,
        dtype: torch.dtype = torch.bfloat16,
        is_creator: bool = False,
    ):
        from multiprocessing.shared_memory import SharedMemory
        self._SharedMemory = SharedMemory
        
        self.dtype = dtype
        self.is_creator = is_creator
        
        # Track created shared memory segments for cleanup on close
        # Note: We only close() these on exit, NOT unlink(), because
        # the reader process needs to be able to access and unlink them
        self._created_shm: dict[str, SharedMemory] = {}
        self._lock = threading.Lock()
        
        # Fast transfer initialization (pinned memory pool)
        self.fast_transfer = os.getenv(
            "VLLM_HIDDEN_BUFFER_FAST_TRANSFER", "0"
        ).lower() in ("1", "true")
        
        if self.fast_transfer:
            from vllm.distributed.kv_transfer.kv_connector.v1.p2p.tensor_memory_pool import (  # noqa: E501
                TensorMemoryPool,
            )
            
            fast_buffer_size = int(os.getenv(
                "VLLM_HIDDEN_BUFFER_FAST_TRANSFER_SIZE",
                str(DEFAULT_FAST_TRANSFER_BUFFER_SIZE),
            ))
            self._tensor_pool = TensorMemoryPool(
                max_block_size=fast_buffer_size,
            )
            self._pool_lock = threading.Lock()
            logger.info(
                "SharedMemoryHiddenBuffer: fast_transfer enabled, "
                "buffer_size=%d bytes (%.2f GiB)",
                fast_buffer_size, fast_buffer_size / (1024**3),
            )
        
        logger.info(
            "SharedMemoryHiddenBuffer initialized (is_creator=%s)",
            is_creator
        )
    
    def _make_shm_name(self, request_id: str) -> str:
        """Generate a valid shared memory name for a request.
        
        Shared memory names have platform-specific restrictions:
        - POSIX: must start with / and be <= 255 chars (we use 250 to be safe)
        - Windows: different restrictions
        
        We use a hash for long request_ids to ensure name length is valid.
        """
        import hashlib
        
        # Create a unique name based on request_id
        # If request_id is too long, hash it
        if len(request_id) > 50:
            hash_suffix = hashlib.md5(request_id.encode()).hexdigest()[:20]
            short_id = request_id[:30] + "_" + hash_suffix
        else:
            short_id = request_id
        
        # Replace invalid characters
        safe_id = short_id.replace("/", "_").replace("\\", "_")
        name = f"{SHM_NAME_PREFIX}{safe_id}"
        
        # Ensure name doesn't exceed limit
        if len(name) > self.MAX_SHM_NAME_LEN:
            name = name[:self.MAX_SHM_NAME_LEN]
        
        return name
    
    def write(
        self,
        tensor: torch.Tensor,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> bool:
        """Write tensor to named shared memory using safetensors serialization."""
        import struct
        
        if self.fast_transfer:
            # _fast_write handles both CUDA and CPU tensors.
            # CPU path: direct memmove from tensor.data_ptr() → SHM
            # (avoids lossy bfloat16→float16 conversion of the fallback).
            return self._fast_write(tensor, request_id, timeout)
        
        shm_name = self._make_shm_name(request_id)
        
        try:
            # Convert to CPU and handle bfloat16
            cpu_tensor = tensor.cpu().contiguous()
            original_dtype = cpu_tensor.dtype
            
            # bfloat16 needs special handling - convert to float16 for storage
            if original_dtype == torch.bfloat16:
                storage_tensor = cpu_tensor.to(torch.float16)
                dtype_code = 1  # bfloat16 indicator
            elif original_dtype == torch.float16:
                storage_tensor = cpu_tensor
                dtype_code = 2
            elif original_dtype == torch.float32:
                storage_tensor = cpu_tensor
                dtype_code = 3
            else:
                # Fallback: convert to float16
                storage_tensor = cpu_tensor.to(torch.float16)
                dtype_code = 2
            
            # Serialize tensor to bytes
            tensor_bytes = storage_tensor.numpy().tobytes()
            
            # Create header: 
            # - 4 bytes: magic number for validation (0xDEADBEEF)
            # - 1 byte: dtype_code
            # - 1 byte: ndim
            # - ndim * 8 bytes: shape (int64 per dimension)
            # - 8 bytes: data length
            shape = storage_tensor.shape
            ndim = len(shape)
            data_len = len(tensor_bytes)
            
            header_format = f'<IBBQ{ndim}q'  # Little-endian
            header = struct.pack(
                header_format,
                0xDEADBEEF,  # Magic number
                dtype_code,
                ndim,
                data_len,
                *shape
            )
            total_size = len(header) + data_len
            
            # Try to unlink any existing shm with same name (cleanup from crashed run)
            try:
                old_shm = self._SharedMemory(name=shm_name, create=False)
                old_shm.close()
                old_shm.unlink()
                logger.debug("Cleaned up stale shm: %s", shm_name)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.debug("Could not cleanup old shm %s: %s", shm_name, e)
            
            # Create new shared memory
            shm = self._SharedMemory(name=shm_name, create=True, size=total_size)
            
            # Write data
            shm.buf[:len(header)] = header
            shm.buf[len(header):total_size] = tensor_bytes
            
            # Track for cleanup (but don't close yet - reader needs it)
            with self._lock:
                self._created_shm[request_id] = shm
            
            logger.debug(
                "SharedMemoryHiddenBuffer.write: request_id=%s, shape=%s, "
                "dtype=%s, shm_name=%s, size=%d bytes",
                request_id, tensor.shape, original_dtype, shm_name, total_size
            )
            return True
            
        except Exception as e:
            logger.error(
                "SharedMemoryHiddenBuffer.write FAILED: request_id=%s, error=%s",
                request_id, e
            )
            import traceback
            logger.error("Traceback: %s", traceback.format_exc())
            return False
    
    def _fast_write(
        self,
        tensor: torch.Tensor,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> bool:
        """Write tensor to SHM in the fast-transfer binary format.
        
        Stores raw tensor bytes (no bfloat16→float16 conversion, lossless)
        with a FAST_TRANSFER_MAGIC header so the reader can distinguish the
        format.
        
        Supports both CUDA and CPU tensors:
        - CUDA: GPU → tensor_pool (pinned) → SHM  (DMA + memmove)
        - CPU:  tensor.data_ptr() → SHM directly   (single memmove)
        
        The CPU path avoids a GPU round-trip and is used by the V2
        hidden-states block buffer whose data already resides in CPU
        pinned memory.
        """
        import ctypes
        import struct
        
        shm_name = self._make_shm_name(request_id)
        is_cuda = tensor.is_cuda
        pool_addr: int | None = None
        
        try:
            # Ensure contiguous layout for direct memory access.
            tensor = tensor.contiguous()
            buffer_size = tensor.numel() * tensor.element_size()
            
            if is_cuda:
                # GPU → registered pinned memory (single fast DMA copy)
                with self._pool_lock:
                    pool_addr = self._tensor_pool.store_tensor(tensor)
                buffer_addr = pool_addr
            else:
                # CPU tensor – use data pointer directly (no copy needed).
                buffer_addr = tensor.data_ptr()
            
            try:
                # Build header (stores original dtype – no lossy conversion)
                original_dtype = tensor.dtype
                dtype_code = SharedHiddenBuffer.DTYPE_CODES.get(
                    original_dtype, 1
                )
                shape = tensor.shape
                ndim = len(shape)
                
                header_format = f'<IBBQ{ndim}q'
                header = struct.pack(
                    header_format,
                    FAST_TRANSFER_MAGIC,
                    dtype_code,
                    ndim,
                    buffer_size,
                    *shape,
                )
                total_size = len(header) + buffer_size
                
                # Cleanup stale SHM
                try:
                    old_shm = self._SharedMemory(
                        name=shm_name, create=False
                    )
                    old_shm.close()
                    old_shm.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    pass
                
                # Create SHM and write header + raw pinned data
                shm = self._SharedMemory(
                    name=shm_name, create=True, size=total_size,
                )
                shm.buf[:len(header)] = header
                
                # Efficient copy: pinned memory → SHM via ctypes memmove
                # NOTE: from_buffer() creates an "exported pointer" that
                # prevents shm.close() while alive, so we must del it
                # immediately after the copy.
                shm_ctypes = (ctypes.c_char * shm.size).from_buffer(
                    shm.buf
                )
                ctypes.memmove(
                    ctypes.addressof(shm_ctypes) + len(header),
                    buffer_addr,
                    buffer_size,
                )
                del shm_ctypes  # release exported pointer
                
                # Track SHM for cleanup
                with self._lock:
                    self._created_shm[request_id] = shm
                
                logger.debug(
                    "SharedMemoryHiddenBuffer._fast_write: "
                    "request_id=%s, shape=%s, dtype=%s, "
                    "shm_name=%s, size=%d bytes, src=%s",
                    request_id, tensor.shape, original_dtype,
                    shm_name, total_size,
                    "cuda" if is_cuda else "cpu",
                )
                return True
            finally:
                if pool_addr is not None:
                    with self._pool_lock:
                        self._tensor_pool.free(pool_addr)
        
        except Exception as e:
            logger.error(
                "SharedMemoryHiddenBuffer._fast_write FAILED: "
                "request_id=%s, error=%s",
                request_id, e,
            )
            import traceback
            logger.error("Traceback: %s", traceback.format_exc())
            return False
    
    def read_by_request_id(
        self,
        request_id: str,
        device: str = "cuda",
        timeout: Optional[float] = None,
        wait: bool = True,
    ) -> Optional[torch.Tensor]:
        """Read tensor from named shared memory."""
        import struct

        import numpy as np
        
        if self.fast_transfer:
            return self._fast_read_by_request_id(
                request_id, device, timeout, wait,
            )
        
        shm_name = self._make_shm_name(request_id)
        start_time = time.time()
        shm = None
        
        while True:
            try:
                # Try to attach to existing shared memory
                shm = self._SharedMemory(name=shm_name, create=False)
                
                # Read and validate header
                # First read magic + dtype + ndim
                basic_header = bytes(shm.buf[:10])
                magic, dtype_code, ndim = struct.unpack('<IBB', basic_header[:6])
                
                if magic != 0xDEADBEEF:
                    raise ValueError(f"Invalid magic number: {magic:#x}")
                
                # Read full header
                # magic + dtype + ndim + len + shape
                header_size = 4 + 1 + 1 + 8 + ndim * 8
                full_header = bytes(shm.buf[:header_size])
                
                header_format = f'<IBBQ{ndim}q'
                unpacked = struct.unpack(header_format, full_header)
                _, _, _, data_len = unpacked[:4]
                shape = unpacked[4:]
                
                # Map dtype code to numpy dtype
                dtype_map = {
                    1: np.float16,  # bfloat16 stored as float16
                    2: np.float16,
                    3: np.float32,
                }
                np_dtype = dtype_map.get(dtype_code, np.float16)
                
                # Read tensor data
                data_start = header_size
                data_end = data_start + data_len
                tensor_bytes = bytes(shm.buf[data_start:data_end])
                
                # Reconstruct tensor
                np_array = np.frombuffer(tensor_bytes, dtype=np_dtype).reshape(shape)
                tensor = torch.from_numpy(np_array.copy())
                
                # Convert back to original dtype
                if dtype_code == 1:  # Was bfloat16
                    tensor = tensor.to(torch.bfloat16)
                
                result = tensor.to(device)
                
                # Cleanup shared memory
                shm.close()
                try:
                    shm.unlink()
                    logger.debug(
                        "SharedMemoryHiddenBuffer: unlinked shm %s after read",
                        shm_name
                    )
                except Exception as e:
                    logger.debug(
                        "SharedMemoryHiddenBuffer: failed to unlink shm %s: %s",
                        shm_name, e
                    )
                
                logger.debug(
                    "SharedMemoryHiddenBuffer.read: request_id=%s, shape=%s, "
                    "device=%s",
                    request_id, result.shape, device
                )
                return result
                
            except FileNotFoundError:
                # Shared memory not created yet, will retry if wait=True
                pass
            except Exception as e:
                logger.warning(
                    "SharedMemoryHiddenBuffer.read error for %s: %s",
                    request_id, e
                )
                if shm is not None:
                    with contextlib.suppress(Exception):
                        shm.close()
                    shm = None
            
            if not wait:
                return None
            
            if timeout is not None and (time.time() - start_time) > timeout:
                logger.warning(
                    "SharedMemoryHiddenBuffer.read TIMEOUT: request_id=%s, "
                    "waited %.2f seconds",
                    request_id, time.time() - start_time
                )
                return None
            
            time.sleep(0.001)
    
    def _fast_read_by_request_id(
        self,
        request_id: str,
        device: str = "cuda",
        timeout: Optional[float] = None,
        wait: bool = True,
    ) -> Optional[torch.Tensor]:
        """Read tensor using pinned memory for fast CPU→GPU DMA transfer.
        
        Detects FAST_TRANSFER_MAGIC header and reads raw tensor bytes
        (no dtype conversion needed – data is stored losslessly).
        """
        import ctypes
        import struct
        
        shm_name = self._make_shm_name(request_id)
        start_time = time.time()
        shm = None
        is_cuda = device != "cpu"
        
        while True:
            try:
                shm = self._SharedMemory(name=shm_name, create=False)
                
                # Read and validate header
                basic_header = bytes(shm.buf[:6])
                magic, dtype_code, ndim = struct.unpack(
                    '<IBB', basic_header[:6]
                )
                
                if magic != FAST_TRANSFER_MAGIC:
                    raise ValueError(
                        f"Unexpected magic {magic:#x}, expected "
                        f"FAST_TRANSFER_MAGIC {FAST_TRANSFER_MAGIC:#x}"
                    )
                
                header_size = 4 + 1 + 1 + 8 + ndim * 8
                full_header = bytes(shm.buf[:header_size])
                
                header_format = f'<IBBQ{ndim}q'
                unpacked = struct.unpack(header_format, full_header)
                _, _, _, data_len = unpacked[:4]
                shape = tuple(unpacked[4:])
                
                # Resolve original dtype (stored losslessly)
                dtype = SharedHiddenBuffer.CODE_TO_DTYPE.get(
                    dtype_code, torch.bfloat16
                )
                
                if is_cuda:
                    # Allocate pinned buffer and copy SHM → pinned via memmove
                    with self._pool_lock:
                        buffer_addr = self._tensor_pool.allocate(data_len)
                else:
                    result = torch.empty(shape, dtype = dtype,device="cpu")
                    buffer_addr = result.data_ptr()

                try:
                    data_start = header_size
                    # NOTE: from_buffer() creates an "exported pointer"
                    # that prevents shm.close(). Must del before close.
                    shm_ctypes = (
                        ctypes.c_char * shm.size
                    ).from_buffer(shm.buf)
                    ctypes.memmove(
                        buffer_addr,
                        ctypes.addressof(shm_ctypes) + data_start,
                        data_len,
                    )
                    del shm_ctypes  # release exported pointer
                    
                    if is_cuda:
                        # Pinned memory → target device (fast DMA)
                        result = self._tensor_pool.load_tensor(
                            buffer_addr, dtype, shape,
                            torch.device(device),
                        )
                finally:
                    if is_cuda:
                        with self._pool_lock:
                            self._tensor_pool.free(buffer_addr)
                
                # Cleanup SHM (safe now – no exported pointers)
                shm.close()
                try:
                    shm.unlink()
                    logger.debug(
                        "SharedMemoryHiddenBuffer: "
                        "unlinked fast shm %s after read",
                        shm_name,
                    )
                except Exception as e:
                    logger.debug(
                        "SharedMemoryHiddenBuffer: "
                        "failed to unlink fast shm %s: %s",
                        shm_name, e,
                    )
                
                logger.debug(
                    "SharedMemoryHiddenBuffer._fast_read: "
                    "request_id=%s, shape=%s, device=%s",
                    request_id, result.shape, device,
                )
                return result
            
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(
                    "SharedMemoryHiddenBuffer._fast_read error "
                    "for %s: %s",
                    request_id, e,
                )
                if shm is not None:
                    with contextlib.suppress(Exception):
                        shm.close()
                    shm = None
            
            if not wait:
                return None
            
            if timeout is not None and (
                time.time() - start_time
            ) > timeout:
                logger.warning(
                    "SharedMemoryHiddenBuffer._fast_read TIMEOUT: "
                    "request_id=%s, waited %.2f seconds",
                    request_id, time.time() - start_time,
                )
                return None
            
            time.sleep(0.001)
    
    def close(self):
        """Close all created shared memory segments.
        
        IMPORTANT: We only close() the shm handles here, NOT unlink().
        The reader is responsible for unlinking after reading.
        This ensures the shm remains accessible to the reader even if
        the writer process exits first.
        """
        # Clean up fast-transfer resources
        if self.fast_transfer and hasattr(self, '_tensor_pool'):
            try:
                self._tensor_pool.cleanup()
                logger.debug(
                    "SharedMemoryHiddenBuffer: "
                    "fast_transfer pool cleaned up"
                )
            except Exception as e:
                logger.debug(
                    "SharedMemoryHiddenBuffer: "
                    "fast_transfer cleanup error: %s", e
                )
        
        with self._lock:
            for request_id, shm in list(self._created_shm.items()):
                try:
                    # Only close, do NOT unlink - reader will unlink after reading
                    shm.close()
                    logger.debug(
                        "SharedMemoryHiddenBuffer: closed shm for %s (not unlinked)",
                        request_id
                    )
                except Exception as e:
                    logger.debug(
                        "SharedMemoryHiddenBuffer: error closing shm for %s: %s",
                        request_id, e
                    )
            self._created_shm.clear()
    
    def cleanup_stale_shm(self, request_id: str):
        """Explicitly unlink a shared memory segment by request_id.
        
        This can be called by the main process to cleanup shm that was
        written but never read (e.g., due to request cancellation).
        """
        shm_name = self._make_shm_name(request_id)
        try:
            shm = self._SharedMemory(name=shm_name, create=False)
            shm.close()
            shm.unlink()
            logger.debug(
                "SharedMemoryHiddenBuffer: cleaned up stale shm %s", 
                shm_name
            )
        except FileNotFoundError:
            pass  # Already cleaned up
        except Exception as e:
            logger.debug(
                "SharedMemoryHiddenBuffer: error cleaning up stale shm %s: %s",
                shm_name, e
            )


class SharedHiddenBuffer:
    """
    High-level interface for prefill hidden states transfer.
    
    Supports:
    - SharedMemory backend (default): For single-node cross-process communication
    - Mooncake backend: For distributed/remote scenarios
    
    Backend selection via VLLM_HIDDEN_BUFFER_BACKEND or MOONCAKE_CONFIG_PATH.
    """

    _global_instance: Optional["SharedHiddenBuffer"] = None
    _lock = threading.Lock()
    
    DTYPE_CODES = {
        torch.bfloat16: 1,
        torch.float16: 2,
        torch.float32: 3,
    }
    CODE_TO_DTYPE = {v: k for k, v in DTYPE_CODES.items()}

    def __init__(
        self,
        backend_type: str = BACKEND_SHM,
        dtype: torch.dtype = torch.bfloat16,
        is_creator: bool = False,
        # Legacy params (ignored for simplified implementation)
        shm_name: Optional[str] = None,
        num_slots: int = 80,
        max_seq_len: int = 8192,
        hidden_dim: int = 3584,
        # Service configuration for dynamic shm sizing
        max_num_seqs: Optional[int] = None,
        max_model_len: Optional[int] = None,
        hidden_size: Optional[int] = None,
    ):
        self.backend_type = backend_type
        self.dtype = dtype
        
        if backend_type == BACKEND_MOONCAKE:
            self.backend: HiddenBufferBackend = MooncakeHiddenBuffer(
                dtype=dtype,
                is_creator=is_creator,
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                hidden_size=hidden_size,
            )
        else:
            self.backend = SharedMemoryHiddenBuffer(
                dtype=dtype,
                is_creator=is_creator,
            )
        
        logger.info("SharedHiddenBuffer initialized with backend: %s", backend_type)
    
    def write(
        self,
        gpu_tensor: torch.Tensor,
        request_id: str,
        timeout: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        """Write prefill hidden states for a request."""
        return self.backend.write(gpu_tensor, request_id, timeout)
    
    def read_by_request_id(
        self,
        request_id: str,
        device: str = "cuda",
        timeout: Optional[float] = None,
        wait: bool = True,
    ) -> Optional[torch.Tensor]:
        """Read prefill hidden states for a specific request_id."""
        return self.backend.read_by_request_id(request_id, device, timeout, wait)
    
    def close(self):
        """Close the backend."""
        self.backend.close()
    
    def cleanup_stale(self, request_id: str):
        """Cleanup stale shared memory for a request that was never read.
        
        This should be called when a request is aborted/cancelled to prevent
        shared memory leaks.
        """
        if hasattr(self.backend, 'cleanup_stale_shm'):
            self.backend.cleanup_stale_shm(request_id)
    
    def __del__(self):
        self.close()
    
    # ========== Global Instance Management ==========
    
    @classmethod
    def _get_backend_type(cls) -> str:
        """Get backend type from environment variable."""
        backend = os.environ.get("VLLM_HIDDEN_BUFFER_BACKEND")
        if backend:
            return backend
        return BACKEND_SHM
    
    @classmethod
    def create_global(
        cls,
        num_slots: int = 80,
        max_seq_len: int = 8192,
        hidden_dim: int = 3584,
        dtype: torch.dtype = torch.bfloat16,
        shm_name: Optional[str] = None,
        backend_type: Optional[str] = None,
        # Service configuration for dynamic shm sizing
        max_num_seqs: Optional[int] = None,
        max_model_len: Optional[int] = None,
        hidden_size: Optional[int] = None,
    ) -> "SharedHiddenBuffer":
        """Create global instance (main process).
        
        Args:
            num_slots: Legacy parameter (ignored)
            max_seq_len: Legacy parameter (ignored)
            hidden_dim: Legacy parameter (ignored)
            dtype: Data type for hidden states (default: bfloat16)
            shm_name: Legacy parameter (ignored)
            backend_type: Backend type ("shm" or "mooncake")
            max_num_seqs: Maximum number of concurrent requests (for shm sizing)
            max_model_len: Maximum context length (for shm sizing)
            hidden_size: Hidden dimension size of the model (for shm sizing)
        """
        if backend_type is None:
            backend_type = cls._get_backend_type()
        
        logger.info(
            "Creating global SharedHiddenBuffer: backend=%s, "
            "max_num_seqs=%s, max_model_len=%s, hidden_size=%s",
            backend_type, max_num_seqs, max_model_len, hidden_size
        )
        
        # When fast_transfer is enabled and service config is available,
        # dynamically calculate the optimal pool size and export it via
        # env var so that worker processes (connect_global) pick it up
        # automatically.
        fast_transfer = os.getenv(
            "VLLM_HIDDEN_BUFFER_FAST_TRANSFER", "0"
        ).lower() in ("1", "true")
        if (fast_transfer
                and max_model_len is not None
                and hidden_size is not None
                and not os.getenv("VLLM_HIDDEN_BUFFER_FAST_TRANSFER_SIZE")):
            dtype_size = 4 if dtype == torch.float32 else 2
            pool_size = calculate_fast_transfer_buffer_size(
                max_model_len=max_model_len,
                hidden_size=hidden_size,
                dtype_size=dtype_size,
            )
            os.environ["VLLM_HIDDEN_BUFFER_FAST_TRANSFER_SIZE"] = str(
                pool_size
            )
            logger.info(
                "Auto-calculated fast_transfer buffer size: %d bytes "
                "(%.2f GiB) and exported to env",
                pool_size, pool_size / (1024**3),
            )
        
        with cls._lock:
            cls._global_instance = cls(
                backend_type=backend_type,
                dtype=dtype,
                is_creator=True,
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                hidden_size=hidden_size,
            )
            os.environ["VLLM_HIDDEN_BUFFER_BACKEND"] = backend_type
            os.environ["VLLM_ACCEPT_HIDDEN_DTYPE"] = str(cls.DTYPE_CODES[dtype])
            
            return cls._global_instance
    
    @classmethod
    def connect_global(cls) -> Optional["SharedHiddenBuffer"]:
        """Connect to existing global instance (child process)."""
        backend_type = cls._get_backend_type()
        
        with cls._lock:
            if cls._global_instance is None:
                try:
                    dtype_code = int(os.environ.get("VLLM_ACCEPT_HIDDEN_DTYPE", 1))
                    cls._global_instance = cls(
                        backend_type=backend_type,
                        dtype=cls.CODE_TO_DTYPE[dtype_code],
                        is_creator=False,
                    )
                except Exception as e:
                    logger.warning("Failed to connect to shared buffer: %s", e)
                    return None
            return cls._global_instance
    
    @classmethod
    def get_global(cls) -> Optional["SharedHiddenBuffer"]:
        """Get global instance, connecting if needed."""
        if cls._global_instance is not None:
            return cls._global_instance
        return cls.connect_global()
    
    @classmethod
    def write_global(
        cls,
        gpu_tensor: torch.Tensor,
        request_id: str = "",
        timeout: Optional[float] = None,
        blocking: bool = True,
    ) -> bool:
        """Write to global instance (worker side)."""
        instance = cls.get_global()
        if instance is None:
            return False
        return instance.write(gpu_tensor, request_id, timeout, blocking)
    
    @classmethod
    def read_global_by_request_id(
        cls,
        request_id: str,
        device: str = "cuda",
        timeout: Optional[float] = None,
        wait: bool = True,
    ) -> Optional[torch.Tensor]:
        """Read from global instance by request_id (main process side)."""
        instance = cls.get_global()
        if instance is None:
            return None
        return instance.read_by_request_id(request_id, device, timeout, wait)
    
    @classmethod
    def clear_global(cls) -> None:
        """Clear global instance."""
        with cls._lock:
            if cls._global_instance is not None:
                cls._global_instance.close()
                cls._global_instance = None


# Backward compatibility alias
SharedHiddenRingBuffer = SharedHiddenBuffer
