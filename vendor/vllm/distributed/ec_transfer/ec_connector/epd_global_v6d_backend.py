import math
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx
import torch
import v6d
from v6d import ObjectAlreadyExistError, ObjectBusyError, ObjectFetchingError, RPCError
from v6d.data.tensor import Tensor

from vllm.logger import init_logger

logger = init_logger(__name__)


class VineyardMultiModalKey:
    """Helper class to generate consistent vineyard keys from mm_hash."""

    SUFFIX_META = '_meta'
    SUFFIX_EMBD = '_embd'

    def __init__(self, mm_hash: str):
        self._mm_hash = mm_hash

    @property
    def meta(self) -> str:
        """Key for metadata object."""
        return self._mm_hash + self.SUFFIX_META

    @property
    def embd(self) -> str:
        """Key for embedding tensor object."""
        return self._mm_hash + self.SUFFIX_EMBD

    @property
    def raw(self) -> str:
        """Original mm_hash value."""
        return self._mm_hash


class VineyardUnknownError(Exception):
    """Vineyard Unkown Error"""

    def __init__(
        self, message="VineyardUnknownError", original_error=None
    ):
        self.original_error = original_error

        if original_error:
            message = f"{message}: {original_error}"

        super().__init__(message)


@dataclass
class MultiModalMetadata:
    """Metadata for multimodal features stored in vineyard."""
    grid_thw: torch.Tensor  # Grid dimensions (time, height, width)
    video_metadata: dict | None  # Video-specific metadata, None for images

def wait_for_server(url, timeout=15):
    """等待服务就绪"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(url, timeout=1.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.2)
    return False

class GlobalVineyardBackend:
    def __init__(
        self,
        v6d_socket: str,
        timeout_seconds: int,
        max_workers: int = 32,
    ):
        self._timeout_seconds = timeout_seconds
        self._v6d_socket = v6d_socket
        self._max_workers = max_workers

        self._base_client = None
        self._base_client_lock = threading.Lock()
        
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    @property
    def base_client(self):
        if self._base_client is None:
            with self._base_client_lock:
                if self._base_client is None:
                    self.vl_v6d_port = os.getenv("DS_VL_V6D_PORT", "11000")
                    if wait_for_server(f"http://localhost:{self.vl_v6d_port}/health"):
                        try:
                            self._base_client = v6d.connect(self._v6d_socket)
                        except RPCError:
                            self._base_client = v6d.connect(self._v6d_socket)
                        print(
                            "Connected to v6d successfully for global encoder cache!",
                            flush=True
                        )
                    else:
                        logger.error("connect V6D server timeout")
        return self._base_client

    def prepare_for_save_embd(
        self,
        mm_hash: str,
        fallback_mm_hash: str,
        tensor_shape: list[int],
        tensor_dtype: torch.dtype,
    ):
        # prepare for save embed tensor in API Server Process

        used_mm_hash = mm_hash
        cache_hit = False
        key = VineyardMultiModalKey(mm_hash)
        try:
            obj = self.base_client.get(key.embd)
            if obj is None:
                num_elements = int(math.prod(tensor_shape))
                nbytes = num_elements * torch.finfo(tensor_dtype).bits // 8
                obj = self.base_client.create(
                    keys=[key.embd],
                    sizes=[nbytes],
                    metas=[{}],
                    object_types=[Tensor.type(shape=tensor_shape, dtype=tensor_dtype)],
                )[0]
            else:
                _ = obj.buffer
                cache_hit = True
        except ObjectFetchingError:
            obj = self.base_client.get(key.embd, wait_timeout=10)
            cache_hit = True
        except (ObjectAlreadyExistError, ObjectBusyError):
            logger.info(
                f"{key.embd} has already exists, create this obj by request_id"
            )
            used_mm_hash = fallback_mm_hash
            fallback_key = VineyardMultiModalKey(fallback_mm_hash)
            num_elements = int(math.prod(tensor_shape))
            nbytes = num_elements * torch.finfo(tensor_dtype).bits // 8
            obj = self.base_client.create(
                keys=[fallback_key.embd],
                sizes=[nbytes],
                metas=[{}],
                object_types=[Tensor.type(shape=tensor_shape, dtype=tensor_dtype)],
            )[0]
        except Exception as e:
            raise VineyardUnknownError(original_error=e) from e

        return obj, used_mm_hash, cache_hit

    def save_meta(self, mm_hash: str, metadata: MultiModalMetadata):
        # save metadata in API Server Process

        key = VineyardMultiModalKey(mm_hash)
        try:
            bytes_of_metadata = pickle.dumps(metadata)
            obj = self.base_client.create(
                keys=[key.meta],
                sizes=[len(bytes_of_metadata)],
                metas=[{}],
            )[0]
            obj.write(bytes_of_metadata)
            obj.seal()
        except ObjectAlreadyExistError:
            obj = self.base_client.get(key.meta, wait_timeout=10)
            _ = obj.buffer
        except Exception as e:
            raise VineyardUnknownError(original_error=e) from e

        return obj

    def save_caches(
        self,
        mm_hash: str,
        embedding_tensor: torch.Tensor,
    ):
        key = VineyardMultiModalKey(mm_hash)
        with self.base_client.get(key.embd, peer="local", unsafe=True) as obj:
            write_torch_tensor = obj.resolver()
            write_torch_tensor.copy_(embedding_tensor.detach().cpu())

    def prepare_for_load_embd(
        self, mm_hashes: list[str], ip_port: str
    ):
        if not mm_hashes:
            return {}

        def _single_prepare_for_load_embd(
            mm_hash: str, ip_port: str | None
        ):
            key = VineyardMultiModalKey(mm_hash)

            try:
                obj = self.base_client.get(key.embd, peer=ip_port)
                _ = obj.buffer 
            except ObjectFetchingError:
                obj = self.base_client.get(key.embd, wait_timeout=10)
            except Exception as e:
                raise VineyardUnknownError(original_error=e) from e

            return obj

        futures = []
        for mm_hash in mm_hashes:
            futures.append(
                self._executor.submit(
                    _single_prepare_for_load_embd, mm_hash, ip_port
                )
            )
        results = [f.result() for f in futures]

        all_obj = dict(zip(mm_hashes, results))
        return all_obj

    def load_caches(self, mm_hash: str):
        key = VineyardMultiModalKey(mm_hash)
        with self.base_client.get(key.embd, peer="local") as obj:
            torch_tensor: torch.Tensor = obj.resolver().detach().clone().cuda()

        return torch_tensor

    def load_metadata(
        self, mm_hashes: list[str], ip_port: str | None, is_pd_node: bool
    ) -> dict[str, MultiModalMetadata]:
        if not mm_hashes:
            return {}

        def _single_load_metadata(
            mm_hash: str, ip_port: str | None, is_pd_node: bool
        ) -> MultiModalMetadata:
            key = VineyardMultiModalKey(mm_hash)

            try:
                with self.base_client.get(key.meta, peer=ip_port) as obj:
                    bytes_of_metadata = bytes(obj.buffer)
                    if is_pd_node:
                        logger.info(f"Holding metadata for {mm_hash}")
                        return {}
                    metadata = pickle.loads(bytes_of_metadata)
                    logger.info(f"Loading metadata for {mm_hash}, metadata: {metadata}")
            except ObjectFetchingError:
                if is_pd_node:
                    return {}
                obj = self.base_client.get(key.meta, wait_timeout=10)
                bytes_of_metadata = bytes(obj.buffer)
                metadata = pickle.loads(bytes_of_metadata)
                logger.info(f"Loading metadata for {mm_hash}, metadata: {metadata}")
            except Exception as e:
                raise VineyardUnknownError(original_error=e) from e

            return metadata
    
        futures = []
        for mm_hash in mm_hashes:
            futures.append(
                self._executor.submit(
                    _single_load_metadata, mm_hash, ip_port, is_pd_node
                )
            )
        results = [f.result() for f in futures]

        all_metadata = dict(zip(mm_hashes, results))
        return all_metadata

    def get_peer_address(self):
        return self.base_client.peer_address
