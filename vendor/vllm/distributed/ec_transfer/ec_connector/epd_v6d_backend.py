import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import List, Tuple

import torch
import vineyard
import vineyard.contrib.ml.torch as vineyard_torch
from vineyard.core.object import VineyardObject, VineyardObjectRef

from vllm.logger import init_logger

logger = init_logger(__name__)

VINEYARD_RETRY_SECONDS = 0.01


class VineyardUnknownError(Exception):
    """Failed after multiple retries"""

    def __init__(
        self, message="Multiple retries of vineyard API failed", original_error=None
    ):
        self.original_error = original_error

        if original_error:
            message = f"{message}. Original error: {original_error}"

        super().__init__(message)


class VineyardConnectionError(Exception):
    """Vineyard Disconnect"""

    def __init__(self, message="Failed to connect to vineyard", original_error=None):
        self.original_error = original_error

        if original_error:
            message = f"{message}. Original error: {original_error}"

        super().__init__(message)


class EPDDisaggV6DSender:
    def __init__(
        self,
        v6d_socket: str,
        timeout_seconds: int,
    ) -> None:
        self._v6d_socket = v6d_socket
        self._sender_client = None
        self._timeout_seconds = timeout_seconds
        self._client_lock = threading.Lock()
    
    @property
    def sender_client(self):
        if self._sender_client is None:
            with self._client_lock:
                if self._sender_client is None:
                    self._sender_client = vineyard.connect(self._v6d_socket)
                    self._sender_client.timeout_seconds = self._timeout_seconds
        return self._sender_client

    def prepare_for_save(
        self, tensor_shape: list, tensor_dtype: torch.dtype
    ) -> Tuple[bytes, VineyardObject]:
        # prepare for save in API Server Process

        kw = vineyard_torch.construct_builtin_tensor_kw(tensor_shape, tensor_dtype, [])
        try:
            obj = self.sender_client.create_v6d_object(
                vineyard_torch.builtin_torch_tensor_builder,
                vineyard_torch.builtin_torch_tensor_resolver,
                **kw,
            )
        except Exception as e:
            for i in range(3):
                logger.warning(
                    f"Retrying {i + 1} times to create tensor object "
                    f"with sender_client. Error:{e} "
                )
                try:
                    obj = self.sender_client.create_v6d_object(
                        vineyard_torch.builtin_torch_tensor_builder,
                        vineyard_torch.builtin_torch_tensor_resolver,
                        **kw,
                    )
                    break
                except Exception as retry_err:
                    time.sleep(VINEYARD_RETRY_SECONDS)
                    if not self.sender_client.connected:
                        raise VineyardConnectionError(
                            original_error=retry_err
                        ) from retry_err
                    if i == 2:
                        raise VineyardUnknownError(
                            original_error=retry_err
                        ) from retry_err

        ref = obj.get_ref()
        ref_nbytes = VineyardObjectRef.serialize(ref)
        return ref_nbytes, obj

    def save_caches(self, embedding_tensor: torch.Tensor, ref_nbytes: bytes) -> None:
        # deserialize ref to unsafe attach vineyard object in Worker Process

        ref = VineyardObjectRef.deserialize(ref_nbytes)
        try:
            obj = self.sender_client.unsafe_attach_v6d_object(ref)
        except Exception as e:
            for i in range(3):
                logger.warning(
                    f"Retrying {i + 1} times to attach tensor object "
                    f"with sender_client. Error:{e} "
                )
                try:
                    obj = self.sender_client.unsafe_attach_v6d_object(ref)
                    break
                except Exception as retry_err:
                    time.sleep(VINEYARD_RETRY_SECONDS)
                    if not self.sender_client.connected:
                        raise VineyardConnectionError(
                            original_error=retry_err
                        ) from retry_err
                    if i == 2:
                        raise VineyardUnknownError(
                            original_error=retry_err
                        ) from retry_err

        # construct torch tensor from vineyard object
        torch_tensor = ref.resolver(obj)
        torch_tensor.copy_(embedding_tensor.detach().cpu())


class EPDDisaggV6DReceiver:
    def __init__(
        self,
        v6d_socket: str,
        timeout_seconds: int,
        max_workers: int = 32,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._v6d_socket = v6d_socket
        self._max_workers = max_workers

        self._base_client = None
        self._recv_client_pool = None
        self._base_client_lock = threading.Lock()
        self._recv_client_pool_lock = threading.Lock()

        self._local = threading.local()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    @property
    def base_client(self):
        if self._base_client is None:
            with self._base_client_lock:
                if self._base_client is None:
                    self._base_client = vineyard.connect(self._v6d_socket)
                    self._base_client.timeout_seconds = self._timeout_seconds
        return self._base_client
    
    @property
    def recv_client_pool(self):
        if self._recv_client_pool is None:
            with self._recv_client_pool_lock:
                if self._recv_client_pool is None:
                    self._recv_client_pool = queue.Queue(maxsize=self._max_workers)
                    for _ in range(self._max_workers):
                        temp_recv_client = self.base_client.fork()
                        temp_recv_client.timeout_seconds = self._timeout_seconds
                        self._recv_client_pool.put(temp_recv_client)
        return self._recv_client_pool

    def _thread_init(self) -> None:
        if not hasattr(self._local, "client"):
            self._local.client = self.recv_client_pool.get()

    def _single_prepare_for_load(
        self, ref_nbytes: bytes
    ) -> Tuple[bytes, VineyardObject]:
        self._thread_init()
        recv_client = self._local.client

        ref = VineyardObjectRef.deserialize(ref_nbytes)
        try:
            obj = recv_client.get_v6d_object(ref)
        except Exception as e:
            for i in range(3):
                logger.warning(
                    f"Retrying {i + 1} times to get tensor object "
                    f"with recv_client. Error:{e} "
                )
                try:
                    obj = recv_client.get_v6d_object(ref)
                    break
                except Exception as retry_err:
                    time.sleep(VINEYARD_RETRY_SECONDS)
                    if not recv_client.connected:
                        raise VineyardConnectionError(
                            original_error=retry_err
                        ) from retry_err
                    if i == 2:
                        raise VineyardUnknownError(
                            original_error=retry_err
                        ) from retry_err

        new_ref = obj.get_ref()
        new_ref_nbytes = VineyardObjectRef.serialize(new_ref)
        return new_ref_nbytes, obj

    def prepare_for_load(
        self, ref_nbytes_list: dict[str, bytes]
    ) -> Tuple[dict[str, bytes], List[VineyardObject]]:
        if not ref_nbytes_list:
            return {}, []

        futures = []
        mm_hashes = []
        for mm_hash, ref_nbytes in ref_nbytes_list.items():
            futures.append(
                self._executor.submit(self._single_prepare_for_load, ref_nbytes)
            )
            mm_hashes.append(mm_hash)

        results = [f.result() for f in futures]

        new_ref_nbytes_list_, obj_list = zip(*results)
        new_ref_nbytes_list = dict(zip(mm_hashes, list(new_ref_nbytes_list_)))
        return new_ref_nbytes_list, list(obj_list)

    def load_caches(self, ref_nbytes: bytes) -> torch.Tensor:
        ref = VineyardObjectRef.deserialize(ref_nbytes)
        try:
            obj = self.base_client.unsafe_attach_v6d_object(ref)
        except Exception as e:
            for i in range(3):
                logger.warning(
                    f"Retrying {i + 1} times to attach tensor object "
                    f"with recv_client. Error:{e} "
                )
                try:
                    obj = self.base_client.unsafe_attach_v6d_object(ref)
                    break
                except Exception as retry_err:
                    time.sleep(VINEYARD_RETRY_SECONDS)
                    if not self.base_client.connected:
                        raise VineyardConnectionError(
                            original_error=retry_err
                        ) from retry_err
                    if i == 2:
                        raise VineyardUnknownError(
                            original_error=retry_err
                        ) from retry_err

        torch_tensor = ref.resolver(obj).detach().clone().cuda()
        return torch_tensor
