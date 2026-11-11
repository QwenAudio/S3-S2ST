# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

import vllm.envs as envs
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

from .epd_global_v6d_backend import GlobalVineyardBackend, MultiModalMetadata

if TYPE_CHECKING:
    from v6d.client.client import Object
    
    from vllm.config import VllmConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class ECGlobalVineyardConnectorMetadata(ECConnectorMetadata):
    mm_hashes: list[str]


class ECGlobalVineyardConnector(ECConnectorBase):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: ECConnectorRole,
    ) -> None:
        super().__init__(vllm_config=vllm_config, role=role)

        timeout_seconds = envs.VLLM_EPD_TRANSFER_TIMEOUT_SECONDS
        logger.info("VLLM_EPD_TRANSFER_TIMEOUT_SECONDS: %d", timeout_seconds)

        self._v6d_socket = vllm_config.ec_transfer_config.ec_v6d_socket

        self._transfer_client = GlobalVineyardBackend(
            v6d_socket=self._v6d_socket,
            timeout_seconds=timeout_seconds,
        )

        self._mm_datas_need_loads: list[str] = []

    def get_peer_address(self) -> str:
        """
        Get the peer address (IP:port) of the local Vineyard instance.
        """
        return self._transfer_client.get_peer_address()
    
    def prepare_for_save_embd(
        self,
        mm_hash: str,
        fallback_mm_hash: str,
        tensor_seqlen: int,
        tensor_dtype: torch.dtype,
        **kwargs,
    ) -> tuple["Object", bool]:
        """
        For Encoder instance in API Server Process,
        prepare the tensor for saving to the v6d shm.

        Args:
            mm_hash (str): The hash of the multimodal data.
            tensor_seqlen (int): The seqlen of the embedding tensor.
            tensor_dtype (torch.dtype): The dtype of the tensor.
            **kwargs (dict): Additional keyword arguments for the connector.

        """
        assert self.is_producer, (
            "prepare_for_save_embd must be called in Encoder instance!"
        )

        hf_config = self._vllm_config.model_config.hf_config
        out_hidden_size = hf_config.vision_config.out_hidden_size
        num_layer_of_deepstack = 1
        if {"Qwen3_VL_ViT", "Qwen3_VL_MOE_ViT"} & set(
            hf_config.architectures
        ):
            vision_config = hf_config.vision_config
            if (
                hasattr(vision_config, "deepstack_visual_multiscale_indexes")
                and vision_config.deepstack_visual_multiscale_indexes is not None
            ):
                indexes = vision_config.deepstack_visual_multiscale_indexes
            elif (
                hasattr(vision_config, "deepstack_visual_indexes")
                and vision_config.deepstack_visual_indexes is not None
            ):
                indexes = vision_config.deepstack_visual_indexes
            else:
                raise AttributeError(
                    "Vision config must have either "
                    " 'deepstack_visual_multiscale_indexes' "
                    "or 'deepstack_visual_indexes' attribute."
                )
            num_layer_of_deepstack = len(indexes) + 1
        tensor_shape = [tensor_seqlen, out_hidden_size * num_layer_of_deepstack]
        obj, used_mm_hash, cache_hit = self._transfer_client.prepare_for_save_embd(
            mm_hash, fallback_mm_hash, tensor_shape, tensor_dtype
        )
        return obj, used_mm_hash, cache_hit
    
    def save_meta(
        self,
        mm_hash: str,
        metadata: MultiModalMetadata,
        **kwargs,
    ) -> "Object":
        """
        For Encoder instance in API Server Process,
        save metadata of the multimodal data to the v6d shm.

        Args:
            mm_hash (str): The hash of the multimodal data.
            metadata (dict): The metadata of the multimodal data.
            **kwargs (dict): Additional keyword arguments for the connector.
        """
        assert self.is_producer, "save_meta must be called in Encoder instance!"

        obj = self._transfer_client.save_meta(mm_hash, metadata)
        return obj

    def prepare_for_load_embd(
        self,
        mm_hashes: list[str],
        ip_port: Optional[str] = None,
        **kwargs,
    ) -> dict[str, "Object"]:
        """
        For PD instance in API Server Process,
        prepare the tensor for loading from the v6d shm.

        Args:
            mm_hashes (list[str]): List of multimodal data hashes.
            ip_port (str): The IP:port of the remote Vineyard instance.
            **kwargs (Any): Additional keyword arguments.
        """
        all_obj = self._transfer_client.prepare_for_load_embd(mm_hashes, ip_port)
        return all_obj
    
    def load_metadata(
        self,
        mm_hashes: list[str],
        ip_port: Optional[str] = None,
        is_pd_node: Optional[bool] = False,
        **kwargs,
    ) -> dict[str, MultiModalMetadata]:
        """
        For Encoder instance in API Server Process,
        load metadata for given multimodal hashes from the Vineyard backend..

        Args:
            mm_hashes (list[str]): List of multimodal data hashes.
            ip_port (str): The IP:port of the remote Vineyard instance.
            **kwargs (Any): Additional keyword arguments.
        """

        all_metadata = self._transfer_client.load_metadata(
            mm_hashes, ip_port, is_pd_node
        )
        return all_metadata

    def save_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
        **kwargs,
    ) -> None:
        """
        For Encoder instance in Worker Process,
        save the embeds to the v6d shm.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping multimodal
                data hashes (`mm_hash`) to encoder cache tensors.
            mm_hash (str): The hash of the multimodal data whose cache is being saved.
            kwargs (dict): Additional keyword arguments for the connector.
        """
        # Check if it is Encoder Instance
        if not self.is_producer:
            return

        assert encoder_cache is not None

        embedding_tensor = encoder_cache[mm_hash]
        self._transfer_client.save_caches(mm_hash, embedding_tensor)
        logger.info(
            f"Save_caches: Saved Encoder Cache "
            f"for mm_hash: {mm_hash}"
        )
            
    def start_load_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        **kwargs,
    ) -> None:
        """
        Start loading the cache from the connector into vLLM's encoder cache.

        This method loads the encoder cache based on metadata provided by the scheduler.
        It is called before `_gather_mm_embeddings` for the EC Connector. For EC,
        the `encoder_cache` and `mm_hash` are stored in `kwargs`.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping multimodal
                data hashes (`mm_hash`) to encoder cache tensors.
            kwargs (dict): Additional keyword arguments for the connector.
        """
        # Check if it is Encoder Instance
        assert not self.is_producer, "start_load_caches must be called in PD instance!"

        # Get the metadata
        metadata: ECConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, ECGlobalVineyardConnectorMetadata)
        assert encoder_cache is not None
        if metadata is None:
            logger.warning(
                (
                    "In connector.start_load_caches, ",
                    "but the connector metadata is None",
                )
            )
            return

        for mm_hash in metadata.mm_hashes:
            if mm_hash in encoder_cache:
                logger.info(
                    f"Start_load_caches: Hit Encoder Cache "
                    f"for mm_hash: {mm_hash}"
                )
                continue
            embedding_tensor = self._transfer_client.load_caches(mm_hash)
            encoder_cache[mm_hash] = embedding_tensor
            logger.info(
                f"Start_load_caches: Loaded Encoder Cache "
                f"for mm_hash: {mm_hash}"
            )

    def has_caches(
        self,
        request: "Request",
    ) -> list[bool]:
        """
        For Encoder instance in API Server Process,
        check if EC has caches(in remote) for each mm in req 
        and always return False now.

        For PD instance in API Server Process,git
        check if EC has caches(in remote) for each mm in req and,
        returns True if request need remote mm xfer,
        returns False if need fallback to local mm encode.

        Args:
            request (Request): the request object.

        Returns:
            List of bool indicate that ith mm_data exist in cache or not
        """
        if (
            self.is_producer
            or request.sampling_params is None
            or request.sampling_params.extra_args is None
            or "enable_disaggregated_vit" not in request.sampling_params.extra_args
        ):
            return [False for _ in request.mm_features]
        else:
            return [True for _ in request.mm_features]

    def update_state_after_alloc(
        self,
        request: "Request",
        index: int,
    ) -> None:
        mm_hash = request.mm_features[index].identifier
        if mm_hash not in self._mm_datas_need_loads:
            self._mm_datas_need_loads.append(mm_hash)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ECConnectorMetadata:
        mm_hashes = []
        if not self.is_producer:
            mm_hashes = self._mm_datas_need_loads.copy()
            self._mm_datas_need_loads.clear()
        else:
            for request in scheduler_output.scheduled_new_reqs:
                for mm_feature in request.mm_features:
                    mm_hash = mm_feature.identifier
                    if mm_hash not in mm_hashes:
                        mm_hashes.append(mm_hash)
                            
        meta = ECGlobalVineyardConnectorMetadata(mm_hashes=mm_hashes)
        return meta

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        return None, None
