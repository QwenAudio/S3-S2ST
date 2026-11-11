# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

from .epd_v6d_backend import EPDDisaggV6DReceiver, EPDDisaggV6DSender

if TYPE_CHECKING:
    from vllm.v1.request import Request

import vllm.envs as envs

logger = init_logger(__name__)


@dataclass
class ECVineyardConnectorMetadata(ECConnectorMetadata):
    ec_transfer_ref: dict[str, list[bytes]]


class ECVineyardConnector(ECConnectorBase):
    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)

        timeout_seconds = envs.VLLM_EPD_TRANSFER_TIMEOUT_SECONDS
        logger.info("VLLM_EPD_TRANSFER_TIMEOUT_SECONDS: %d", timeout_seconds)

        self._v6d_socket = vllm_config.ec_transfer_config.ec_v6d_socket

        if self._is_producer:
            self._transfer_client = EPDDisaggV6DSender(
                v6d_socket=self._v6d_socket,
                timeout_seconds=timeout_seconds,
            )
        else:
            self._transfer_client = EPDDisaggV6DReceiver(
                v6d_socket=self._v6d_socket,
                timeout_seconds=timeout_seconds,
            )

        self._mm_datas_need_loads: dict[str, list[bytes]] = {}

    def prepare_for_save(self, tensor_seqlen: int, tensor_dtype: torch.dtype, **kwargs):
        """
        For Encoder instance in API Server Process,
        prepare the tensor for saving to the v6d shm.

        Args:
            tensor_seqlen (int): The seqlen of the embedding tensor.
            tensor_dtype (torch.dtype): The dtype of the tensor.
            **kwargs (dict): Additional keyword arguments for the connector.

        """
        assert self.is_producer, "prepare_for_save must be called in Encoder instance!"

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
        ref_nbytes, obj = self._transfer_client.prepare_for_save(
            tensor_shape, tensor_dtype
        )
        return ref_nbytes, obj

    def prepare_for_load(self, ref_nbytes: dict[str, bytes], **kwargs):
        """
        For PD instance in API Server Process,
        prepare the tensor for loading from the v6d shm.

        Args:
            ref_nbytes (dict[str, bytes]): A dictionary mapping multimodal
                data hashes (`mm_hash`) to reference bytes.
                The keys are multimodal data hashes (`mm_hash`).
                The values are reference bytes.
            **kwargs (dict): Additional keyword arguments for the connector.
        """
        assert not self.is_producer, "prepare_for_load must be called in PD instance!"

        new_ref_nbytes, obj = self._transfer_client.prepare_for_load(ref_nbytes)
        return new_ref_nbytes, obj

    def save_caches(self, encoder_cache, mm_hash, **kwargs) -> None:
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

        # Get the metadata
        metadata: ECConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, ECVineyardConnectorMetadata)
        assert encoder_cache is not None
        if metadata is None:
            logger.warning(
                (
                    "In connector.save_caches, ",
                    "but the connector metadata is None",
                )
            )
            return

        ref_nbytes = metadata.ec_transfer_ref[mm_hash]
        embedding_tensor = encoder_cache[mm_hash]
        for ref_nbytes_ in ref_nbytes:
            self._transfer_client.save_caches(embedding_tensor, ref_nbytes_)
        logger.info(
            f"Save_caches: Saved Encoder Cache "
            f"for mm_hash: {mm_hash}"
        )

    def start_load_caches(self, encoder_cache, **kwargs) -> None:
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
        assert isinstance(metadata, ECVineyardConnectorMetadata)
        assert encoder_cache is not None
        if metadata is None:
            logger.warning(
                (
                    "In connector.start_load_caches, ",
                    "but the connector metadata is None",
                )
            )
            return

        for mm_hash, ref_nbytes in metadata.ec_transfer_ref.items():
            if mm_hash in encoder_cache:
                logger.info(
                    f"Start_load_caches: Hit Encoder Cache "
                    f"for mm_hash: {mm_hash}"
                )
                continue
            embedding_tensor = self._transfer_client.load_caches(ref_nbytes[0])
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

        For PD instance in API Server Process,
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
            or "ec_transfer_reference" not in request.sampling_params.extra_args
        ):
            return [False for _ in request.mm_features]
        else:
            return [True for _ in request.mm_features]

    def update_state_after_alloc(
        self,
        request: "Request",
        index: int,
    ):
        if (
            request.sampling_params is not None
            and request.sampling_params.extra_args is not None
            and "ec_transfer_reference" in request.sampling_params.extra_args
        ):
            current_ref = (
                request.sampling_params.extra_args["ec_transfer_reference"]
            )
            
            for key, value in current_ref.items():
                existing = self._mm_datas_need_loads.get(key)
                if existing is not None:
                    self._mm_datas_need_loads[key].append(value)
                else:
                    self._mm_datas_need_loads[key] = [value]

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ECConnectorMetadata:
        ec_transfer_ref = {}
        if not self.is_producer:
            ec_transfer_ref = self._mm_datas_need_loads.copy()
            self._mm_datas_need_loads.clear()
        else:
            for request in scheduler_output.scheduled_new_reqs:
                if (
                    request.sampling_params is not None
                    and request.sampling_params.extra_args is not None
                    and "ec_transfer_reference" in request.sampling_params.extra_args
                ):
                    current_ref = (
                        request.sampling_params.extra_args["ec_transfer_reference"]
                    )

                    for key, value in current_ref.items():
                        existing = ec_transfer_ref.get(key)
                        if existing is not None:
                            ec_transfer_ref[key].append(value)
                        else:
                            ec_transfer_ref[key] = [value]
                            
        meta = ECVineyardConnectorMetadata(ec_transfer_ref=ec_transfer_ref)
        return meta

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        return None, None
