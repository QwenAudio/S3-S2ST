from typing import Iterable, Optional, Set

import numpy as np
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerConfig,
)

from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
)
from vllm.v1.sample.metadata import SubtalkerSamplingMetadata

logger = init_logger(__name__)


class CodePredictor:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        config: Qwen3OmniMoeTalkerConfig = vllm_config.model_config.hf_config
        if hasattr(config, "talker_config"):
            self.config = config.talker_config
        else:
            self.config = config
        self.code_predictor_config = self.config.code_predictor_config

        self.device = device
        self.dtype = vllm_config.model_config.dtype
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.token_arange_np = np.arange(self.max_num_tokens)
        self.hidden_size = self.code_predictor_config.hidden_size

        compilation_config = self.vllm_config.compilation_config
        self.cudagraph_mode = self.vllm_config.compilation_config.cudagraph_mode
        self.enforce_eager = True
        if compilation_config.mode == CompilationMode.VLLM_COMPILE:
            self.enforce_eager = self.vllm_config.model_config.enforce_eager
        self.cudagraph_batch_sizes = sorted(
            self.vllm_config.compilation_config.cudagraph_capture_sizes
        )

        # persistent buffers for cuda graph
        self.use_cuda_graph = bool(self.cudagraph_batch_sizes)

        # We need +1 here because the arange is used to set query_start_loc,
        # which has one more element than batch_size.
        self.max_batch_size = vllm_config.scheduler_config.max_num_seqs

        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, 2, self.hidden_size), dtype=self.dtype, device=device
        )

        self.sampling_tensors = SubtalkerSamplingMetadata(
            temperature=torch.full(
                [self.max_num_tokens], 0.9, dtype=self.dtype, device=self.device
            ),
            top_p=torch.full(
                [self.max_num_tokens], 1.0, dtype=self.dtype, device=self.device
            ),
            top_k=torch.full(
                [self.max_num_tokens], 50, dtype=torch.int64, device=self.device
            ),
        )
        self.rand_buf = torch.rand(
            [self.max_num_tokens, 15, self.code_predictor_config.vocab_size],
            dtype=self.dtype,
            device=self.device,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Set[str]:
        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={
                "talker.code_predictor": "code_predictor",
            }
        )

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=[
                "thinker.",
                "token2wav.",
                "talker.hidden_projection",
                "talker.text_projection",
                "talker.model",
                "talker.codec_head",
            ],
        )
        return loader.load_weights(weights, mapper=hf_to_vllm_mapper)

    def load_model(self, model) -> None:
        self.model = model

        if not self.enforce_eager and self.cudagraph_mode.has_full_cudagraphs():
            logger.info("Using cuda graph for code predictor")
            self.model.generate_fix_times = CUDAGraphWrapper(
                self.model.generate_fix_times,
                self.vllm_config,
                runtime_mode=CUDAGraphMode.FULL,
            )

    def propose(
        self,
        inputs_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = inputs_embeds.shape[0]

        cudagraph_runtime_mode = self.cudagraph_mode.decode_mode()
        use_cuda_graph = (
            cudagraph_runtime_mode != CUDAGraphMode.NONE and not self.enforce_eager
        )

        if use_cuda_graph and num_tokens <= self.cudagraph_batch_sizes[-1]:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)
        else:
            cudagraph_runtime_mode = CUDAGraphMode.NONE
            num_input_tokens = num_tokens
        # copy inputs to buffer for cudagraph
        self.inputs_embeds[:num_tokens] = inputs_embeds
        inputs_embeds_pad = self.inputs_embeds[:num_input_tokens]
        self.rand_buf[:num_tokens].exponential_()
        sampling_tensors = self.get_sampling_tensors_prefix(num_input_tokens)
        rand_buf = self.rand_buf[:num_input_tokens]

        with (
            set_current_vllm_config(self.vllm_config),
            set_forward_context(
                None,
                self.vllm_config,
                num_tokens=num_input_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                is_prompt_batch=False,  # the subsequent draft steps are decoding
            ),
        ):
            subtalker_codec, subtalker_hidden_states = self.model.generate_fix_times(
                inputs_embeds=inputs_embeds_pad,
                sampling_tensors=sampling_tensors,
                rand_buf=rand_buf,
            )
            subtalker_codec = subtalker_codec[:num_tokens]
            subtalker_hidden_states = subtalker_hidden_states[:num_tokens]

        return subtalker_codec, subtalker_hidden_states

    def prepare_token_ids(
        self, input_token_ids: list[list[int]]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        values = []
        valid = []
        for token_ids in input_token_ids:
            if (
                len(token_ids) == 0
                or token_ids[0] >= self.code_predictor_config.vocab_size
            ):
                valid.append(0)
            else:
                values.append(token_ids[0])
                valid.append(1)
        if len(values) == 0:
            return None, None
        token_ids = torch.tensor(values, device=self.device)
        valid_tensor = torch.tensor(valid, device=self.device)
        decode_indices = valid_tensor.nonzero(as_tuple=True)[0]

        return token_ids, decode_indices

    def post_process_hidden_states(
        self,
        text_embeddings: torch.Tensor,
        subtalker_codec: torch.Tensor,
        subtalker_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        codec_hiddens = torch.cat(
            [text_embeddings]
            + [subtalker_hidden_states]
            + [self._get_code_predictor_last_embeddings(subtalker_codec[..., -1:])],
            dim=1,
        )
        subtalker_hidden_states = codec_hiddens.sum(1)
        return subtalker_hidden_states

    def _get_code_predictor_last_embeddings(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.model.codec_embedding[-1](input_ids)

    def get_sampling_tensors_prefix(
        self,
        num_tokens: int
    ) -> SubtalkerSamplingMetadata:
        st = self.sampling_tensors
        return SubtalkerSamplingMetadata(
            temperature=st.temperature[:num_tokens],
            top_p=st.top_p[:num_tokens],
            top_k=st.top_k[:num_tokens],
        )

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        inputs_embeds: Optional[torch.Tensor] = None,
        is_profile: bool = False,
    ) -> None:
        # Determine if CUDA graphs should be used for this run.
        if use_cudagraphs and num_tokens <= self.cudagraph_batch_sizes[-1]:
            num_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)
        if is_profile:
            num_tokens = 1
        cudagraph_runtime_mode = (
            (self.cudagraph_mode.decode_mode())
            if use_cudagraphs
            else CUDAGraphMode.NONE
        )

        inputs_embeds = self.inputs_embeds[:num_tokens]
        sampling_tensors = self.get_sampling_tensors_prefix(num_tokens)
        rand_buf = self.rand_buf[:num_tokens]

        with (
            set_current_vllm_config(self.vllm_config),
            set_forward_context(
                None,
                self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                is_prompt_batch=False,  # the subsequent draft steps are decoding
            ),
        ):
            self.model.generate_fix_times(
                inputs_embeds=inputs_embeds,
                sampling_tensors=sampling_tensors,
                rand_buf=rand_buf,
            )
