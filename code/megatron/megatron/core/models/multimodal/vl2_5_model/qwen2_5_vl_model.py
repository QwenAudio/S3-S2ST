# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from megatron.core import InferenceParams
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings.qwenvl_embedding import (
    QwenVLLanguageModelEmbedding,
)
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.multimodal.rotary_pos_embedding import Qwen2VLRotaryEmbedding
from megatron.core.models.multimodal.vl2_5_model.vision_transformer_block import (
    VisionTransformerBlock,
)
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import (
    TransformerConfig,
    VLVisionConfig,
)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )
        return hidden_states


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs.float()


class Qwen2_5_VLVisionModel(VisionModule):
    """Qwen2 ViT vision model, adapted from CLIPViTModel to support Naive Dynamic Resolution.

    Args:
        vl_config (VLVisionConfig): Vision model configuration.
        vision_model_spec (ModuleSpec): Specifies module to use for vision model.
        projection_config (TransformerConfig): Projection layer configuration.
        projection_layer_spec (ModuleSpec): Specifies module to use for projection layer.
        pre_process (bool, optional): Whether to pre-process the input. Defaults to True.
        post_process (bool, optional): Whether to post-process the output. Defaults to True.
    """

    def __init__(
        self,
        vl_config: VLVisionConfig,
        vision_model_spec: ModuleSpec,
        projection_config: TransformerConfig,
        projection_layer_spec: ModuleSpec,
        pre_process: bool = True,
        post_process: bool = True,
    ) -> None:
        super().__init__(config=vl_config)

        self.spatial_merge_size = vl_config.spatial_merge_size
        self.window_size = vl_config.vision_window_size
        self.num_layers = vl_config.num_layers
        self.fullatt_block_indexes = vl_config.fullatt_block_indexes

        embed_dim = vl_config.hidden_size
        num_heads = vl_config.num_attention_heads
        temporal_patch_size = vl_config.temporal_patch_size
        patch_size = vl_config.spatial_patch_size
        in_channels = vl_config.in_channels

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        head_dim = embed_dim // num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.model_type = ModelType.encoder_or_decoder
        self.pre_process = pre_process
        self.post_process = post_process

        # Transformer layers.
        # TODO: Follow-up changes will make pre and post_process configurable. They are needed for supporting pipeline parallelism.
        # NOTE: a final layer norm and/or linear layer present in some implementations are omitted here.
        self.decoder = VisionTransformerBlock(
            config=vl_config,
            spec=vision_model_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=True,
        )

        self.merge_hidden_size = projection_config.ffn_hidden_size
        self.square_merge_size = self.merge_hidden_size // embed_dim

        if self.post_process:
            projection_type = "mlp"
            self.projection = MultimodalProjector(
                projection_config,
                projection_layer_spec.submodules,
                projection_type,
                projection_config.ffn_hidden_size,
            )
        else:
            self.projection = None

        self.input_tensor = None

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        if self.pre_process:  # always True
            self.input_tensor = input_tensor
        else:
            raise NotImplementedError()

    def forward(
        self,
        vision_data: Optional[torch.Tensor],
        grid_thw: torch.Tensor,
        extra_block_kwargs: dict = None,
    ) -> torch.Tensor:
        """Forward function of the Qwen2 Vision Model. This function passes the input tensors
        through the embedding layer and then the transformer.

        Args:
            x (torch.Tensor): input image/video data of shape [n_tokens, n_dims]
            grid_thw (torch.Tensor): the size tensor indicates grid size of each image/frame
            packed_seq_params (PackedSeqParams): parameters to build attention mask in the backend

        Returns:
            x (torch.Tensor): output after final transformer block of shape [b, s, h].
        """
        # NOTE: each epp stage should have thw_grids to build PackedSedParams
        assert grid_thw is not None
        if self.input_tensor is not None:
            vision_data = self.input_tensor[:, None]
            self.input_tensor = None

        (
            window_index,
            reverse_index,
            pos_ids,
            cu_seqlens,
            cu_window_seqlens,
        ) = self.build_windows_index(grid_thw, vision_data.device)
        pos_ids = self.window_partition(pos_ids, window_index, is_before_merger=True)
        rotary_pos_emb = self.rot_pos_emb(pos_ids)
        packed_seq_params_list = self.build_packed_seq_params_list(
            cu_seqlens, cu_window_seqlens
        )

        assert self.pre_process, f"{self.pre_process=}"
        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        if self.pre_process:
            vision_data = self.patch_embed(vision_data)[:, None]
            vision_data = self.window_partition(
                vision_data, window_index, is_before_merger=True
            )

        hidden_states = self.decoder(
            hidden_states=vision_data,
            attention_mask=None,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params_list=packed_seq_params_list,
            **(extra_block_kwargs or {}),
        )

        if not self.post_process:
            return hidden_states

        hidden_states = self.projection(hidden_states.view(-1, self.merge_hidden_size))
        hidden_states = self.window_unpartition(
            hidden_states, reverse_index, is_before_merger=False
        )
        return hidden_states

    def build_windows_index(
        self,
        grid_thw: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """calculate window mapping and positions for each token in the input tensor.

        Args:
            grid_thw (torch.Tensor): [N, 3] tensor, each row is [T, H, W], N is the number of images

        Returns:
            window_index: torch.Tensor, shape [M1].
            reverse_index: torch.Tensor, shape [M1].
            positions: torch.Tensor, shape [M2, 2].
            cu_seqlens: torch.Tensor, shape [N+1]. use for attention mask representation in full attention.
            cu_window_seqlens: torch.Tensor, shape [M3+1]. use for attention mask representation in window attention.

            Where M1 is the total number of tokens after merger, i.e. in llm.
                M2 is the total number of tokens before merger, i.e. in vit.
                satisfying M2 = M1 x self.spatial_merge_size x self.spatial_merge_size.
        """
        window_size = self.window_size
        spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        # window_index, reverse_index, cu_window_seqlens
        window_index = []
        window_index_id = 0
        cu_window_seqlens: list = [0]
        grid_thw_list = grid_thw.cpu().tolist()
        for grid_t, grid_h, grid_w in grid_thw_list:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w, device=device).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = window_size - llm_grid_h % window_size
            pad_w = window_size - llm_grid_w % window_size
            num_windows_h = (llm_grid_h + pad_h) // window_size
            num_windows_w = (llm_grid_w + pad_w) // window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t, num_windows_h, window_size, num_windows_w, window_size
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t, num_windows_h * num_windows_w, window_size, window_size
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            window_index_id += grid_t * llm_grid_h * llm_grid_w
            cu_seqlens_tmp = (
                seqlens.cumsum(0) * spatial_merge_unit + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
        window_index = torch.cat(window_index, dim=0)
        reverse_index = torch.argsort(window_index)
        cu_window_seqlens = torch.tensor(cu_window_seqlens, device=device, dtype=torch.int32)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens) 

        # positions, cu_seqlens
        pos_ids: list = []
        cu_seqlens: list = [0]
        for grid_t, grid_h, grid_w in grid_thw_list:
            hpos_ids = torch.arange(grid_h, device=device).unsqueeze(1).expand(-1, grid_w)
            wpos_ids = torch.arange(grid_w, device=device).unsqueeze(0).expand(grid_h, -1)
            hpos_ids = (
                hpos_ids.reshape(
                    grid_h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    grid_w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wpos_ids = (
                wpos_ids.reshape(
                    grid_h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    grid_w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(grid_t, 1))
            cu_seqlens.extend(
                [cu_seqlens[-1] + i * grid_h * grid_w for i in range(1, grid_t + 1)]
            )
        pos_ids = torch.cat(pos_ids, dim=0)
        cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.int32)

        return window_index, reverse_index, pos_ids, cu_seqlens, cu_window_seqlens

    def window_partition(
        self, x: torch.Tensor, window_index: torch.Tensor, is_before_merger: bool
    ) -> torch.Tensor:
        """Rearrange input tensor so that the tokens in the same window are contiguous in index.

        Args:
            x (torch.Tensor): input tensor of shape [N, ...]
            window_index (torch.Tensor): index mapping of shape [M1]
            is_before_merger (bool):
                if True, the input tensor should be group by self.spatial_merge_size x self.spatial_merge_size before partition.
                else, the input tensor is ready for partition.

        Returns: torch.Tensor: output tensor of shape [N, ...]
        """
        if is_before_merger:
            spatial_merge_size_unit = self.spatial_merge_size * self.spatial_merge_size
            seq_len = x.size(0) // spatial_merge_size_unit
            x = rearrange(
                x, "(n m) ... -> n m ... ", n=seq_len, m=spatial_merge_size_unit
            )
            x = x[window_index]
            x = rearrange(
                x, "n m ... -> (n m) ...", n=seq_len, m=spatial_merge_size_unit
            )
        else:
            x = x[window_index]
        return x

    def window_unpartition(
        self, x: torch.Tensor, reverse_index: torch.Tensor, is_before_merger: bool
    ) -> torch.Tensor:
        """Rearrange input tensor back to the original order.

        Args:
            x (torch.Tensor): input tensor of shape [N, ...]
            window_index (torch.Tensor): reverse index mapping of shape [M1]
            is_before_merger (bool):
                if True, the input tensor should be group by self.spatial_merge_size x self.spatial_merge_size before partition.
                else, the input tensor is ready for partition.

        Returns: torch.Tensor: output tensor of shape [N, ...]
        """
        if is_before_merger:
            spatial_merge_size_unit = self.spatial_merge_size * self.spatial_merge_size
            seq_len = x.size(0) // spatial_merge_size_unit
            x = rearrange(
                x, "(n m) ... -> n m ... ", n=seq_len, m=spatial_merge_size_unit
            )
            x = x[reverse_index]
            x = rearrange(
                x, "n m ... -> (n m) ...", n=seq_len, m=spatial_merge_size_unit
            )
        else:
            x = x[reverse_index]
        return x

    def rot_pos_emb(self, pos_ids: torch.Tensor) -> torch.Tensor:
        """Build rotary position embedding for the input tensor.

        Args:
            pos_ids (torch.Tensor): position ids of shape [N, 2]

        Returns:
            torch.Tensor: rotary position embedding of shape [N, H]
        """
        max_grid_size = pos_ids.max() + 1
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size).to(pos_ids.device)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        rotary_pos_emb = rotary_pos_emb[:, None, None, :].repeat(1, 1, 1, 2)
        return rotary_pos_emb

    def build_packed_seq_params_list(
        self,
        cu_seqlens: torch.Tensor,
        cu_window_seqlens: torch.Tensor,
    ) -> List[PackedSeqParams]:
        """Build packed sequence parameters for each transformer block.

        Args:
            cu_seqlens (torch.Tensor): [N+1] tensor, use for attention mask representation in full attention.
            cu_window_seqlens (torch.Tensor): [M3+1] tensor, use for attention mask representation in window attention.

        Returns:
            List[PackedSeqParams]: list of PackedSeqParams for each transformer block.
        """
        cu_seqlens_pack_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens, qkv_format="thd"
        )

        cu_window_seqlens_pack_params = PackedSeqParams(
            cu_seqlens_q=cu_window_seqlens,
            cu_seqlens_kv=cu_window_seqlens,
            qkv_format="thd",
        )

        packed_seq_params_list = [
            cu_seqlens_pack_params
            if i in self.fullatt_block_indexes
            else cu_window_seqlens_pack_params
            for i in range(self.num_layers)
        ]
        return packed_seq_params_list


class Qwen2_5_VLGPTModel(GPTModel):
    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal[
            "learned_absolute", "rope", "none"
        ] = "learned_absolute",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        seq_len_interpolation_factor: Optional[float] = None,
        mrope_section: List[int] = None,
    ) -> None:
        super().__init__(
            config,
            transformer_layer_spec,
            vocab_size,
            max_sequence_length,
            pre_process,
            post_process,
            fp16_lm_cross_entropy,
            parallel_output,
            share_embeddings_and_output_weights,
            position_embedding_type,
            rotary_percent,
            rotary_base,
            rope_scaling,
            seq_len_interpolation_factor,
        )

        if self.pre_process:
            self.embedding = QwenVLLanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
            )

        if self.position_embedding_type == "rope":
            self.rotary_pos_emb = Qwen2VLRotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
            )
            self.mrope_section = mrope_section or [16, 24, 24]

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(
                input_ids=input_ids, position_ids=position_ids
            )
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if self.position_embedding_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(position_ids, self.mrope_section)

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        if not self.post_process:
            return hidden_states

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(hidden_states, weight=output_weight)

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)

        return loss


class Qwen2_5_VLModel(LanguageModule):
    """Qwen2VL model.

    Args:
        config (TransformerConfig):
            Transformer config
        transformer_layer_spec (ModuleSpec):
            Specifies module to use for transformer layers
        vl_model_spec (ModuleSpec):
            Specifies module to use for vision-language model
        vocab_size (int):
            Vocabulary size
        max_sequence_length (int):
            maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional):
            Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional):
            Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional):
            Defaults to False.
        parallel_output (bool, optional):
            Do not gather the outputs, keep them split across tensor
            parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional):
            When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):
            Position embedding type.. Defaults to 'learned_absolute'.
        rotary_percent (float, optional):
            Percent of rotary dimension to use for rotary position embeddings.
            Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional):
            Base period for rotary position embeddings. Ignored unless
            position_embedding_type is 'rope'.
            Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional):
            scale of linearly interpolating RoPE for longer sequences.
            The value must be a float larger than 1.0. Defaults to None.
    """

    def __init__(
        self,
        config: TransformerConfig,
        vision_config: VLVisionConfig,
        projection_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vision_model_spec: ModuleSpec,
        projection_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal[
            "learned_absolute", "rope", "none"
        ] = "learned_absolute",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        seq_len_interpolation_factor: Optional[float] = None,
    ) -> None:
        super().__init__(config=config)

        self.pre_process = pre_process
        self.post_process = post_process

        if self.pre_process:
            self.vision_model = Qwen2_5_VLVisionModel(
                vl_config=vision_config,
                vision_model_spec=vision_model_spec,
                projection_config=projection_config,
                projection_layer_spec=projection_layer_spec,
            )

        self.lm_model = Qwen2_5_VLGPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
        )

        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_patch_size = vision_config.spatial_patch_size
        self.temporal_patch_size = vision_config.temporal_patch_size

        self.share_embeddings_and_output_weights = (
            self.lm_model.share_embeddings_and_output_weights
        )

    def set_input_tensor(self, input_tensor) -> None:
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, (
            "input_tensor should only be length 1 for Qwen2VL"
        )

        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.lm_model.set_input_tensor(input_tensor[0])

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        images: torch.Tensor = None,
        grid_thw: torch.Tensor = None,
        image_input_mask: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
    ) -> torch.Tensor:
        """Forward function of the Qwen2VL model.

        Args:
            image_data (torch.Tensor): input image of shape [total_thw_size, n_features].
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): attention mask for the language model [batch, 1, combined_seq_len, combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.
            *_input_mask: should not be None in the first PP stage
        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape [b, s, vocab_size].
        """
        use_inference_kv_cache = (
            inference_params is not None
            and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        if use_inference_kv_cache:
            raise NotImplementedError()

        if self.pre_process:
            image_embeds = None
            if grid_thw is not None and grid_thw.shape[0] > 0:
                image_embeds = self.vision_model(
                    vision_data=images,  # If None, vision model should use intermediate outputs (EPP > 1)
                    grid_thw=grid_thw,  # should provided in each EPP stage
                )
            else:
                images = (
                    torch.zeros(
                        self.spatial_merge_size**2,
                        self.spatial_patch_size**2 * 3,
                        dtype=torch.float32,
                        device=input_ids.device,
                    )
                    .unsqueeze(1)
                    .expand(-1, self.temporal_patch_size, -1)
                    .reshape(self.spatial_merge_size**2, -1)
                )
                grid_thw = torch.tensor(
                    [[1, self.spatial_merge_size, self.spatial_merge_size]],
                )
                image_embeds = (
                    self.vision_model(
                        vision_data=images,  # If None, vision model should use intermediate outputs (EPP > 1)
                        grid_thw=grid_thw,  # should provided in each EPP stage
                    ).mean()
                    * 0.0
                )

            combined_embeddings = self.lm_model.embedding(
                input_ids=input_ids,
                position_ids=None,
                image_embeds=image_embeds,
                image_input_mask=image_input_mask,
            )

        else:
            combined_embeddings = None

        output = self.lm_model(
            input_ids=None,
            position_ids=position_ids,  # None in encoder
            attention_mask=attention_mask,  # None in encoder
            decoder_input=combined_embeddings,  # only not None in the first decoder PP stage
            labels=labels,  # only not None in the last decoder PP stage
            inference_params=inference_params,  # currently always None
            packed_seq_params=packed_seq_params,  # currently always None
            **(extra_block_kwargs or {}),
        )
        return output

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: Optional[Dict] = None,
    ) -> ShardedStateDict:
        """Sharded state dict implementation for GPTModel backward-compatibility
        (removing extra state).

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        # 绕过对output_layer的检验
        sharded_state_dict = MegatronModule.sharded_state_dict(
            self, prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata
        )

        return sharded_state_dict
