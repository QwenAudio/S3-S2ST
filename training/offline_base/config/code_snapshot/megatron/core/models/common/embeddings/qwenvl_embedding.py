
import torch
from torch import Tensor
from copy import deepcopy
from typing import Literal
from .language_model_embedding import LanguageModelEmbedding
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core import tensor_parallel

class QwenVLLanguageModelEmbedding(LanguageModelEmbedding):

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
        num_tokentypes: int = 0,
    ):
        lm_config = deepcopy(config)
        lm_config.hidden_dropout = 0.0
        lm_config.sequence_parallel = False
        super().__init__(
            config=lm_config,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            position_embedding_type=position_embedding_type,
            num_tokentypes=num_tokentypes,
        )

        # for vl
        self.sequence_parallel = config.sequence_parallel        
        self.reduce_scatter_embeddings = False
        self.final_embedding_dropout = torch.nn.Dropout(config.hidden_dropout)

    def forward(self, 
                input_ids: Tensor, 
                position_ids: Tensor, 
                tokentype_ids: int = None,
                image_embeds: Tensor = None,
                image_input_mask: Tensor = None,
                ) -> Tensor:

        embeddings = super().forward(input_ids, position_ids, tokentype_ids)

        embeddings = embeddings.clone()
        if image_embeds is not None:
            embeddings[image_input_mask.T] = image_embeds

        if self.sequence_parallel:
            if not self.reduce_scatter_embeddings:
                embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings)
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            with tensor_parallel.get_cuda_rng_tracker().fork():
                embeddings = self.final_embedding_dropout(embeddings)
        else:
            embeddings = self.final_embedding_dropout(embeddings)

        return embeddings
