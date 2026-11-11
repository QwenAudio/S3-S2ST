from typing import Any, Optional, Union

import torch
from transformers.configuration_utils import PretrainedConfig

from vllm.logger import init_logger

logger = init_logger(__name__)


# Utility functions for static/sliding cache update logic
def _static_cache_update(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_position: Optional[torch.LongTensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Updates the static cache tensors in place.

    Args:
        k_cache (`torch.Tensor`): The key cache tensor to update.
        v_cache (`torch.Tensor`): The value cache tensor to update.
        key_states (`torch.Tensor`): The new key states to add.
        value_states (`torch.Tensor`): The new value states to add.
        cache_position (`Optional[torch.LongTensor]`): The position indices where the new states should be inserted.
                                                       If None, the entire cache is overwritten (prefill).

    Returns:
        tuple[`torch.Tensor`, `torch.Tensor`]: The updated key and value cache tensors (modified in-place).
    """ # noqa: E501
    batch_size = key_states.size(0)
    if cache_position is None:
        # Prefill phase where seq_len potentially equals max_cache_len. Directly copy.
        k_cache.copy_(key_states)
        v_cache.copy_(value_states)
    else:
        # Generation phase. Update specific positions.

        k_cache[:batch_size, :, cache_position, :] = key_states
        v_cache[:batch_size, :, cache_position, :] = value_states

    length = cache_position[-1].item() + 1
    return k_cache[:batch_size, :, :length, :], v_cache[:batch_size, :, :length, :]


class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """ # noqa: E501

    is_compileable = False

    def __init__(self):
        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """ # noqa: E501
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed.""" # noqa: E501
        # TODO: deprecate this function in favor of `cache_position`
        raise NotImplementedError(
            "Make sure to implement `get_seq_length` in a subclass."
        )

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length (i.e. max capacity) of the cache object""" # noqa: E501
        raise NotImplementedError(
            "Make sure to implement `get_max_cache_shape` in a subclass."
        )

    def get_usable_length(
        self, new_seq_length: int, layer_idx: Optional[int] = 0
    ) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache.""" # noqa: E501
        
        """
        Cache without size limit -> all cache is usable
        Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
          length, we will need to evict part of the cache (and thus not all cache is usable)
        """ # noqa: E501

        max_length = self.get_max_cache_shape()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].numel():
                device = self.key_cache[layer_idx].device
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                    0, beam_idx.to(device)
                )
            if self.value_cache[layer_idx].numel():
                device = self.value_cache[layer_idx].device
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                    0, beam_idx.to(device)
                )

    @property
    def seen_tokens(self):
        logger.warning_once(
            "The `seen_tokens` attribute is deprecated and will be removed in v4.41. "
            "Use the `cache_position` model input instead."
        )
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        else:
            return None

    def get_mask_sizes(
        self, cache_position: torch.Tensor, layer_idx: int
    ) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns (i.e. sliding_window, chunk_size),
        for each layer.
        """ # noqa: E501
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length()
        kv_length = query_length + past_seen_tokens
        return kv_length, 0


class StaticCache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)` and `torch.export()`.

    Parameters:
        config (`PretrainedConfig`):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used. If you are manually setting the batch size, make sure to take into account the
            number of beams if you are running beam search
        max_cache_len (`int`, *optional*):
            The maximum sequence length with which the model will be used.
        device (`torch.device` or `str`, *optional*):
            The device on which the cache should be initialized. If you're using more than 1 computation device, you
            should pass the `layer_device_map` argument instead.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
        layer_device_map (`Optional[dict[int, Union[str, torch.device, int]]]]`, *optional*):
            Mapping between the layers and its device. This is required when you are manually initializing the cache
            and the model is split between different gpus. You can know which layers mapped to which device by
            checking the associated device_map: `model.hf_device_map`.


    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

        >>> inputs = tokenizer(text="My name is Llama", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        StaticCache()
        ```
    """ # noqa: E501

    is_compileable = True

    def __init__(
        self,
        config: PretrainedConfig,
        max_batch_size: int,
        max_cache_len: Optional[int] = None,
        device: Union[torch.device, str, None] = None,
        dtype: torch.dtype = torch.float32,
        layer_device_map: Optional[dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = (
            config.max_position_embeddings if max_cache_len is None else max_cache_len
        )

        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads  # noqa: E501
        self.head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )

        self._dtype = dtype
        self.num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )

        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

        # Note: There will be significant perf decrease
        # if switching to use 5D tensors instead.
        cache_shape = (
            self.max_batch_size,
            self.num_key_value_heads,
            self.max_cache_len,
            self.head_dim,
        )
        device = torch.device(device) if device is not None else None
        for idx in range(config.num_hidden_layers):
            if layer_device_map is not None:
                layer_device = layer_device_map[idx]
            else:
                layer_device = device
            new_layer_key_cache = torch.zeros(
                cache_shape, dtype=self._dtype, device=layer_device
            )
            new_layer_value_cache = torch.zeros(
                cache_shape, dtype=self._dtype, device=layer_device
            )

            # Note: `mark_static_address` is used to tag the cache
            # as a fixed data pointer, preventing compiled graph breaks
            # when updating the cache.
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """ # noqa: E501
        if cache_kwargs is None:
            cache_kwargs = {}

        key_states = key_states.to(self.key_cache[layer_idx].dtype)
        value_states = value_states.to(self.value_cache[layer_idx].dtype)
        return _static_cache_update(
            self.key_cache[layer_idx],
            self.value_cache[layer_idx],
            key_states,
            value_states,
            cache_kwargs.get("cache_position"),
        )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""  # noqa: E501
        # Occupied cache == any slot in the 3rd dim (sequence length)
        # holds a non-zero value. To save on compute, let's limit the
        # check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum()

    def get_max_cache_shape(self) -> Optional[int]:
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()

    def get_mask_sizes(
        self, cache_position: torch.Tensor, layer_idx: int
    ) -> tuple[int, int]:
        """
        Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for
        the given layer at `layer_idx`.
        The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns (i.e. sliding_window, chunk_size),
        for each layer.
        """ # noqa: E501
        kv_length = self.get_max_cache_shape()
        return kv_length, 0
