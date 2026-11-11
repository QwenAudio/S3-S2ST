"""MinMo S2T model specifications for Megatron."""

from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.models.multimodal.audio_encoder import SenseVoiceEncoder
from megatron.core.models.multimodal.audio_adaptor import AudioAdaptor


def get_fun_audio_encoder_spec(use_te: bool = False) -> ModuleSpec:
    if use_te:
        # Use Transformer Engine implementation
        return ModuleSpec(
            module=SenseVoiceEncoder,
            submodules={}
        )
    else:
        # Use local implementation
        return ModuleSpec(
            module=SenseVoiceEncoder,
            submodules={}
        )


def get_fun_audio_adaptor_spec(use_te: bool = False) -> ModuleSpec:
    if use_te:
        # Use Transformer Engine implementation
        return ModuleSpec(
            module=AudioAdaptor,
            submodules={}
        )
    else:
        # Use local implementation
        return ModuleSpec(
            module=AudioAdaptor,
            submodules={}
        )