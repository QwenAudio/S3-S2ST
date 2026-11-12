"""Joint codec training with the verified per-WRITE Talker-only drain mapping."""
import os
import warnings

os.environ["STREAMING_DEBUG_DISABLE_GATE"] = "0"
from streaming.s2st_exploration import pretrain_streaming_s2st_e2e as joint
from streaming.s2st_exploration.pretrain_streaming_s2st_segment_padded import (
    _install_talker_only_segment_padding,
    _install_codec_envelope_label_shift,
)


def model_provider(pre_process=True, post_process=True):
    model = joint.model_provider(pre_process, post_process)
    # Reuse exactly the successful segmentpad label semantics and differentiable
    # gather; never call its Talker-only model_provider (which freezes Thinker).
    _install_codec_envelope_label_shift(model)
    _install_talker_only_segment_padding(model)
    return model


if __name__ == "__main__":
    joint.streaming._patch_torch_dist_writer_compat()
    warnings.filterwarnings("ignore", message="The given NumPy array is not writable.*", category=UserWarning)
    provider = joint.streaming.build_streaming_data_iterators
    provider.is_distributed = True
    joint.pretrain(
        provider, model_provider, joint.ModelType.encoder_or_decoder,
        joint.streaming.forward_step,
        args_defaults={"tokenizer_type": "QWenVLTokenizer"},
        use_train_valid_test_dataset_provider_as_data_iterators=True,
        scale_lr_cond=joint.streaming.offline.scale_lr_cond,
    )
