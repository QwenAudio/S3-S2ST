import torchaudio.compliance.kaldi as Kaldi

CAMPPLUS_VOX = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
    },
}

CAMPPLUS_COMMON = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_VOX = {
    'obj': 'speakerlab.models.eres2net.ResNet.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_COMMON = {
    'obj': 'speakerlab.models.eres2net.ResNet_aug.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_Base_3D_Speaker = {
    'obj': 'speakerlab.models.eres2net.ResNet.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 32,
    },
}

ERes2Net_Large_3D_Speaker = {
    'obj': 'speakerlab.models.eres2net.ResNet.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 64,
    },
}

supports = {
    'damo/speech_campplus_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2',
        'model': CAMPPLUS_VOX,
        'model_pt': 'campplus_voxceleb.bin',
    },
    'damo/speech_campplus_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0',
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_common.bin',
    },
    'damo/speech_eres2net_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2',
        'model': ERes2Net_VOX,
        'model_pt': 'pretrained_eres2net.ckpt',
    },
    'damo/speech_eres2net_sv_zh-cn_16k-common': {
        'revision': 'v1.0.4',
        'model': ERes2Net_COMMON,
        'model_pt': 'pretrained_eres2net_aug.ckpt',
    },
    'damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.1',
        'model': ERes2Net_Base_3D_Speaker,
        'model_pt': 'eres2net_base_model.ckpt',
    },
    'damo/speech_eres2net_large_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.0',
        'model': ERes2Net_Large_3D_Speaker,
        'model_pt': 'eres2net_large_model.ckpt',
    },
}


class FBank(object):
    def __init__(self,
        n_mels,
        sample_rate,
        mean_nor: bool = False,
    ):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def __call__(self, wav, dither=0):
        sr = 16000
        assert sr==self.sample_rate
        
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        if len(wav.shape) == 3:
            wav = wav.squeeze(0)
        # select single channel
        if wav.shape[0] > 1:
            wav = wav[0, :]
        assert len(wav.shape) == 2 and wav.shape[0]==1
        feat = Kaldi.fbank(wav, num_mel_bins=self.n_mels,
            sample_frequency=sr, dither=dither, snip_edges=False)
        # feat: [T, N]
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat
