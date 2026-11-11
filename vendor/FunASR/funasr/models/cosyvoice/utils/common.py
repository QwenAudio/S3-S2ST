import torch, os
from torch.nn.utils import remove_weight_norm


def causal_linear_interpolation(x, ratio):
    assert x.shape[1] >=2
    x_interpolate = torch.zeros(x.shape[0], x.shape[1] * ratio, x.shape[2]).to(x)
    # 前2帧单独处理
    for i in range(ratio * 2):
        x_interpolate[:, i] = x[:, 1] * (i + 1) / (ratio * 2) + x[:, 0] * (ratio * 2 - i - 1) / (ratio * 2)
    # 后N-2帧按4倍插值
    for i in range(2, x.shape[1]):
        for j in range(ratio):
            x_interpolate[:, i * ratio + j] = x[:, i] * (j + 1) / ratio + x[:, i - 1] * (ratio - j - 1) / ratio
    return x_interpolate

class CausalUpsample(torch.nn.Module):
    def __init__(self, dim, ratio=4):
        super().__init__()
        # kernel_size = ratio，stride = ratio，无需 padding → 天然因果
        self.conv_transpose = torch.nn.ConvTranspose1d(dim, dim, kernel_size=ratio, stride=ratio)

    def forward(self, x):
        # x: (B, T, D) → (B, D, T)
        x = x.transpose(1, 2)
        x = self.conv_transpose(x)
        return x.transpose(1, 2)  # (B, T*ratio, D)

def fade_in_out(fade_in_mel, fade_out_mel, window):
    device = fade_in_mel.device
    fade_in_mel, fade_out_mel = fade_in_mel.cpu(), fade_out_mel.cpu()
    mel_overlap_len = int(window.shape[1] / 2)
    if fade_in_mel.device == torch.device('cpu'):
        fade_in_mel = fade_in_mel.clone()
    fade_in_mel[:, :mel_overlap_len] = fade_in_mel[:, :mel_overlap_len] * window[:, :mel_overlap_len] + fade_out_mel[:, -mel_overlap_len:] * window[:, mel_overlap_len:]
    return fade_in_mel.to(device)


class VAEExtraction():
    def __init__(self, vae_model_path='', vae_frame_rate=50, **kwargs):
        from funasr.models.cosyvoice.vae import EAR_VAE
        self.vae = EAR_VAE(frame_rate=vae_frame_rate)
        vae_state_dict = {k.replace('generator.model.', ''): v for k, v in torch.load(vae_model_path).items() if k.startswith('generator.model.')}
        print(self.vae.load_state_dict(vae_state_dict, strict=False))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = 'cuda:{}'.format(self.local_rank)
        self.vae.eval()
        self.vae.to(self.device)
        for n, m in self.vae.named_modules():
            try:
                remove_weight_norm(m)
                print('remove_weight_norm {}'.format(n))
            except:
                continue
        for m, n in self.vae.named_parameters():
            n.detach_()

    @torch.no_grad()
    def inference(self, y, y_lengths):
        if self.vae.training is True:
            self.vae.eval()
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
            feat = self.vae.encode(y.unsqueeze(dim=1), use_sample=True)
        subsample_rate = y.shape[1] / feat.shape[2]
        feat_lengths = (y_lengths / subsample_rate).to(y_lengths)
        return feat, feat_lengths