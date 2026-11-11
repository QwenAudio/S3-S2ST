import torch
from typing import Tuple
import torch.nn as nn
from torch.nn import functional as F
from .matcha_decoder import Upsample1D, Downsample1D
from funasr.models.transformer.utils.nets_utils import make_pad_mask, pad_list
from einops import repeat, pack
import logging


class UpSamplingRegulator(nn.Module):
    def __init__(
            self,
            channels: int,
            sampling_ratios: Tuple,
            out_channels: int = None,
            groups: int = 1,
    ):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        out_channels = out_channels or channels
        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            for ratio in sampling_ratios:
                if ratio > 1:
                    module = Upsample1D(channels=channels, channel_first=False)
                else:
                    module = nn.Linear(channels, channels)
                norm = nn.LayerNorm(channels)
                act = nn.LeakyReLU()
                model.extend([module, norm, act])
        model.append(
            nn.Linear(channels, out_channels)
        )
        self.model = nn.Sequential(*model)

    def forward(self, x, xlens, y=None, y_lens=None, **kwargs):
        # x, out, y in (B, T, D)
        out = self.model(x)
        out = out[:, :y.shape[1]]
        olens = y_lens

        return out, olens


class SimpleRepeatRegulator(nn.Module):
    def __init__(
            self,
            channels: int,
            sampling_ratios: int,
    ):
        super().__init__()
        self.sampling_ratios = int(sampling_ratios)

    def forward(self, x, xlens, y=None, y_lens=None, **kwargs):
        out = torch.repeat_interleave(x, self.sampling_ratios, dim=1)
        olens = xlens * self.sampling_ratios

        return out, olens


class PreLookaheadLayer(nn.Module):
    def __init__(self, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=pre_lookahead_len+1,
            stride=1, padding=0,
        )

    def forward(self, inputs, ilens):
        """
        inputs: (batch_size, seq_len, channels)
        """
        outputs = inputs.transpose(1, 2).contiguous()
        # look ahead
        outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode='constant', value=0)
        outputs = self.conv1(outputs).transpose(1, 2)

        mask = (~make_pad_mask(ilens).unsqueeze(-1).to(inputs.device))
        # residual connection
        outputs = (outputs + inputs) * mask

        return outputs, ilens


class LookaheadRepeatRegulator(nn.Module):
    def __init__(
        self,
        channels: int,
        sampling_ratios: int,
        lookahead_len: int,
    ):
        super().__init__()
        self.sampling_ratios = int(sampling_ratios)
        self.pre_lookahead_layer = PreLookaheadLayer(channels, lookahead_len)

    def forward(self, x, xlens, y=None, y_lens=None, **kwargs):
        x, _ = self.pre_lookahead_layer(x, xlens)
        out = torch.repeat_interleave(x, self.sampling_ratios, dim=1)
        olens = xlens * self.sampling_ratios

        return out, olens


class DownSamplingRegulator(nn.Module):
    def __init__(
            self,
            channels: int,
            sampling_ratios: Tuple,
            out_channels: int = None,
            groups: int = 1,
    ):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        out_channels = out_channels or channels
        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            for ratio in sampling_ratios:
                if ratio > 1:
                    module = Downsample1D(dim=channels, channel_first=False, padding=2)
                else:
                    module = nn.Linear(channels, channels)
                norm = nn.LayerNorm(channels)
                act = nn.LeakyReLU()
                model.extend([module, norm, act])

        model.append(
            nn.Linear(channels, out_channels)
        )
        self.model = nn.Sequential(*model)

    def forward(self, x, xlens, y=None, y_lens=None, **kwargs):
        # x, out, y in (B, T, D)
        out = self.model(x)
        out = out[:, :y.shape[1]]
        olens = y_lens

        return out, olens


class InterpolateRegulator(nn.Module):
    def __init__(
            self,
            channels: int,
            sampling_ratios: Tuple,
            out_channels: int = None,
            groups: int = 1,
            mode="nearest",
            align_corners=False,
    ):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        out_channels = out_channels or channels
        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            for _ in sampling_ratios:
                module = nn.Conv1d(channels, channels, 3, 1, 1)
                norm = nn.GroupNorm(groups, channels)
                act = nn.Mish()
                model.extend([module, norm, act])

        model.append(
            nn.Conv1d(channels, out_channels, 1, 1)
        )
        self.model = nn.Sequential(*model)
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x, xlens, y=None, ylens=None, **kwargs):
        # x in (B, T, D)
        mask = (~make_pad_mask(ylens)).to(x).unsqueeze(-1)
        align_corners_opt = {}
        if self.mode in ["linear", "bilinear","bicubic", "trilinear"]:
            align_corners_opt = dict(align_corners=self.align_corners)
        x = F.interpolate(x.transpose(1, 2).contiguous(), size=y.shape[1],
                          mode=self.mode, **align_corners_opt)
        out = self.model(x).transpose(1, 2).contiguous()
        olens = ylens

        return out * mask, olens


class UpsamplingCifRegulator(nn.Module):
    def __init__(
            self,
            channels: int,
            hidden_channels: int = 256,
            upsamling_ratios: Tuple = (2, 2, 2, 2),
            n_attn_layers: int = 2,
            attention_heads: int = 8,
            dropout_rate: float = 0.1,
            prompt_dim: int = None,
            concat_after: bool = False,
    ):
        super().__init__()
        self.prompt_dim = prompt_dim
        # 2 for content + xvec
        self.in_conv = nn.Sequential(
            nn.Conv1d(channels * 2, hidden_channels, 3, 1, 1),
            nn.GroupNorm(num_groups=1, num_channels=hidden_channels),
            nn.Mish()
        )
        self.out_conv = nn.Conv1d(hidden_channels, channels, 3, 1, 1)

        from .matcha_decoder import Upsample1D
        from funasr.models.transformer.decoder import (
            DecoderLayer, MultiHeadedAttention, PositionwiseFeedForward, repeat
        )
        self.upsamling_ratios = upsamling_ratios
        self.module_list = nn.ModuleList()
        for ratio in upsamling_ratios:
            # add xvec to each block
            conv_module = Upsample1D(
                hidden_channels + channels,
                use_conv=False,
                use_conv_transpose=True,
                stride=ratio,
                channel_first=True,
                out_channels=hidden_channels,
            )
            attn_block = repeat(
                n_attn_layers,
                lambda lnum: DecoderLayer(
                    hidden_channels,
                    MultiHeadedAttention(
                        attention_heads, hidden_channels, dropout_rate
                    ),
                    MultiHeadedAttention(
                        attention_heads, hidden_channels, dropout_rate
                    ),
                    PositionwiseFeedForward(hidden_channels, hidden_channels * 4, dropout_rate),
                    dropout_rate,
                    normalize_before=True,
                    concat_after=concat_after,
                ),
            )
            one_block = nn.ModuleList([conv_module, attn_block])
            if prompt_dim is not None:
                prompt_proj = torch.nn.Linear(prompt_dim, hidden_channels)
                one_block.append(prompt_proj)
            self.module_list.append(one_block)

        from funasr.models.paraformer.cif_predictor import CifPredictor
        self.cif_predictor = CifPredictor(
            idim=hidden_channels,
            l_order=16,
            r_order=16,
        )
        from funasr.models.transformer.embedding import PositionalEncoding
        self.pos_enc = PositionalEncoding(hidden_channels, dropout_rate=0.0)

    def forward(
            self,
            h: torch.Tensor, h_lengths: torch.Tensor,
            y: torch.Tensor = None, y_lengths: torch.Tensor = None,
            conditions: dict = None, **kwargs
    ):
        # xvec in (B,D), prompt in (B,T,D), h in (B,T,D)
        xvec, (prompt, prompt_lengths) = conditions["xvec"], conditions["prompt"]

        spks = repeat(xvec, "b c -> b c t", t=h.shape[1])
        x = pack([h.transpose(1, 2).contiguous(), spks], "b * t")[0]
        x = self.in_conv(x)  # x in (B, C ,T)
        x_lens = h_lengths
        for modules, ratio in zip(self.module_list, self.upsamling_ratios):
            conv_module, attn_block, prompt_proj = modules
            # expand xvec and forward conv
            spks = repeat(xvec, "b c -> b c t", t=x.shape[2])
            x = conv_module(pack([x, spks], "b * t")[0])
            x_lens = x_lens * ratio

            # make masks and forward attention layer
            x_mask = (~make_pad_mask(x_lens)[:, None, :]).to(x.device)
            prompt_mask = (~make_pad_mask(prompt_lengths)[:, None, :]).to(x.device)
            x = x * x_mask
            x = attn_block(
                self.pos_enc(x.transpose(1, 2).contiguous()), x_mask,
                self.pos_enc(prompt_proj(prompt)), prompt_mask
            )[0]
            x = x.transpose(1, 2).contiguous() * x_mask

        # x in (B,C,T)
        x, pred_token_num, _, _ = self.cif_predictor(x.transpose(1, 2).contiguous(), target_label_length=y_lengths)
        x_lens = pred_token_num.long() if y_lengths is None else y_lengths
        x_mask = (~make_pad_mask(x_lens)[:, None, :]).to(x.device)
        out = self.out_conv(x.transpose(1, 2).contiguous() * x_mask) * x_mask

        return out.transpose(1, 2).contiguous(), pred_token_num


class UpsamplingBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            channels: int,
            stride=2,
            groups=1,
            channel_first=False,
    ):
        super().__init__()
        self.channel_first = channel_first
        self.stride = stride

        self.up_conv = nn.ConvTranspose1d(in_channels, channels, stride * 2, stride, 1)
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv1d(channels, channels, 3, padding=1),
            torch.nn.GroupNorm(groups, channels),
            nn.Mish(),
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv1d(channels, channels, 3, padding=1),
            torch.nn.GroupNorm(groups, channels),
            nn.Mish(),
        )
        self.res_conv = torch.nn.Conv1d(channels, channels, 1)

    def forward(self, x, ilens, **kwargs):
        if not self.channel_first:
            x = x.transpose(1, 2)

        olens = ilens * self.stride
        o_masks = (~make_pad_mask(olens))[:, None, :].to(x)
        res = out = self.up_conv(x) * o_masks

        out = self.block1(out) * o_masks + out
        out = self.block2(out) * o_masks + out
        out = out + self.res_conv(res) * o_masks

        if not self.channel_first:
            out = out.transpose(1, 2)

        return out, olens


class UpsamplingCifRegulatorV2(nn.Module):
    def __init__(
            self,
            channels: int,
            hidden_channels: int = 256,
            upsamling_ratios: Tuple = (2, 2, 2, 2),
            n_attn_layers: int = 2,
            attention_heads: int = 8,
            dropout_rate: float = 0.1,
            prompt_dim: int = None,
            concat_after: bool = False,
    ):
        super().__init__()
        self.prompt_dim = prompt_dim
        # 2 for content + xvec
        self.in_conv = nn.Sequential(
            nn.Conv1d(channels * 2, hidden_channels, 3, 1, 1),
            nn.GroupNorm(num_groups=1, num_channels=hidden_channels),
            nn.ReLU()
        )
        self.out_conv = nn.Conv1d(hidden_channels, channels, 3, 1, 1)

        self.upsamling_ratios = upsamling_ratios
        self.up_modules = nn.ModuleList()
        for ratio in upsamling_ratios:
            # add xvec to each block
            conv_module = UpsamplingBlock(
                hidden_channels,
                hidden_channels,
                stride=ratio,
                channel_first=False,
            )
            self.up_modules.append(conv_module)

        from funasr.models.conformer.encoder import ConformerEncoder
        self.attn_model = ConformerEncoder(
            input_size=hidden_channels,
            output_size=hidden_channels,
            attention_heads=attention_heads,
            linear_units=hidden_channels * 4,
            num_blocks=n_attn_layers,
            dropout_rate=dropout_rate,
            positional_dropout_rate=dropout_rate,
            attention_dropout_rate=0.0,
            input_layer="linear",
            normalize_before=True,
            concat_after=concat_after,
            macaron_style=True,
            rel_pos_type="latest",
            pos_enc_layer_type="abs_pos",
            selfattention_layer_type="selfattn",
            use_cnn_module=True,
            cnn_module_kernel=31,
        )

        self.prompt_proj = None
        if prompt_dim is not None:
            self.prompt_proj = torch.nn.Linear(prompt_dim, hidden_channels)

        from funasr.models.paraformer.cif_predictor import CifPredictor
        self.cif_predictor = CifPredictor(
            idim=hidden_channels,
            l_order=32,
            r_order=32,
        )
        self.dur_block = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.LayerNorm(1),
            nn.ReLU(),
        )

    def forward(
            self,
            h: torch.Tensor, h_lengths: torch.Tensor,
            y: torch.Tensor = None, y_lengths: torch.Tensor = None,
            conditions: dict = None, **kwargs
    ):
        # xvec in (B,D), prompt in (B,T,D), h in (B,T,D)
        xvec, (prompt, prompt_lengths) = conditions["xvec"], conditions["prompt"]

        spks = repeat(xvec, "b c -> b c t", t=h.shape[1])
        x = pack([h.transpose(1, 2).contiguous(), spks], "b * t")[0]
        x_mask = (~make_pad_mask(h_lengths))[:, :, None].to(h)
        x = self.in_conv(x).transpose(1, 2) * x_mask  # x -> (B, T, C)
        # add prompt to sequence
        x_lens = h_lengths
        if self.prompt_proj is not None:
            prompt = self.prompt_proj(prompt)
            xs_list, x_len_list = [], []
            for idx, (_prompt_len, _x_len) in enumerate(zip(prompt_lengths, x_lens)):
                xs_list.append(torch.concat([prompt[idx, :_prompt_len], x[idx, :_x_len]], dim=0))
                x_len_list.append(_prompt_len + _x_len)

            x = pad_list(xs_list, pad_value=0.0)
            x_lens = torch.tensor(x_len_list, dtype=torch.int64).to(x.device)

        x, x_lens, _ = self.attn_model(x, x_lens)

        # remove prompt from sequence
        if self.prompt_proj is not None:
            xs_list = []
            for idx, (_prompt_len, _x_len) in enumerate(zip(prompt_lengths, h_lengths)):
                xs_list.append(x[idx, _prompt_len: _prompt_len+_x_len])

            x = pad_list(xs_list, pad_value=0.0)

        x_mask = (~make_pad_mask(h_lengths)).to(x.device)
        phone_durs = self.dur_block(x).squeeze(-1) * x_mask

        x_lens = h_lengths
        for conv_module in self.up_modules:
            x, x_lens = conv_module(x, x_lens)

        # x in (B,C,T)
        x, pred_token_num, _, _ = self.cif_predictor(x, target_label_length=y_lengths)
        x_lens = pred_token_num.long() if y_lengths is None else y_lengths
        x_mask = (~make_pad_mask(x_lens)[:, None, :]).to(x.device)
        out = self.out_conv(x.transpose(1, 2).contiguous() * x_mask) * x_mask

        return out.transpose(1, 2).contiguous(), pred_token_num, phone_durs


class RepeatLengthRegulator(torch.nn.Module):
    """Repeat Length regulator module for feed-forward Transformer.

    This is a module of length regulator described in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The length regulator expands char or
    phoneme-level embedding features to frame-level by repeating each
    feature based on the corresponding predicted durations.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(self, pad_value=0.0):
        """Initilize length regulator module.

        Args:
            pad_value (float, optional): Value used for padding.

        """
        super().__init__()
        self.pad_value = pad_value

    def forward(
            self,
            xs: torch.Tensor, xs_len: torch.Tensor=None,
            ys: torch.Tensor=None, ys_len: torch.Tensor=None,
            **kwargs
    ):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of sequences of char or phoneme embeddings (B, Tmax, D).
            ds (LongTensor): Batch of durations of each frame (B, T).
            alpha (float, optional): Alpha value to control speed of speech.

        Returns:
            Tensor: replicated input tensor based on durations (B, T*, D).

        """
        ds: torch.Tensor = kwargs.get('ds', None)
        assert ds is not None
        alpha: float = kwargs.get('alpha', 1.0)
        if alpha != 1.0:
            assert alpha > 0
            ds = torch.round(ds.float() * alpha).long()

        if ds.sum() == 0:
            logging.warning(
                "predicted durations includes all 0 sequences. "
                "fill the first element with 1."
            )
            # NOTE(kan-bayashi): This case must not be happened in teacher forcing.
            #   It will be happened in inference with a bad duration predictor.
            #   So we do not need to care the padded sequence case here.
            ds[ds.sum(dim=1).eq(0)] = 1

        if ys_len is not None:
            ds = ds * (ys_len / ds.sum(dim=1)).unsqueeze(-1)
        ds = ds.round().long()

        repeat = []
        for i, (x, d) in enumerate(zip(xs, ds)):
            ex = torch.repeat_interleave(x, d, dim=0)
            if ys_len is not None:
                ex = F.interpolate(
                    ex.transpose(0, 1).unsqueeze(0),
                    size=ys_len[i], mode="nearest",
                )
                ex = ex.squeeze(0).transpose(0, 1)
            repeat.append(ex)

        if ys_len is not None:
            return pad_list(repeat, self.pad_value), ys_len
        else:
            return pad_list(repeat, self.pad_value), ds.sum(dim=1)


class PositionalDecoderRegulator(nn.Module):
    def __init__(
            self,
            in_dim: int,
            decoder_conf: dict,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.decoder_conf = decoder_conf
        hidden_dim = decoder_conf['encoder_output_size']
        self.in_proj, self.out_proj = None, None
        if in_dim != hidden_dim:
            self.in_proj = nn.Linear(in_dim, hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, in_dim)

        from funasr.models.transformer.embedding import PositionalEncoding
        self.positional_encoding = PositionalEncoding(hidden_dim, 0.1)
        from funasr.models.transformer.decoder import TransformerDecoder
        self.decoder = TransformerDecoder(**decoder_conf)

    def forward(
            self,
            x: torch.Tensor, xlens: torch.Tensor,
            y: torch.Tensor = None, ylens: torch.Tensor = None,
            **kwargs
    ):
        # x in (B, T, D)
        ds: torch.Tensor = kwargs.get('ds', None)
        assert ds is not None

        if self.in_proj is not None:
            xmask = (~make_pad_mask(xlens)).to(x).unsqueeze(-1)
            x = self.in_proj(x) * xmask

        # add positional encoding
        x = self.positional_encoding(x)

        if ylens is not None:
            hlens = ylens   # for training
        else:
            hlens = ds.sum(dim=1).round().long()  # for inference

        hs = torch.zeros([x.shape[0], hlens.max(), x.shape[-1]]).to(x)
        outs, _ = self.decoder(x, xlens, hs, hlens)

        if self.out_proj is not None:
            out_mask = (~make_pad_mask(hlens)).to(outs).unsqueeze(-1)
            outs = self.out_proj(outs) * out_mask

        return outs, hlens
