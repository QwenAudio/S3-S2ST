from abc import ABC
import torch
import torch.nn.functional as F
from .matcha_decoder import (Decoder, ConditionalDecoder)
import logging
from funasr.utils.hinter import hint_once
import time
from funasr.train_utils.set_all_random_seed import set_all_random_seed


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        n_feats: int,
        cfm_params: dict,
        n_spks: int = 1,
        spk_emb_dim: int = 128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.get("solver", "euler")
        self.sigma_min = cfm_params.get("sigma_min", 1e-4)

        self.estimator = None
        self.t_scheduler = cfm_params.get("t_scheduler", "linear")  # used for inference
        self.training_timestep_scheduler = cfm_params.get("training_timestep_scheduler", "cosine")  # used for training
        self.training_cfg_rate = cfm_params.get("training_cfg_rate", 0.0)
        self.inference_cfg_rate = cfm_params.get("inference_cfg_rate", 0.0)
        self.reg_loss_type = cfm_params.get("reg_loss_type", "l2")
        self.learn_target = cfm_params.get("learn_target", False)
        self.t_sensitive_loss = cfm_params.get("t_sensitive_loss", None)
        self.use_fixed_noise_infer = cfm_params.get("use_fixed_noise_infer", False)
        self.rand_noise = None
        self.cfm_params = cfm_params

    def get_rand_noise(self, mu: torch.Tensor):
        if self.use_fixed_noise_infer:
            max_len = self.cfm_params.get("max_len", 50*300)  # frame_rate=50, max duration: 300 s
            if self.rand_noise is None or self.rand_noise.shape[2] < mu.shape[2]:
                self.rand_noise = torch.randn([1, mu.shape[1], max_len]).to(mu)
                logging.info("init random noise for Flow")
            return self.rand_noise[:, :, :mu.shape[2]]
        else:
            return torch.randn_like(mu)

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, mask_rand=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = self.get_rand_noise(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1).to(mu)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, mask_rand=mask_rand)

    def solve_euler(self, x, t_span, mu, mask, spks=None, cond=None, mask_rand=None):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        # sol = []

        steps = 1
        z, bz = x, x.shape[0]
        while steps <= len(t_span) - 1:
            if self.inference_cfg_rate > 0:
                x_in = torch.concat([x, x], dim=0)
                mask_in = torch.concat([mask, mask], dim=0)
                mu_in = torch.concat([mu, torch.zeros_like(mu)], dim=0)
                t_in = torch.concat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)
                spks_in = torch.concat([spks, torch.zeros_like(spks)], dim=0) if spks is not None else None
                if isinstance(cond, torch.Tensor):
                    cond_in = torch.concat([cond, torch.zeros_like(cond)], dim=0)
                else:
                    cond_in = dict(
                        prompt=[
                            torch.concat([cond["prompt"][0], torch.zeros_like(cond["prompt"][0])], dim=0),
                            torch.concat([cond["prompt"][1], cond["prompt"][1]], dim=0),
                        ]
                    )
            else:
                x_in, mask_in, mu_in, t_in = x, mask, mu, t
                spks_in, cond_in = spks, cond
            fm_dec_onstep_time = time.time()
            dphi_dt = self.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in, mask_rand=mask_rand)
            logging.info(f"fm dec {steps} step time: {(time.time() - fm_dec_onstep_time) * 1000.0:.2f} ms")
            if self.inference_cfg_rate > 0:
                dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [bz, bz], dim=0)
                dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt -
                           self.inference_cfg_rate * cfg_dphi_dt)
            # Classifier-Free Guidance inference introduced in VoiceBox
            # if self.inference_cfg_rate > 0:
            #     cfg_dphi_dt = self.estimator(
            #         x, mask,
            #         torch.zeros_like(mu), t,
            #         torch.zeros_like(spks) if spks is not None else None,
            #         torch.zeros_like(cond) if isinstance(cond, torch.Tensor) else dict(
            #             prompt=[torch.zeros_like(cond["prompt"][0]), cond["prompt"][1]]
            #         ),
            #     )
            #     dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt -
            #                self.inference_cfg_rate * cfg_dphi_dt)

            if not self.learn_target:
                x = x + dt * dphi_dt
            else:
                x = x + dt * (dphi_dt - z)
            t = t + dt
            # sol.append(x)
            if steps < len(t_span) - 1:
                dt = t_span[steps + 1] - t
            steps += 1

        return x
        # return sol[-1]

    def calc_reg_loss(self, prediction, target, loss_mask):
        if self.reg_loss_type == 'l1':
            hint_once("use l1 loss to train CFM", "CFM_LOSS_L1")
            l1_loss = F.l1_loss(prediction, target, reduction="none")
            l1_loss = l1_loss * loss_mask
            return l1_loss
        elif self.reg_loss_type == 'l2':
            hint_once("use l2 loss to train CFM", "CFM_LOSS_L2")
            l2_loss = F.mse_loss(prediction, target, reduction="none")
            l2_loss = l2_loss * loss_mask
            return l2_loss
        elif self.reg_loss_type == 'huber':
            hint_once("use huber loss to train CFM", "CFM_LOSS_HUBER")
            huber_loss = F.huber_loss(prediction, target, reduction="none", delta=1.0)
            huber_loss = huber_loss * loss_mask
            return huber_loss
        else:
            hint_once("use l1+l2 loss to train CFM", "CFM_LOSS_L1_L2")
            l1_loss = F.l1_loss(prediction, target, reduction="none")
            l1_loss = l1_loss * loss_mask
            l2_loss = 0.5 * F.mse_loss(prediction, target, reduction="none")
            l2_loss = l2_loss * loss_mask
            return l1_loss * 0.5 + l2_loss * 0.5

    def compute_loss(self, x1, mask, mu, spks=None, cond=None, reduction='none', z=None, mask_rand=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        hint_once(f"using {self.training_timestep_scheduler} timestep scheduler for training", "timestep_scheduler")
        if self.training_timestep_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        # sample noise p(x_0)
        if z is None:
            hint_once(f"using random noise for training", "noise_noise")
            z = torch.randn_like(x1)
        else:
            hint_once(f"using fixed noise for training", "noise_noise")

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype) > self.training_cfg_rate
            mu = mu * cfg_mask
            if spks is not None:
                spks = spks * cfg_mask.squeeze(-1)
            if cond is not None:
                if isinstance(cond, torch.Tensor):
                    cond = cond * cfg_mask
                elif isinstance(cond, dict):
                    for key in cond.keys():
                        # cond must be List[value tensor, length tensor]
                        cond[key][0] = cond[key][0] * cfg_mask

        pred, hs = self.estimator(y, mask, mu, t.squeeze(), spks, cond, return_hidden=True, mask_rand=mask_rand)
        if not self.learn_target:
            loss = self.calc_reg_loss(pred, u, mask)
            hint_once("learn path direction.", "learn_target")
        else:
            pred_u = pred - (1 - self.sigma_min) * z
            loss = self.calc_reg_loss(pred_u, u, mask)
            hint_once("learn target directly.", "learn_target")
        if self.t_sensitive_loss is not None:
            if isinstance(self.t_sensitive_loss, dict):
                if self.t_sensitive_loss["name"] == "exp":
                    alpha, beta = self.t_sensitive_loss["betas"]
                    loss = loss * torch.pow(alpha, beta * (1.0 - t))
                    hint_once(f"use exponential t sensitive loss, betas: ({alpha}, {beta})", "t_sensitive_loss")
            else:
                loss = loss * ((1.0 - t) + 1.0)
                hint_once("use linear t sensitive loss", "t_sensitive_loss")
        if reduction == "mean":
            loss = loss.sum() / (torch.sum(mask) * u.shape[1])
        return loss, hs


class CFM(BASECFM):
    def __init__(self, in_channels, out_channel, cfm_params, decoder_params,
                 n_spks=1, spk_emb_dim=64, decoder_name="Decoder"):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # Just change the architecture of the estimator here
        if decoder_name == "Decoder":
            self.estimator = Decoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)
        else:
            self.estimator = ConditionalDecoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)
