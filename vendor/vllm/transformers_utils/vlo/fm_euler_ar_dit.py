from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class FlowMatchEulerDiscreteSchedulerARDiT(FlowMatchEulerDiscreteScheduler):
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        per_token_timesteps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        per_timesteps: Optional[torch.Tensor] = None,
        next_timesteps: Optional[torch.Tensor] = None,
        num_inference_steps_ar_dit: Optional[int] = None,
        batch_size: Optional[int] = None,
        patch_size: Optional[int] = None,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            per_token_timesteps (`torch.Tensor`, *optional*):
                The timesteps for each token in the sample.
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.
        """

        if isinstance(timestep, tuple([int, torch.IntTensor, torch.LongTensor])):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `FlowMatchEulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        if per_timesteps is None:
            if per_token_timesteps is not None:
                per_token_sigmas = per_token_timesteps / self.config.num_train_timesteps

                sigmas = self.sigmas[:, None, None]
                lower_mask = sigmas < per_token_sigmas[None] - 1e-6
                lower_sigmas = lower_mask * sigmas
                lower_sigmas, _ = lower_sigmas.max(dim=0)
                dt = (per_token_sigmas - lower_sigmas)[..., None]
            else:
                sigma = self.sigmas[self.step_index]
                sigma_next = self.sigmas[self.step_index + 1]
                dt = sigma_next - sigma
            # print(dt)
        # import ipdb; ipdb.set_trace()
        # print(dt)
        if per_timesteps is not None:  # [3200, 1]
            dt = (next_timesteps - per_timesteps) / num_inference_steps_ar_dit
            cur_t = per_timesteps / num_inference_steps_ar_dit
            dt = torch.clamp(dt, min=-1, max=0)
            # print(dt)
            dt = dt.reshape(
                batch_size,
                1,
                model_output.shape[-2] // patch_size,
                1,
                model_output.shape[-1] // patch_size,
                1,
            )  # torch.Size([2, 1, 40, 1, 40, 1])
            dt = dt.expand(
                batch_size,
                model_output.shape[1],
                model_output.shape[-2] // patch_size,
                patch_size,
                model_output.shape[-1] // patch_size,
                patch_size,
            ).reshape(
                batch_size,
                model_output.shape[1],
                model_output.shape[-2],
                model_output.shape[-1],
            )  # torch.Size([2, 16, 80, 80])

            cur_t = cur_t.reshape(
                batch_size,
                1,
                model_output.shape[-2] // patch_size,
                1,
                model_output.shape[-1] // patch_size,
                1,
            )  # torch.Size([2, 1, 40, 1, 40, 1])
            cur_t = cur_t.expand(
                batch_size,
                model_output.shape[1],
                model_output.shape[-2] // patch_size,
                patch_size,
                model_output.shape[-1] // patch_size,
                patch_size,
            ).reshape(
                batch_size,
                model_output.shape[1],
                model_output.shape[-2],
                model_output.shape[-1],
            )  # torch.Size([2, 16, 80, 80])

        prev_sample = (
            sample + dt * model_output
        )  # model_output & prev_sample : torch.Size([2, 16, 80, 80]
        x0 = sample - cur_t * model_output

        # upon completion increase step index by one
        self._step_index += 1
        if per_token_timesteps is None:
            # Cast sample back to model compatible dtype
            prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample,)

        if per_timesteps is not None:
            return (
                FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample),
                dt,
                x0,
            )

        else:
            return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
