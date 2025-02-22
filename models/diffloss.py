import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math

from diffusion import create_diffusion


class DiffLoss(nn.Module):
    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps):
        super().__init__()
        self.target_channels = target_channels
        self.z_channels = z_channels
        self.depth = depth
        self.width = width
        self.num_sampling_steps = num_sampling_steps

        # Initialize diffusion models
        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="cosine")

        # Define the network for diffusion
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,
            z_channels=z_channels,
            num_res_blocks=depth
        )

    def forward(self, target, z, mask=None):
        # Debug: Print shapes of target and z
        print(f"Target shape: {target.shape}")
        print(f"z shape: {z.shape}")

        # Ensure target and z have the correct shapes
        target = target.view(target.size(0), -1)  # Flatten spatial dimensions
        z = z.view(z.size(0), -1)  # Flatten spatial dimensions

        # Compute the diffusion loss
        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss

    def sample(self, z, temperature=1.0, cfg=1.0):
        # diffusion loss sampling
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.target_channels).cuda()
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.target_channels).cuda()
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
            temperature=temperature
        )

        return sampled_token_latent
