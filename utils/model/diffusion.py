import torch.nn.functional as F
import torch
import time
import numpy as np
from ..data.utils import lengths_to_mask

def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand

def karras_schedule(
    num_timesteps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: torch.device = None,
):
    rho_inv = 1.0 / rho
    # Clamp steps to 1 so that we don't get nans
    steps = torch.arange(num_timesteps, device=device) / max(num_timesteps - 1, 1)
    sigmas = sigma_min**rho_inv + steps * (
        sigma_max**rho_inv - sigma_min**rho_inv
    )
    sigmas = sigmas**rho

    return sigmas

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

def pseudo_huber_loss(input, target):
    c = 0.001
    return torch.sqrt((input - target) ** 2 + c**2) - c

class NoiseScheduleVP:
    def __init__(
            self,
            schedule='discrete',
            continuous_beta_0=0.1,
            continuous_beta_1=20.,
        ):
        self.schedule = schedule
        self.T = 1.
        self.total_N = 1000
        self.beta_0 = continuous_beta_0
        self.beta_1 = continuous_beta_1

    def numerical_clip_alpha(self, log_alphas, clipped_lambda=-5.1):
        """
        For some beta schedules such as cosine schedule, the log-SNR has numerical isssues. 
        We clip the log-SNR near t=T within -5.1 to ensure the stability.
        Such a trick is very useful for diffusion models with the cosine schedule, such as i-DDPM, guided-diffusion and GLIDE.
        """
        log_sigmas = 0.5 * torch.log(1. - torch.exp(2. * log_alphas))
        lambs = log_alphas - log_sigmas  
        idx = torch.searchsorted(torch.flip(lambs, [0]), clipped_lambda)
        if idx > 0:
            log_alphas = log_alphas[:-idx]
        return log_alphas

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
        Delta = self.beta_0**2 + tmp
        return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)

class Diffusion(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.num_timesteps = cfg.diffusion.num_timesteps
        self.noise_scheduler = NoiseScheduleVP(schedule='linear')
        self.ode_type = cfg.diffusion.type

        self.skip_steps = self.cfg.train.skip_steps
        if cfg.diffusion.timeschedule == 'linear':
            self.timesteps_schedule = torch.linspace(cfg.diffusion.sigma_min, cfg.diffusion.sigma_max, cfg.diffusion.num_timesteps//self.skip_steps, device=f'cuda:{int(cfg.gpu)}')
        else:
            self.timesteps_schedule = karras_schedule(num_timesteps=cfg.diffusion.num_timesteps//self.skip_steps, sigma_min=cfg.diffusion.sigma_min, sigma_max=cfg.diffusion.sigma_max, rho=cfg.diffusion.rho, device=cfg.device + str(cfg.gpu))

    def add_noise(self, x, noise=None, t=0, eps=1e-5):
        if noise is None:
            noise = torch.randn_like(x)

        if self.ode_type == 'vpode':
            alpha_t = append_dims(self.noise_scheduler.marginal_alpha(t), x.ndim)
            sigma_t = append_dims(self.noise_scheduler.marginal_std(t), x.ndim)

            perturbed_x = alpha_t * x + sigma_t * noise
        else:
            perturbed_x = x + append_dims(t, x.ndim) * noise
        return perturbed_x

    def ddim_step(self, x, t, t_offset, pred_model_s):
        if self.ode_type == 'vpode':
            ns = self.noise_scheduler
            lambda_t, lambda_t_offset = ns.marginal_lambda(t), ns.marginal_lambda(t_offset)
            h = lambda_t_offset - lambda_t
            log_alpha_t_offset = ns.marginal_log_mean_coeff(t_offset)
            sigma_t, sigma_t_offset = ns.marginal_std(t), ns.marginal_std(t_offset)
            alpha_t_offset = torch.exp(log_alpha_t_offset)
            phi_1 = torch.expm1(-h)
            [sigma_t, sigma_t_offset, alpha_t_offset, phi_1] = [append_dims(v, x.ndim) for v in [sigma_t, sigma_t_offset, alpha_t_offset, phi_1]]
            
            x_t = (
                sigma_t_offset / sigma_t * x
                - alpha_t_offset * phi_1 * pred_model_s
            )
        else:
            t = append_dims(t, x.ndim)
            t_offset = append_dims(t_offset, x.ndim)
            x_t = x + (t_offset - t) * (x - pred_model_s) / t
        return x_t

    def cal_loss(self, latents, latent_cond, latent_cond_none, denoiser, diffusion_ema):
        # Sample noise that we'll add to the latents
        device = latents.device
        noise = torch.randn_like(latents)

        # time schedule
        t = torch.randint(1, self.num_timesteps // self.skip_steps, (latents.shape[0], ), device=latents.device)
        t_offset = t - 1
        t_offset = torch.where(t_offset < 0, torch.zeros_like(t_offset), t_offset)
        t = self.timesteps_schedule[t]
        t_offset = self.timesteps_schedule[t_offset]

        noisy_model_input = self.add_noise(latents, noise, t)

        pred_out = denoiser(
            x_t=noisy_model_input, 
            timesteps=t, 
            text_emb=latent_cond
        )

        pred_out_uncond = denoiser(
            x_t=noisy_model_input, 
            timesteps=t, 
            text_emb=latent_cond_none
        )

        with torch.no_grad():
            # cfg step
            cond_x_0 = latents
            uncond_x_0 = pred_out_uncond
            pred_x0 = cond_x_0 + self.cfg.train.w * (cond_x_0 - uncond_x_0)

            if self.cfg.motion_ae.type == 'qae':
                pred_x0 = pred_x0.clamp(-1, 1)

            noisy_model_input_prev = self.ddim_step(noisy_model_input, t, t_offset, pred_x0)
            
            pred_out_ema = diffusion_ema(
                x_t=noisy_model_input_prev, 
                timesteps=t_offset, 
                text_emb=latent_cond
            )

        weight = (1 / (t - t_offset)).clamp(min=1e-5, max=1e7)

        if self.cfg.train.w != 0:
            loss = self.cfg.train.lambda_uncod * pseudo_huber_loss(pred_out_uncond, latents).mean()
        else:
            loss = 0

        loss += (pseudo_huber_loss(pred_out, pred_out_ema) * append_dims(weight, pred_out.ndim)).mean()
        

        return loss
    
    def cal_mld_loss(self, latents, latent_cond, denoiser):
        noise = torch.randn_like(latents)

        # time schedule
        t = torch.randint(1, self.num_timesteps // self.skip_steps, (latents.shape[0], ), device=latents.device)
        t = self.timesteps_schedule[t]

        noisy_model_input = self.add_noise(latents, noise, t)

        pred_out = denoiser(
            x_t=noisy_model_input, 
            timesteps=t, 
            text_emb=latent_cond
        )

        loss = pseudo_huber_loss(pred_out, latents).mean()

        return loss

    
    @torch.no_grad()
    def sample(self, bs, lengths, condition, steps, denoiser, motion_encoder, condition_encoder, device='cuda:0'):
        shape = (bs, denoiser.token_num, denoiser.latent_dim)

        x_T = torch.randn(shape, device=device) * self.cfg.diffusion.sigma_max
        
        latent_condition = condition_encoder(condition).to(device)

        x_t = x_T
        reverse_time = list(int(r_t) for r_t in np.linspace((self.num_timesteps // self.cfg.train.skip_steps) - 1, 0, steps+1))
        reverse_time = self.timesteps_schedule[reverse_time]

        for t_idx in range(len(reverse_time) - 1):
            t = torch.full((bs,), reverse_time[t_idx], device=device, dtype=torch.float)
            x = denoiser(
                x_t,
                t, 
                latent_condition
            )

            if motion_encoder.type_ == 'qae':
                x = x.clamp(-1, 1)

            if t_idx < len(reverse_time) - 1:
                x_t = self.add_noise(x, torch.randn_like(x_T), t=torch.tensor(reverse_time[t_idx+1].clone().detach(), device=device))

        mask = lengths_to_mask(lengths, device=device)
        pred_motion = motion_encoder.motion_decode(x, ~mask)
        return pred_motion
    
    @torch.no_grad()
    def test_time(self, bs, lengths, condition, steps, denoiser, motion_encoder, condition_encoder, device='cuda:0'):
        shape = (bs, denoiser.token_num, denoiser.latent_dim)

        x_T = torch.randn(shape, device=device) * self.cfg.diffusion.sigma_max
        
        start_time = time.time()
        latent_condition = condition_encoder(condition).to(device)
        condition_time = time.time() - start_time

        x_t = x_T
        reverse_time = list(int(r_t) for r_t in np.linspace((self.num_timesteps // self.cfg.train.skip_steps) - 1, 0, steps+1))
        reverse_time = self.timesteps_schedule[reverse_time]

        start_time = time.time()
        for t_idx in range(len(reverse_time) - 1):
            t = torch.full((bs,), reverse_time[t_idx], device=device, dtype=torch.float)
            x = denoiser(
                x_t,
                t, 
                latent_condition
            )

            if motion_encoder.type_ == 'qae':
                x = x.clamp(-1, 1)

            if t_idx < len(reverse_time) - 1:
                x_t = self.add_noise(x, torch.randn_like(x_T), t=torch.tensor(reverse_time[t_idx+1].clone().detach(), device=device))
        denoiser_time = time.time() - start_time

        mask = lengths_to_mask(lengths, device=device)
        start_time = time.time()
        pred_motion = motion_encoder.motion_decode(x, ~mask)
        decoder_time = time.time() - start_time
        return pred_motion, [condition_time, denoiser_time, decoder_time]
    