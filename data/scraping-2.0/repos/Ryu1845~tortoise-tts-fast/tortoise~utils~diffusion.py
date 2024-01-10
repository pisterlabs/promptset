"""
This is an almost carbon copy of gaussian_diffusion.py from OpenAI's ImprovedDiffusion repo, which itself:

This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
# AGPL: a notification must be added stating that changes have been made to that file.

import enum

import numpy as np
import torch
import torch as th
from k_diffusion.sampling import sample_dpmpp_2m, sample_euler_ancestral
from tqdm import tqdm

from tortoise.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP, model_wrapper

K_DIFFUSION_SAMPLERS = {"k_euler_a": sample_euler_ancestral, "dpm++2m": sample_dpmpp_2m}
SAMPLERS = ["dpm++2m", "p", "ddim"]


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = "previous_x"  # the model predicts x_{t-1}
    START_X = "start_x"  # the model predicts x_0
    EPSILON = "epsilon"  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = "learned"
    FIXED_SMALL = "fixed_small"
    FIXED_LARGE = "fixed_large"
    LEARNED_RANGE = "learned_range"


class LossType(enum.Enum):
    MSE = "mse"  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        "rescaled_mse"  # use raw MSE loss (with RESCALED_KL when learning variances)
    )
    KL = "kl"  # use the variational lower-bound
    RESCALED_KL = "rescaled_kl"  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,  # this is generally False
        conditioning_free=False,
        conditioning_free_k=1,
        ramp_conditioning_free=True,
        sampler="ddim",
    ):
        self.sampler = sampler
        self.model_mean_type = ModelMeanType(model_mean_type)
        self.model_var_type = ModelVarType(model_var_type)
        self.loss_type = LossType(loss_type)
        self.rescale_timesteps = rescale_timesteps
        self.conditioning_free = conditioning_free
        self.conditioning_free_k = conditioning_free_k
        self.ramp_conditioning_free = ramp_conditioning_free

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def k_diffusion_sample_loop(
        self,
        k_sampler,
        pbar,
        model,
        shape,
        noise=None,  # all given
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        device=None,  # ALL UNUSED
        model_kwargs=None,  # {'precomputed_aligned_embeddings': precomputed_embeddings},
        progress=False,  # unused as well
    ):
        assert isinstance(model_kwargs, dict)
        if device is None:
            device = next(model.parameters()).device
        noise.new_ones([noise.shape[0]])

        def model_split(*args, **kwargs):
            model_output = model(*args, **kwargs)
            model_epsilon, model_var = th.split(
                model_output, model_output.shape[1] // 2, dim=1
            )
            return model_epsilon, model_var

        #
        """
        print(self.betas)
        print(th.tensor(self.betas))
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=th.tensor(self.betas))
        """
        noise_schedule = NoiseScheduleVP(
            schedule="linear", continuous_beta_0=0.1 / 4, continuous_beta_1=20.0 / 4
        )

        def model_fn_prewrap(x, t, *args, **kwargs):
            """
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                c_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
            print(t)
            print(self.timestep_map)
            exit()
            """
            """
            model_output = model(x, self._scale_timesteps(t*4000), **model_kwargs)
            out = self.p_mean_variance(model, x, t*4000, model_kwargs=model_kwargs)
            return out['pred_xstart']
            """
            x, _ = x.chunk(2)
            t, _ = (t * 1000).chunk(2)
            res = torch.cat(
                [
                    model_split(x, t, conditioning_free=True, **model_kwargs)[0],
                    model_split(x, t, **model_kwargs)[0],
                ]
            )
            pbar.update(1)
            return res

        model_fn = model_wrapper(
            model_fn_prewrap,
            noise_schedule,
            model_type="noise",  # "noise" or "x_start" or "v" or "score"
            model_kwargs=model_kwargs,
            guidance_type="classifier-free",
            condition=th.Tensor(1),
            unconditional_condition=th.Tensor(1),
            guidance_scale=self.conditioning_free_k,
        )
        """
        model_fn = model_wrapper(
            model_fn_prewrap,
            noise_schedule,
            model_type='x_start',
            model_kwargs={}
        )
        #
        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver")
        x_sample = dpm_solver.sample(
            noise,
            steps=20,
            order=3,
            skip_type="time_uniform",
            method="singlestep",
        )
        """
        dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
        x_sample = dpm_solver.sample(
            noise,
            steps=self.num_timesteps,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        )
        #'''
        return x_sample

        # HF DIFFUSION ATTEMPT
        """
        from .hf_diffusion import EulerAncestralDiscreteScheduler
        Scheduler = EulerAncestralDiscreteScheduler()
        Scheduler.set_timesteps(100)
        for timestep in Scheduler.timesteps:
            noise_input = Scheduler.scale_model_input(noise, timestep)
            ts = s_in * timestep
            model_output = model(noise_input, ts, **model_kwargs)
            model_epsilon, _model_var = th.split(model_output, model_output.shape[1]//2, dim=1)
            noise, _x0 = Scheduler.step(model_epsilon, timestep, noise)
        return noise
        """

        # KARRAS DIFFUSION ATTEMPT
        """
        TRAINED_DIFFUSION_STEPS = 4000 # HARDCODED
        ratio = TRAINED_DIFFUSION_STEPS/14.5
        def call_model(*args, **kwargs):
            model_output = model(*args, **kwargs)
            model_output, model_var_values = th.split(model_output, model_output.shape[1]//2, dim=1)
            return model_output
        print(get_sigmas_karras(self.num_timesteps, sigma_min=0.0, sigma_max=4000, device=device))
        exit()
        sigmas = get_sigmas_karras(self.num_timesteps, sigma_min=0.03, sigma_max=14.5, device=device)
        return k_sampler(call_model, noise, sigmas, extra_args=model_kwargs, disable=not progress)
        '''
        sigmas = get_sigmas_karras(self.num_timesteps, sigma_min=0.03, sigma_max=14.5, device=device)
        step = 0 # LMAO
        global_sigmas = None
        #
        def fakemodel(x, t, **model_kwargs):
            print(t,global_sigmas*ratio)
            return model(x, t, **model_kwargs)
        def denoised(x, sigmas, **extra_args):
            t = th.tensor([self.num_timesteps-step-1] * shape[0], device=device)
            nonlocal global_sigmas
            global_sigmas = sigmas
            with th.no_grad():
                out = self.p_sample(
                    fakemodel,
                    x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                return out["sample"]
        def callback(d):
            nonlocal step
            step += 1

        return k_sampler(denoised, noise, sigmas, extra_args=model_kwargs, callback=callback, disable=not progress)
        '''
        """

    def sample_loop(self, *args, **kwargs):
        s = self.sampler
        if s == "p":
            return self.p_sample_loop(*args, **kwargs)
        elif s == "ddim":
            return self.ddim_sample_loop(*args, **kwargs)
        elif s == "dpm++2m":
            if self.conditioning_free is not True:
                raise RuntimeError("cond_free must be true")
            with tqdm(total=self.num_timesteps) as pbar:
                return self.k_diffusion_sample_loop(
                    K_DIFFUSION_SAMPLERS[s], pbar, *args, **kwargs
                )
        else:
            raise RuntimeError("sampler not impl")


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)
