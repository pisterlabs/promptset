"""
Adapted from: https://github.com/educating-dip/score_based_model_baselines/blob/main/src/utils/exp_utils.py

"""

import os
import time
import torch
import functools
from math import ceil
from pathlib import Path

from .sde import VESDE, VPSDE, HeatDiffusion
from .ema import ExponentialMovingAverage

from ..third_party_models import OpenAiUNetModel
from ..samplers import (BaseSampler, Euler_Maruyama_sde_predictor, Langevin_sde_corrector, 
                    chain_simple_init, decomposed_diffusion_sampling_sde_predictor)

def get_standard_score(config, sde, use_ema, load_path = None, load_model=True):
    if load_model:
        assert load_path is not None, "set load path"

    if str(config.model.model_name).lower() == 'OpenAiUNetModel'.lower():
        score = OpenAiUNetModel(
            image_size=config.data.im_size,
            in_channels=config.model.in_channels,
            model_channels=config.model.model_channels,
            out_channels=config.model.out_channels,
            num_res_blocks=config.model.num_res_blocks,
            attention_resolutions=config.model.attention_resolutions,
            marginal_prob_std=None if isinstance(sde,HeatDiffusion) else sde.marginal_prob_std,
            channel_mult=config.model.channel_mult,
            conv_resample=config.model.conv_resample,
            dims=config.model.dims,
            num_heads=config.model.num_heads,
            num_head_channels=config.model.num_head_channels,
            num_heads_upsample=config.model.num_heads_upsample,
            use_scale_shift_norm=config.model.use_scale_shift_norm,
            resblock_updown=config.model.resblock_updown,
            use_new_attention_order=config.model.use_new_attention_order,
            max_period=config.model.max_period
            )
    else:
        raise NotImplementedError

    if load_model: 
        print(f'load score model from path: {load_path}')
        if use_ema:
            ema = ExponentialMovingAverage(score.parameters(), decay=0.999)
            ema.load_state_dict(torch.load(os.path.join(load_path,'ema_model.pt')))
            ema.copy_to(score.parameters())
        else:
            score.load_state_dict(torch.load(os.path.join(load_path, config.sampling.model_name)))

    return score

def get_standard_sde(config):

    if config.sde.type.lower() == 'vesde':
        sde = VESDE(
            sigma_min=config.sde.sigma_min, 
            sigma_max=config.sde.sigma_max
            )
    elif config.sde.type.lower() == 'vpsde':
        sde = VPSDE(
            beta_min=config.sde.beta_min, 
            beta_max=config.sde.beta_max
            )
    elif config.sde.type.lower() == "heatdiffusion":
        sde = HeatDiffusion(
            sigma_min=config.sde.sigma_min,
            sigma_max=config.sde.sigma_max,
            T_max=config.sde.T_max
        )

    else:
        raise NotImplementedError

    return sde

def get_standard_sampler(config, score, sde, nll, im_shape, observation=None, 
                            osem=None, guidance_imgs=None, device=None):
    """
    nll should be a function of x, i.e. a functools.partial with fixed norm_factors, attn_factors, contamination, measurements

    """
    if config.sampling.name.lower() == 'naive':
        predictor = functools.partial(
            Euler_Maruyama_sde_predictor,
            nloglik = nll) 
        sample_kwargs = {
            'num_steps': int(config.sampling.num_steps),
            'start_time_step': ceil(float(config.sampling.pct_chain_elapsed) * int(config.sampling.num_steps)),
            'batch_size': config.sampling.batch_size,
            'im_shape': im_shape,
            'eps': config.sampling.eps,
            'predictor': {'aTweedy': False, 'penalty': float(config.sampling.penalty), "guidance_imgs": guidance_imgs, "guidance_strength": config.sampling.guidance_strength},
            'corrector': {}
            }
    elif config.sampling.name.lower() == 'dps':
        predictor = functools.partial(
            Euler_Maruyama_sde_predictor,
            nloglik = nll) 
        sample_kwargs = {
            'num_steps': int(config.sampling.num_steps),
            'batch_size': config.sampling.batch_size,
            'start_time_step': ceil(float(config.sampling.pct_chain_elapsed) * int(config.sampling.num_steps)),
            'im_shape': im_shape,
            'eps': config.sampling.eps,
            'predictor': {'aTweedy': True, 'penalty': float(config.sampling.penalty), "guidance_imgs": guidance_imgs, "guidance_strength": config.sampling.guidance_strength},
            'corrector': {},
            }
    elif config.sampling.name.lower() == 'dds' or config.sampling.name.lower() == 'dds_3d':
        predictor = functools.partial(
                    decomposed_diffusion_sampling_sde_predictor,
                    nloglik = nll)
        sample_kwargs = {
            'num_steps': int(config.sampling.num_steps),
            'batch_size': config.sampling.batch_size,
            'start_time_step': ceil(float(config.sampling.pct_chain_elapsed) * int(config.sampling.num_steps)),
            'im_shape': im_shape,
            'eps': config.sampling.eps,
            'predictor': {"guidance_imgs": guidance_imgs, 
                        "guidance_strength": config.sampling.guidance_strength,
                        'use_simplified_eqn': True, 
                        'eta': config.sampling.stochasticity},
            'corrector': {},
            }
    else:
        raise NotImplementedError

    corrector = None
    if config.sampling.add_corrector:
        corrector = functools.partial(Langevin_sde_corrector,
            nloglik = nll  )
        sample_kwargs['corrector']['corrector_steps'] = 1
        sample_kwargs['corrector']['penalty'] = float(config.sampling.penalty)

    init_chain_fn = None
    if sample_kwargs['start_time_step'] > 0:
        init_chain_fn = functools.partial(  
        chain_simple_init,
        sde=sde,
        osem=osem,
        start_time_step=sample_kwargs['start_time_step'],
        im_shape=im_shape,
        batch_size=sample_kwargs['batch_size'],
        device=device
        )

    sampler = BaseSampler(
        score=score, 
        sde=sde,
        predictor=predictor,         
        corrector=corrector,
        init_chain_fn=init_chain_fn,
        sample_kwargs=sample_kwargs, 
        device=config.device,
        )
    
    return sampler