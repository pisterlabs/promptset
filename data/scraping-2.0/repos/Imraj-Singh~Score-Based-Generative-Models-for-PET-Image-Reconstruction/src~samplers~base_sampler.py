''' 
Inspired to https://github.com/yang-song/score_sde_pytorch/blob/main/sampling.py 
'''
from typing import Optional, Any, Dict, Tuple

import os
import torchvision
import numpy as np
import torch
import datetime

from tqdm import tqdm
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from ..utils import SDE, PSNR, SSIM
from ..third_party_models import OpenAiUNetModel

class BaseSampler:
    def __init__(self, 
        score: OpenAiUNetModel, 
        sde: SDE,
        predictor: callable,
        sample_kwargs: Dict,
        init_chain_fn: Optional[callable] = None,
        corrector: Optional[callable] = None,
        device: Optional[Any] = None
        ) -> None:

        self.score = score
        self.sde = sde
        self.predictor = predictor
        self.init_chain_fn = init_chain_fn
        self.sample_kwargs = sample_kwargs
        self.corrector = corrector
        self.device = device
    
    def sample(self,
        logg_kwargs: Dict = {},
        logging: bool = True 
        ) -> Tensor:
        if logging:
            writer = SummaryWriter(log_dir=os.path.join(logg_kwargs['log_dir'], str(logg_kwargs['sample_num'])))
        
        time_steps = np.linspace(1., self.sample_kwargs['eps'], self.sample_kwargs['num_steps'])

        step_size = time_steps[0] - time_steps[1]
        if self.sample_kwargs['start_time_step'] == 0:
            t = torch.ones(self.sample_kwargs['batch_size'], device=self.device)
            
            init_x = self.sde.prior_sampling([self.sample_kwargs['batch_size'], *self.sample_kwargs['im_shape']]).to(self.device)

        else:
            init_x = self.init_chain_fn(time_steps=time_steps).to(self.device)
        
        if logging:
            writer.add_image('init_x', torchvision.utils.make_grid(init_x, 
                normalize=True, scale_each=True), global_step=0)
            if logg_kwargs['ground_truth'] is not None: writer.add_image(
                'ground_truth', torchvision.utils.make_grid(logg_kwargs['ground_truth'], 
                    normalize=True, scale_each=True), global_step=0)
            if logg_kwargs['osem'] is not None: writer.add_image(
                'osem', torchvision.utils.make_grid(logg_kwargs['osem'], 
                    normalize=True, scale_each=True), global_step=0)
        
        x = init_x
        for i in tqdm(range(self.sample_kwargs['start_time_step'], self.sample_kwargs['num_steps'])):     
            time_step = torch.ones(self.sample_kwargs['batch_size'], device=self.device) * time_steps[i]
            x, x_mean, norm_factors = self.predictor(
                score=self.score,
                sde=self.sde,
                x=x,
                time_step=time_step,
                step_size=step_size,
                datafitscale=i/self.sample_kwargs['num_steps'],
                **self.sample_kwargs['predictor']
                )

            if self.corrector is not None:
                x = self.corrector(
                    x=x,
                    score=self.score,
                    sde=self.sde,
                    time_step=time_step,
                    datafitscale=i/self.sample_kwargs['num_steps'],
                    **self.sample_kwargs['corrector']
                    )

            if logging:
                if (i - self.sample_kwargs['start_time_step']) % logg_kwargs['num_img_in_log'] == 0:
                    writer.add_image('reco', torchvision.utils.make_grid(x_mean, normalize=True, scale_each=True), i)
                writer.add_scalar('PSNR', PSNR(x_mean[0, 0].cpu().numpy()*norm_factors[0,0].cpu().numpy(), logg_kwargs['ground_truth'][0, 0].cpu().numpy()), i)
                writer.add_scalar('SSIM', SSIM(x_mean[0, 0].cpu().numpy()*norm_factors[0,0].cpu().numpy(), logg_kwargs['ground_truth'][0, 0].cpu().numpy()), i)
        if logging:
            return x_mean, writer
        else:
            return x_mean, None 
