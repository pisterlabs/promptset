"""
Adapted from: https://github.com/educating-dip/score_based_model_baselines/blob/main/src/utils/trainer.py

"""


from typing import Optional, Any, Dict
import os 
import torch 
import torchvision
import numpy as np 
import functools 

from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader
from .losses import loss_fn
from .ema import ExponentialMovingAverage
from .sde import SDE

from ..third_party_models import OpenAiUNetModel
from ..samplers import BaseSampler, Euler_Maruyama_sde_predictor, Langevin_sde_corrector, soft_diffusion_momentum_sde_predictor


def score_model_simple_trainer(
	score: OpenAiUNetModel,
	sde: SDE, 
	train_dl: DataLoader, 
	optim_kwargs: Dict,
	val_kwargs: Dict,
	device: Optional[Any] = None, 
	log_dir: str ='./',
	guided_p_uncond: Optional[Any] = None,
	) -> None:

	writer = SummaryWriter(log_dir=log_dir, comment='training-score-model')
	optimizer = Adam(score.parameters(), lr=optim_kwargs['lr'])
	for epoch in range(optim_kwargs['epochs']):
		avg_loss, num_items = 0, 0
		score.train()
		for idx, batch in tqdm(enumerate(train_dl), total = len(train_dl)):
			x = batch.to(device)
			if guided_p_uncond is not None:
				mask = torch.asarray(np.random.choice([0, 1], size=(len(x),), p=[guided_p_uncond, 1 - guided_p_uncond])).to(device)
				x[:,1,...] = x[:,1,...] * mask[:,None,None]
			loss = loss_fn(score, x, sde)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			avg_loss += loss.item() * x.shape[0]
			num_items += x.shape[0]
			if idx % optim_kwargs['log_freq'] == 0:
				writer.add_scalar('train/loss', loss.item(), epoch*len(train_dl) + idx) 
			if epoch == 0 and idx == optim_kwargs['ema_warm_start_steps']:
				ema = ExponentialMovingAverage(score.parameters(), decay=optim_kwargs['ema_decay'])
			if idx > optim_kwargs['ema_warm_start_steps'] or epoch > 0:
				ema.update(score.parameters())

		print('Average Loss: {:5f}'.format(avg_loss / num_items))
		writer.add_scalar('train/mean_loss_per_epoch', avg_loss / num_items, epoch + 1)
		torch.save(score.state_dict(), os.path.join(log_dir,'model.pt'))
		torch.save(ema.state_dict(), os.path.join(log_dir, 'ema_model.pt'))
		if val_kwargs['sample_freq'] > 0:
			if  epoch % val_kwargs['sample_freq']== 0:
				score.eval()

				predictor = functools.partial(Euler_Maruyama_sde_predictor, nloglik = None)
				corrector = functools.partial(Langevin_sde_corrector, nloglik = None) 

				sample_kwargs={
						'num_steps': val_kwargs['num_steps'],
						'start_time_step': 0,
						'batch_size': val_kwargs['batch_size'] if guided_p_uncond is None else x.shape[0],
						'im_shape': [1, *x.shape[2:]],
						'eps': val_kwargs['eps'],
						'predictor': {'aTweedy': False},
						'corrector': {'corrector_steps': 1}
						}

				if guided_p_uncond is not None:
					sample_kwargs['predictor'] = {
						"guidance_imgs": x[:,1,...].unsqueeze(1),
						"guidance_strength": 0.4
					}
					sample_kwargs['corrector'] = {
						"guidance_imgs": x[:,1,...].unsqueeze(1),
						"guidance_strength": 0.4
					}

				sampler = BaseSampler(
					score=score,
					sde=sde,
					predictor=predictor,
					corrector=corrector,
					init_chain_fn=None,
					sample_kwargs=sample_kwargs,
					device=device)
				x_mean, _ = sampler.sample(logging=False)

				if guided_p_uncond is not None:
					x_mean = torch.cat([x_mean[:,[0],...], x[:,[1],...]], dim=0)
					sample_grid = torchvision.utils.make_grid(x_mean, normalize=True, scale_each=True, nrow = x.shape[0])
					writer.add_image('unconditional samples', sample_grid, global_step=epoch)
				else:
					sample_grid = torchvision.utils.make_grid(x_mean, normalize=True, scale_each=True)
					writer.add_image('unconditional samples', sample_grid, global_step=epoch)
