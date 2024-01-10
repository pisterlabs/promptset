"""
Author: Stankevich Andrey, MIPT <stankevich.as@phystech.edu>
Implementation is mostly copied from openai DALL-E open-source
repository https://github.com/openai/DALL-E
"""

import torch
import torch.nn as nn

from collections import OrderedDict
from functools import partial


class EncoderBlock(nn.Module):

	def __init__(
		self,
		n_in,
		n_out,
		n_layers
	):

		super().__init__()

		self.n_in  = n_in
		self.n_out = n_out
		self.n_hid = self.n_out

		self.n_layers  = n_layers
		self.post_gain = 1.

		self.id_path = nn.Conv2d(self.n_in, self.n_out, 1) \
			if (self.n_in != self.n_out) else nn.Identity()



		self.res_path = nn.Sequential(
			OrderedDict([
				('conv_1', nn.Conv2d(self.n_in, self.n_hid, kernel_size=3, padding=1)),
				('relu_1', nn.ReLU()),
				('conv_2', nn.Conv2d(self.n_hid, self.n_hid, kernel_size=3, padding=1)),
				('relu_2', nn.ReLU()),
				('conv_3', nn.Conv2d(self.n_hid, self.n_hid, kernel_size=3, padding=1)),
				('relu_3', nn.ReLU()),
				('conv_4', nn.Conv2d(self.n_hid, self.n_out, kernel_size=3, padding=1)),
				('relu_4', nn.ReLU())				
			]))


	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.id_path(x) + self.post_gain * self.res_path(x) 


class Encoder(nn.Module):

	def __init__(
		self, 
		n_groups=3,
		n_block_per_group=2,
		input_channels=3,
		n_hid=256,
		n_out=256
	):

		super().__init__()
		self.n_groups          = n_groups
		self.n_block_per_group = n_block_per_group
		self.input_channels    = input_channels
		self.n_hid             = n_hid
		self.n_out             = n_out   

		br       = range(self.n_block_per_group)
		n_layers = self.n_groups * self.n_block_per_group
		make_b   = partial(EncoderBlock, n_layers=n_layers)

		blocks   = [
			('input'  , nn.Conv2d(self.input_channels, self.n_hid, kernel_size=1, stride=1)),
			('relu'   , nn.ReLU()),
			('group_1', nn.Sequential(
				OrderedDict([
						*[(f'block_{j + 1}', make_b(
							1 * self.n_hid, 1 * self.n_hid)) for j in br
						],
						('pool', nn.Conv2d(self.n_hid, self.n_hid, kernel_size=2, stride=2))
			])))
		]

		for i in range(1, self.n_groups):
			factor = int(2 ** (i - 1))

			blocks.extend([
				(f'group_{i+1}', nn.Sequential(
					OrderedDict([
						*[(f'block_{j + 1}', make_b(
							factor * self.n_hid if j == 0 \
							else factor * 2 * self.n_hid, factor * 2 * self.n_hid)) \
							for j in br], 
							('pool', nn.Conv2d(factor * 2 * self.n_hid, factor * 2 * self.n_hid, kernel_size=2, stride=2))
			])))])

		blocks.append(('output', nn.Sequential(
			OrderedDict([
				('relu'   , nn.ReLU()),
				('conv', nn.Conv2d((2 ** (self.n_groups - 1)) * self.n_hid, self.n_out, 1))
			])
		)))

		self.blocks = nn.Sequential(OrderedDict(blocks))


	def forward(self, x: torch.Tensor) -> torch.Tensor:

		if len(x.shape) != 4: raise ValueError(f'input shape {x.shape} is not 4d')
		if x.shape[1] != self.input_channels:
			raise ValueError(f'input has {x.shape[1]} channels but model built for {self.input_channels}')
			
		return self.blocks(x)