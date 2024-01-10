import sys
sys.path.append('openai_vpt')

import cv2
import torch
import pickle
import numpy as np
from gym3.types import DictType
from openai_vpt.lib.policy import MinecraftAgentPolicy

from distance_fns import DISTANCE_FUNCTIONS
from LatentSpaceMineCLIP import SLIDING_WINDOW_SIZE

AGENT_RESOLUTION = (128, 128)
CONTEXT = {'first': torch.tensor([[False]])}

class LatentSpaceVPT:
    def __init__(self, distance_fn='euclidean', device='cuda'):
        self.latents = []  # Python List while training, Numpy array while inference
        self.distance_function = DISTANCE_FUNCTIONS[distance_fn]
        self.device = device
    
    @torch.no_grad()
    def load(self, episode_actions, latents_folder='weights/ts_bc/latents_vpt/'):
        for vid_id, _ in episode_actions.episode_starts:
            _, name = vid_id.rsplit('/', 1)
            vid_latents = np.load(latents_folder + name + '.npy', allow_pickle=True)
            self.latents.append(vid_latents)

        self.latents = torch.from_numpy(np.vstack(self.latents)).to(self.device)
        print(f'Loaded VPT latent space with {len(self.latents)} latents')
        return self
    
    @torch.no_grad()
    def load_OLD(self, latents_file='weights/ts_bc/latents_vpt.npy'):  # TODO update to new format
        self.latents = torch.from_numpy(np.load(latents_file, allow_pickle=True)).to(self.device)
        print(f'Loaded VPT latent space with {len(self.latents)} latents')
        return self
    
    def save(self, latents_file='weights/ts_bc/latents_vpt'):  # TODO remove?
        latents = np.array(self.latents)
        np.save(latents_file, latents)

    @torch.no_grad()
    def train_episode(self, vpt_model, frames, vid_id, save_dir='weights/ts_bc/latents_vpt/'):
        episode_latents = []
        model_state = vpt_model.initial_state(1)

        resized_frames = np.empty((frames.shape[0], AGENT_RESOLUTION[1], AGENT_RESOLUTION[0], 3), dtype=np.uint8)
        for ts in range(frames.shape[0]):
            resized_frame = cv2.resize(frames[ts], AGENT_RESOLUTION)
            resized_frames[ts] = resized_frame
        frames = torch.tensor(resized_frames).to(self.device)

        for ts in range(SLIDING_WINDOW_SIZE-1, len(frames)):  # Start at Frame 15 because of MineCLIP needing 16-frame batches
            frame = frames[ts].unsqueeze(0).unsqueeze(0)  # Add 2 extra dimensions for vpt
            (latent, _), model_state = vpt_model.net({'img': frame}, model_state, context=CONTEXT)
            latent = latent[0][0].to('cpu').numpy().astype('float16')

            episode_latents.append(latent)
        
        del(frames)

        np.save(save_dir + vid_id.rsplit('/', 1)[-1], np.array(episode_latents))

    def get_distances(self, latent):
        return self.distance_function(self.latents, latent)
    
    def get_distance(self, idx, latent):
        return self.distance_function(self.latents[idx], latent)

    def get_nearest(self, latent): # TODO removed episode_starts
        # TODO assert latents is numpy array

        diffs = self.latents - latent
        diffs = abs(diffs).sum(1)  # Sum up along the single latents exponential difference to the current latent
        nearest_idx = diffs.argmin()#.to('cpu').item() # TODO remove .to('cpu').item()
        return nearest_idx

def load_vpt(model_file='weights/vpt/foundation-model-1x.model', weights_file='weights/vpt/foundation-model-1x.weights', device='cuda'):
    agent_parameters = pickle.load(open(model_file, 'rb'))

    policy_kwargs = agent_parameters['model']['args']['net']['args']
    pi_head_kwargs = agent_parameters['model']['args']['pi_head_opts']
    pi_head_kwargs['temperature'] = float(pi_head_kwargs['temperature'])

    agent = MinecraftAgentPolicy(policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs, action_space=DictType())
    agent.load_state_dict(torch.load(weights_file), strict=False)
    agent.eval()

    return agent.to(device)
