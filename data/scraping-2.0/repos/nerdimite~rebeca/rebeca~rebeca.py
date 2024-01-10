import pickle
from tqdm.auto import tqdm

import cv2
import numpy as np
from scipy.spatial.distance import euclidean
import torch
from torch import nn
from model import VPTEncoder, Controller, VPTCNNEncoder
from memory import Memory

import torch as th
from gym3.types import DictType
from gym import spaces

from openai_vpt.lib.action_mapping import CameraHierarchicalMapping
from openai_vpt.lib.actions import ActionTransformer
from openai_vpt.lib.torch_util import default_device_type, set_default_torch_device
from openai_vpt.lib.action_head import make_action_head
from gym3.types import DictType
from openai_vpt.lib.tree_util import tree_map
from action_utils import cache_process

# Hardcoded settings
AGENT_RESOLUTION = (128, 128)

POLICY_KWARGS = dict(
    attention_heads=16,
    attention_mask_style="clipped_causal",
    attention_memory_size=256,
    diff_mlp_embedding=False,
    hidsize=2048,
    img_shape=[128, 128, 3],
    impala_chans=[16, 32, 32],
    impala_kwargs={"post_pool_groups": 1},
    impala_width=8,
    init_norm_kwargs={"batch_norm": False, "group_norm_groups": 1},
    n_recurrence_layers=4,
    only_img_input=True,
    pointwise_ratio=4,
    pointwise_use_activation=False,
    recurrence_is_residual=True,
    recurrence_type="transformer",
    timesteps=128,
    use_pointwise_layer=True,
    use_pre_lstm_ln=False,
)

PI_HEAD_KWARGS = dict(temperature=2.0)

ACTION_TRANSFORMER_KWARGS = dict(
    camera_binsize=2,
    camera_maxval=10,
    camera_mu=10,
    camera_quantization_scheme="mu_law",
)

ENV_KWARGS = dict(
    fov_range=[70, 70],
    frameskip=1,
    gamma_range=[2, 2],
    guiscale_range=[1, 1],
    resolution=[640, 360],
    cursor_size_range=[16.0, 16.0],
)

TARGET_ACTION_SPACE = {
    "ESC": spaces.Discrete(2),
    "attack": spaces.Discrete(2),
    "back": spaces.Discrete(2),
    "camera": spaces.Box(low=-180.0, high=180.0, shape=(2,)),
    "drop": spaces.Discrete(2),
    "forward": spaces.Discrete(2),
    "hotbar.1": spaces.Discrete(2),
    "hotbar.2": spaces.Discrete(2),
    "hotbar.3": spaces.Discrete(2),
    "hotbar.4": spaces.Discrete(2),
    "hotbar.5": spaces.Discrete(2),
    "hotbar.6": spaces.Discrete(2),
    "hotbar.7": spaces.Discrete(2),
    "hotbar.8": spaces.Discrete(2),
    "hotbar.9": spaces.Discrete(2),
    "inventory": spaces.Discrete(2),
    "jump": spaces.Discrete(2),
    "left": spaces.Discrete(2),
    "pickItem": spaces.Discrete(2),
    "right": spaces.Discrete(2),
    "sneak": spaces.Discrete(2),
    "sprint": spaces.Discrete(2),
    "swapHands": spaces.Discrete(2),
    "use": spaces.Discrete(2)
}


def resize_image(img, target_resolution):
    # For your sanity, do not resize with any function than INTER_LINEAR
    img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
    return img


def l2_distance(a, b):
    return euclidean(a, b) ** 2

def one_hot_encode(actions: list, num_classes: int, add_batch_dim: bool = True):
    '''One-hot encodes the actions'''
    actions = torch.tensor(actions)
    if add_batch_dim:
        actions = actions.unsqueeze(0)
    return torch.nn.functional.one_hot(actions, num_classes=num_classes).float()

def preprocess_situation(situation, device='cuda'):
    retrieved_situation = torch.Tensor(situation['embedding']).to(device).reshape(1, 1, -1)
    retrieved_actions = {
        "camera": one_hot_encode(situation['situation_actions']['camera'], 121).to(device),
        "buttons": one_hot_encode(situation['situation_actions']['buttons'], 8641).to(device)
    }

    # print(len(situation['situation_actions']['camera']), len(situation['situation_actions']['buttons']))
    
    next_action = {
        "camera": one_hot_encode(situation['next_action']['camera'], 121).to(device),
        "buttons": one_hot_encode(situation['next_action']['buttons'], 8641).to(device)
    }

    return retrieved_situation, retrieved_actions, next_action

class Retriever:
    def __init__(self, encoder_model, encoder_weights, memory_path):
        self.vpt = VPTCNNEncoder(encoder_model, encoder_weights)
        self.vpt.eval()
        self.memory = Memory()
        self.memory.load_index(memory_path)

    def encode_query(self, query_obs):
        return self.vpt(query_obs).squeeze().cpu().numpy()

    def retrieve(self, query_obs, k=1, encode_obs=True):
        if encode_obs:
            query_obs = self.encode_query(query_obs)
        results = self.memory.search(query_obs, k=k)
        
        return results[0], query_obs


class REBECA(nn.Module):
    def __init__(self, cnn_model, trf_model, cnn_weights, trf_weights, memory_path, device='auto'):
        super().__init__()

        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.retriever = Retriever(cnn_model, cnn_weights, memory_path)
        self.vpt_cnn = VPTCNNEncoder(cnn_model, cnn_weights)
        self.vpt_cnn.eval()

        self.controller = Controller(trf_model)
        if trf_weights:
            self.controller.vpt_transformers.load_state_dict(torch.load(trf_weights, map_location=torch.device('cuda')))

        # Action processing
        set_default_torch_device(self.device)
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
        action_space = self.action_mapper.get_action_space_update()
        action_space = DictType(**action_space)
        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

        # Action head
        self.action_head = make_action_head(action_space, self.controller.hid_dim, **PI_HEAD_KWARGS).to('cuda')

    def forward(self, obs, state_in, cache=True):
    
        # extract features from observation
        obs_feats = self.vpt_cnn(obs)

        # retrieve situation from memory
        situation, _ = self.retriever.retrieve(obs_feats.to('cpu'), k=1, encode_obs=False)

        # process retrieved situations
        situation_embed, situation_actions, next_action = preprocess_situation(situation, self.device)
        
        if not cache:
            # forward pass through controller
            latent, state_out = self.controller(obs_feats, situation_embed, situation_actions, next_action, state_in)

            # get action logits
            action_logits = self.action_head(latent)

            return action_logits, state_out
        else:
            return cache_process(obs, state_in)

    def _env_action_to_agent(self, minerl_action_transformed, to_torch=False, check_if_null=False):
        """
        Turn action from MineRL to model's action.

        Note that this will add batch dimensions to the action.
        Returns numpy arrays, unless `to_torch` is True, in which case it returns torch tensors.

        If `check_if_null` is True, check if the action is null (no action) after the initial
        transformation. This matches the behaviour done in OpenAI's VPT work.
        If action is null, return "None" instead
        """
        minerl_action = self.action_transformer.env2policy(minerl_action_transformed)
        if check_if_null:
            if np.all(minerl_action["buttons"] == 0) and np.all(minerl_action["camera"] == self.action_transformer.camera_zero_bin):
                return None

        # Add batch dims if not existant
        if minerl_action["camera"].ndim == 1:
            minerl_action = {k: v[None] for k, v in minerl_action.items()}
        action = self.action_mapper.from_factored(minerl_action)
        if to_torch:
            action = {k: torch.from_numpy(v).to(self.device) for k, v in action.items()}
        return action

    def get_logprob_of_action(self, pred, action):
        """
        Get logprob of taking action `action` given probability distribution
        (see `get_gradient_for_action` to get this distribution)
        """
        ac = tree_map(lambda x: x.unsqueeze(1), action)
        log_prob = self.action_head.logprob(ac, pred)
        assert not th.isnan(log_prob).any()
        return log_prob[:, 0]