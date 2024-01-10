import cv2
import gym
import numpy as np
import torch
import torchvision.transforms as T

from drl.deepq.game import Action
from drl.openai.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, FireResetEnv, ClipRewardEnv
from drl.openai.game_openai import OpenAIGame


class Pong(OpenAIGame):
  actions = [Action('up', 2, 0),
             Action('down', 3, 1)]

  def __init__(self, x: int, y: int, t: int, skip=4,
               store_frames_as: torch.dtype = torch.float, force_scale_01: bool = False):
    self.transform = T.Compose([T.ToPILImage(), T.Resize((y, x)), T.Grayscale(), T.ToTensor()])
    self.scale = 255 if not store_frames_as.is_floating_point else None
    scale2 = 255 if not store_frames_as.is_floating_point and force_scale_01 else None
    super().__init__('PongNoFrameskip-v4', t, store_frames_as, scale2)
    self.env = MaxAndSkipEnv(self.env, skip=skip)

  @property
  def name(self) -> str:
    return 'pong'

  def _get_frame(self, env_state):
    raw = env_state[..., :3]
    image = self.transform(raw)
    if self.scale is not None:
      image = image * self.scale
    return image.squeeze(0)


class Pong30Min(OpenAIGame):
  actions = [Action('up', 2, 0),
             Action('down', 3, 1)]

  def __init__(self, store_frames_as: torch.dtype = torch.float):
    super().__init__('PongNoFrameskip-v4', 4, store_frames_as, None)
    env = self.env
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
      env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = ClipRewardEnv(env)
    self.env = env

  @property
  def name(self) -> str:
    return 'pong'

  def _get_frame(self, env_state):
    return torch.tensor(env_state).squeeze(0).float()


class ProcessFrame84(gym.ObservationWrapper):
  def __init__(self, env=None):
    super(ProcessFrame84, self).__init__(env)
    self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

  def observation(self, obs):
    return ProcessFrame84.process(obs)

  @staticmethod
  def process(frame):
    if frame.size == 210 * 160 * 3:
      img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    elif frame.size == 250 * 160 * 3:
      img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
    else:
      assert False, "Unknown resolution."
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
    x_t = resized_screen[18:102, :]
    x_t = np.reshape(x_t, [84, 84, 1])
    return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
  """
  Change image shape to CWH
  """

  def __init__(self, env):
    super(ImageToPyTorch, self).__init__(env)
    old_shape = self.observation_space.shape
    self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                            dtype=np.float32)

  def observation(self, observation):
    return np.swapaxes(observation, 2, 0)
