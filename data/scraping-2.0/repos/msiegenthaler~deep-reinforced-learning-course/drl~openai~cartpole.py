from torch import Tensor

from drl.deepq.game import Action
from drl.openai.game_openai import OpenAIFrameGame
import torchvision.transforms as T


class CartPoleVisual(OpenAIFrameGame):
  actions = [Action('left', 0, 0),
             Action('right', 1, 1)]

  def __init__(self, x: int, y: int, t: int):
    self.transform = T.Compose([T.ToPILImage(), T.Resize((y, x)), T.Grayscale(), T.ToTensor()])
    super().__init__('CartPole-v0', t)

  @property
  def name(self) -> str:
    return 'cardpole'

  def _get_frame(self, env_state) -> Tensor:
    image = self.transform(self._get_raw_frame()[330:660, 0:1200, :])
    return image.squeeze(0)
