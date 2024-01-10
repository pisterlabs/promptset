from typing import Optional, Type

from openai.types import ImagesResponse

from autobots.action.action_type.abc.IAction import IAction, ActionOutputType, ActionInputType, ActionConfigType
from autobots.action.action.action_doc_model import ActionCreate
from autobots.action.action_type.action_types import ActionType
from autobots.action.action.common_action_models import TextObj
from autobots.conn.openai.openai_images.image_model import ImageReq
from autobots.conn.openai.openai_client import get_openai


class ActionCreateGenImageDalleOpenai(ActionCreate):
    type: ActionType = ActionType.text2img_dalle_openai
    config: ImageReq
    input: Optional[TextObj] = None
    output: Optional[ImagesResponse] = None


class ActionGenImageDalleOpenAiV2(IAction[ImageReq, TextObj, ImagesResponse]):
    type = ActionType.text2img_dalle_openai

    @staticmethod
    def get_config_type() -> Type[ActionConfigType]:
        return ImageReq

    @staticmethod
    def get_input_type() -> Type[ActionInputType]:
        return TextObj

    @staticmethod
    def get_output_type() -> Type[ActionOutputType]:
        return ImagesResponse

    def __init__(self, action_config: ImageReq):
        super().__init__(action_config)

    async def run_action(self, action_input: TextObj) -> ImagesResponse:
        self.action_config.prompt = f"{self.action_config.prompt}\n{action_input.text}"
        images = await get_openai().openai_images.create_image(self.action_config)
        return images
