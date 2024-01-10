from pathlib import Path
from typing import Type, Optional, Literal

from openai.types import ImagesResponse
from pydantic import BaseModel
from pydantic_core import Url

from autobots.action.action_type.abc.IAction import IAction, ActionConfigType, ActionInputType, ActionOutputType
from autobots.action.action_type.action_types import ActionType
from autobots.conn.openai.openai_client import get_openai
from autobots.conn.openai.openai_images.image_model import ImageCreateVariation


class ImageVariationInput(BaseModel):
    image: Url | Path | bytes | None = None
    size: Optional[Literal["256x256", "512x512", "1024x1024"]] = None


class ActionImg2ImgVariationOpenai(IAction[ImageCreateVariation, ImageVariationInput, ImagesResponse]):
    type = ActionType.img2img_variation_openai

    @staticmethod
    def get_config_type() -> Type[ActionConfigType]:
        return ImageCreateVariation

    @staticmethod
    def get_input_type() -> Type[ActionInputType]:
        return ImageVariationInput

    @staticmethod
    def get_output_type() -> Type[ActionOutputType]:
        return ImagesResponse

    def __init__(self, action_config: ImageCreateVariation):
        super().__init__(action_config)

    async def run_action(self, action_input: ImageVariationInput) -> ImagesResponse:
        if action_input.image:
            self.action_config.image = action_input.image
        if action_input.size:
            self.action_config.size = action_input.size

        images = await get_openai().openai_images.create_image_variation(self.action_config)
        return images
