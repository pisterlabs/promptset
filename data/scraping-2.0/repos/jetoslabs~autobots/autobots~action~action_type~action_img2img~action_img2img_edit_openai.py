from pathlib import Path
from typing import Type, Optional, Literal

from openai.types import ImagesResponse
from pydantic import BaseModel
from pydantic_core import Url

from autobots.action.action_type.abc.IAction import IAction, ActionConfigType, ActionInputType, ActionOutputType
from autobots.action.action_type.action_types import ActionType
from autobots.conn.openai.openai_client import get_openai
from autobots.conn.openai.openai_images.image_model import ImageEdit


class ImageEditInput(BaseModel):
    image: Url | Path | bytes | None = None
    prompt: str | None = None
    mask: Path | Url | bytes | None = None
    size: Optional[Literal["256x256", "512x512", "1024x1024"]] = None


class ActionImg2ImgEditOpenai(IAction[ImageEdit, ImageEditInput, ImagesResponse]):
    type = ActionType.img2img_edit_openai

    @staticmethod
    def get_config_type() -> Type[ActionConfigType]:
        return ImageEdit

    @staticmethod
    def get_input_type() -> Type[ActionInputType]:
        return ImageEditInput

    @staticmethod
    def get_output_type() -> Type[ActionOutputType]:
        return ImagesResponse

    def __init__(self, action_config: ImageEdit):
        super().__init__(action_config)

    async def run_action(self, action_input: ImageEditInput) -> ImagesResponse:
        if action_input.prompt:
            self.action_config.prompt = f"{self.action_config.prompt} {action_input.prompt}"
        if action_input.image:
            self.action_config.image = action_input.image
        if action_input.mask:
            self.action_config.mask = action_input.mask
        if action_input.size:
            self.action_config.size = action_input.size

        images = await get_openai().openai_images.create_image_edit(self.action_config)
        return images
