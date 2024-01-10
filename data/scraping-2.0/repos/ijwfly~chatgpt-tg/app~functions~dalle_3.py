from enum import Enum
from typing import Optional

import httpx
from pydantic import Field

from app.bot.utils import send_photo
from app.context.dialog_manager import DialogUtils
from app.functions.base import OpenAIFunction, OpenAIFunctionParams
from app.openai_helpers.utils import OpenAIAsync


class ImageSize(str, Enum):
    size_1024x1024 = "1024x1024"
    size_1024x1792 = "1024x1792"
    size_1792x1024 = "1792x1024"


class GenerateImageDalle3Params(OpenAIFunctionParams):
    image_prompt: str = Field(..., description="detailed tailored prompt to generate image from (translated to english, if needed)")
    image_size: ImageSize = Field(ImageSize.size_1024x1024, description="image size to generate")


class GenerateImageDalle3(OpenAIFunction):
    PARAMS_SCHEMA = GenerateImageDalle3Params

    @staticmethod
    async def download_image(url):
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise Exception(f'Image download failed with status code {resp.status_code}')
            return resp.content

    async def run(self, params: GenerateImageDalle3Params) -> Optional[str]:
        model = "dall-e-3"
        try:
            resp = await OpenAIAsync.instance().images.generate(
                model=model,
                prompt=params.image_prompt,
                size=params.image_size,
                quality="standard",
                n=1,
            )
            await self.db.create_image_generation_usage(self.user.id, model, params.image_size)

            image_url = resp.data[0].url
            image_bytes = await self.download_image(image_url)

            caption = 'Image generated from prompt:\n'
            caption += params.image_prompt
            caption += '\n\nRevised prompt:\n'
            caption += resp.data[0].revised_prompt

            # truncate caption to 1024 symbols, telegram limit
            tg_caption = caption[:1021] + '...' if len(caption) > 1024 else caption

            response = await send_photo(self.message, image_bytes, tg_caption)
            text = caption + '\n\nImage:\n<image.png>'
            dialog_message = DialogUtils.prepare_function_response(self.get_name(), text)
            await self.context_manager.add_message(dialog_message, response.message_id)
            return None
        except Exception as e:
            return f"Error: {e}"

    @classmethod
    def get_name(cls) -> str:
        return "generate_image_dalle_3"

    @classmethod
    def get_description(cls) -> str:
        return "Use dalle-3 to generate image from prompt. Image prompt must be in english. Generate tailored detailed prompt for user idea."

    @classmethod
    def get_system_prompt_addition(cls) -> Optional[str]:
        return "If user asks you to generate image, use some imagination and write a long 500 symbols good tailored detailed prompt for Dall-E 3 model and call function. Don't generate image if user didn't ask you to do so directly."

