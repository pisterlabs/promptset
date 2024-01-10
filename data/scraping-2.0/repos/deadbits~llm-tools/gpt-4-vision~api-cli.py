import os
import base64
import argparse
import logging

from typing import List, Union, Any
from pydantic import BaseModel, ValidationError
from openai import OpenAI, OpenAIError
from loguru import logger


class ImageURLContent(BaseModel):
    type: str = 'image_url'
    image_url: str


class ImageBase64Content(BaseModel):
    type: str = 'image_url'
    image_base64: str


class TextContent(BaseModel):
    type: str = 'text'
    text: str


ContentItem = Union[TextContent, ImageURLContent, ImageBase64Content]


class UserMessage(BaseModel):
    role: str = "user"
    content: List[ContentItem]


class VisionAPI:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.max_tokens = 300
        self.history = []

    def encode_image_to_base64(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            logger.error(f'image path not found: {image_path}')
            return None
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def submit_request(self, user_message: UserMessage) -> Any:
        try:
            formatted_content = []
            for content_item in user_message.content:
                if isinstance(content_item, TextContent):
                    formatted_content.append(
                        {
                            "type": "text",
                            "text": content_item.text
                        }
                    )

                elif isinstance(content_item, ImageURLContent):
                    formatted_content.append(
                        {
                            "type": "image_url",
                            "image_url": content_item.image_url
                        }
                    )
                elif isinstance(content_item, ImageBase64Content):
                    formatted_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{content_item.image_base64}"
                            },
                        }
                    )

            messages_payload = [
                {
                    "role": "user",
                    "content": formatted_content
                }
            ]

            logger.debug(messages_payload)

            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=messages_payload,
                max_tokens=self.max_tokens
            )

            self.history.append({
                "request": messages_payload,
                "response": response.choices[0].message.content
            })

            return response.choices[0].message.content

        except ValidationError as err:
            logger.error(f"Validation Error: {err.json()}")
            raise
        except OpenAIError as err:
            logger.error(f"OpenAI Error: {str(err)}")
            raise
        except Exception as err:
            logger.error(f"An error occurred: {str(err)}")
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images with GPT-4 Vision.')

    parser.add_argument(
        '--api_key',
        type=str,
        required=True,
        help='OpenAI API Key'
    )

    parser.add_argument(
        '--text',
        type=str,
        help='Text prompt to send along with the images'
    )

    parser.add_argument(
        '--image_paths',
        nargs='*',
        help='Paths to local images to encode and submit'
    )

    parser.add_argument(
        '--image_urls',
        nargs='*',
        help='URLs of images to submit'
    )

    args = parser.parse_args()

    vision_api = VisionAPI(api_key=args.api_key)

    content_list: List[ContentItem] = []

    if args.text:
        content_list.append(TextContent(text=args.text))

    if args.image_paths:
        for path in args.image_paths:
            base64_image = vision_api.encode_image_to_base64(path)
            content_list.append(ImageBase64Content(image_base64=base64_image))

    if args.image_urls:
        for url in args.image_urls:
            content_list.append(ImageURLContent(type="image_url", image_url=url))

    if not content_list:
        raise ValueError("At least one image path or URL must be provided.")

    user_message = UserMessage(content=content_list)

    try:
        result = vision_api.submit_request(user_message)
        print(result)
    except Exception as err:
        logging.error(f"An error occurred: {str(err)}")
