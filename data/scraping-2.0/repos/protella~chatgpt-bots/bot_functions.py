from openai import OpenAI
import os
from dotenv import load_dotenv
from textwrap import dedent
import base64
from io import BytesIO
from PIL import Image
from copy import deepcopy


load_dotenv()  # load auth tokens from .env file

# Default models: https://platform.openai.com/docs/models
GPT_MODEL = 'gpt-4-1106-preview'
DALLE_MODEL = 'dall-e-3'
GPT_VISION_MODEL = 'gpt-4-vision-preview'


class ChatBot:
    def __init__(self, INITIALIZE_TEXT, streaming_client=False):
        self.messages = [INITIALIZE_TEXT]
        self.INITIALIZE_TEXT = INITIALIZE_TEXT
        self.streaming_client = streaming_client  # ToDo: Implement streaming support
        self.usage = {}
        self.processing = False
        self.config_option_defaults = {
            'temperature': 0.5,  # 0.0 - 2.0
            'top_p': 1,
            'max_tokens': 2048,  # max 4096
            'custom_init': '',
            'gpt_model': GPT_MODEL,
            'dalle_model': DALLE_MODEL,
            'gpt_vision_model': GPT_VISION_MODEL,
            'size': '1024x1024',  # 1024x1024, 1024x1792 or 1792x1024
            'quality': 'hd',  # standard or hd
            'style': 'vivid',  # natural or vivid
            'number': 1,  # number of images. Only 1 supported for Dalle3
            'detail': 'auto'  # vision parameter: auto, low, high
        }
        self.current_config_options = self.config_option_defaults.copy()
        self.client = OpenAI(api_key=os.environ['OPENAI_KEY'])

    def chat_context_mgr(self, message_text, files=''):
        self.processing = True
        self.messages.append({'role': 'user', 'content': message_text})

        # Using vision model for all chat prompts since images passed to non-vision model throws an error.
        # GPT4v is an extension of GPT4 with all the same functions and features.
        gpt_output = self.get_gpt_response(
            self.messages, self.current_config_options['gpt_vision_model'])
        self.processing = False

        if hasattr(gpt_output, 'role'):
            is_error = False
            if gpt_output.role == 'assistant':
                self.messages.append(
                    {'role': 'assistant', 'content': gpt_output.content})

            return gpt_output.content, is_error
        else:
            is_error = True
            self.messages.pop()
            return gpt_output, is_error

    def image_context_mgr(self, message_text):
        self.processing = True
        self.messages.append(
            {'role': 'user', 'content': f'Dalle3 User Prompt: {message_text}'})
        image, revised_prompt = self.get_dalle_response(
            message_text, self.current_config_options['dalle_model'])
        self.processing = False

        if revised_prompt:
            is_error = False
            self.messages.append(
                {'role': 'assistant', 'content': dedent(
                    f"""\
                        Based on the DALL-E 3 prompt, I can imagine an image like this: {revised_prompt} 
                        While I can't create or view images,
                        I can describe what such an image might look like or the ideas it might convey.
                        I will act as if I actually created the described image."""
                )})

        else:
            is_error = True
            self.messages.pop()
            image = None
            revised_prompt = None

        return image, revised_prompt, is_error

    def vision_context_mgr(self, message_text, images):
        self.processing = True
        if not message_text:
            message_text = ''
        multi_part_msg = {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': f'{message_text}'
                }
            ]
        }

        for image in images:
            new_image_element = {
                'type': 'image_url',
                'image_url': {
                    'url': f'data:image/png;base64,{image}',
                    'detail': self.current_config_options['detail']
                }
            }
            multi_part_msg['content'].append(new_image_element)

        self.messages.append(multi_part_msg)
        gpt_output = self.get_gpt_response(
            self.messages, self.current_config_options['gpt_vision_model'])
        self.processing = False

        if hasattr(gpt_output, 'role'):
            is_error = False
            if gpt_output.role == 'assistant':
                self.messages.append(
                    {'role': 'assistant', 'content': gpt_output.content})

            return gpt_output.content, is_error
        else:
            is_error = True
            self.messages.pop()
            return gpt_output, is_error

    def get_dalle_response(self, image_prompt, model):
        try:
            response = self.client.images.generate(
                model=model,
                prompt=image_prompt,
                size=self.current_config_options['size'],
                quality=self.current_config_options['quality'],
                style=self.current_config_options['style'],
                n=1,  # Value of 1 is the only value supported in DALLE-3 as of now
                response_format='b64_json'
            )

            image_binary = base64.b64decode(response.data[0].b64_json)
            image_object = BytesIO(image_binary)
            revised_prompt = response.data[0].revised_prompt
            # Convert from webP format to PNG
            with Image.open(image_object) as webp_image:
                png_image = BytesIO()

                webp_image.save(png_image, 'PNG')
                png_image.seek(0)

                return png_image, revised_prompt

        except Exception as e:
            print(f'##################\n{e}\n##################')
            return None, e

    def get_vision_response(self, messages_history):
        # Not needed? Can use existing gpt response function below?
        pass

    def get_gpt_response(self, messages_history, model):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages_history,
                stream=self.streaming_client,
                temperature=float(self.current_config_options['temperature']),
                max_tokens=int(self.current_config_options['max_tokens']),
                top_p=float(self.current_config_options['top_p']),

            )
            self.usage = response.usage
            return response.choices[0].message

        except Exception as e:
            print(f'##################\n{e}\n##################')
            return e

    def usage_command(self):
        if self.usage:
            return dedent(
                f'''\
                Cumulative Token stats since last reset:
                Prompt Tokens: {self.usage.prompt_tokens}
                Completion Tokens: {self.usage.completion_tokens}
                Total Tokens: {self.usage.total_tokens}'''
            )

        else:
            return 'No usage info yet. Ask the bot something and check again.'

    def history_command(self):
        # We don't want to display the b64 encoded images that may be present in the history, so replace them with placeholder text.
        # This is done in a separate instance of the message history so that the images remain for the bot to analyze in future conversations.
        display_history = deepcopy(self.messages)
        for message in display_history:
            if 'content' in message and isinstance(message['content'], list):
                for content_item in message['content']:
                    if isinstance(content_item, dict) and content_item.get('type') == 'image_url':
                        # Replace image data with placeholder text
                        content_item['image_url'] = {
                            'url': 'Image content not displayed'}

        return display_history

    def set_config(self, setting, value):
        if setting in self.current_config_options:
            self.current_config_options[setting] = value
            return f'Updated {setting} to {value}'
        return f'Unknown setting: {setting}'

    def view_config(self):
        return '\n'.join(
            f'{setting}: {value}'
            for setting, value in self.current_config_options.items()
        )

    def reset_history(self):
        self.messages = [self.INITIALIZE_TEXT]
        self.usage = {}
        self.processing = False

        return 'Rebooting. Beep Beep Boop. My memory has been wiped!'

    def reset_config(self):
        self.current_config_options = self.config_option_defaults

        return 'Configuration Defaults Reset!'

    @staticmethod
    def help_command():
        return dedent(
            """\
            !help - This help.
            /dalle-3 {prompt} generate an image via text with Dalle-3.
            !config - Displays the current configuration values.
            !config [option] [value] - Sets one of the options seen in '!config' to a custom value. Beware of the model's ranges for these values.
              --see https://platform.openai.com/docs/api-reference for more info.
            !history - Prints a json dump of the chat history since last reset.
            !reset config - Sets the config options back to defaults, e.g., temperature, max_tokens, etc.
            !reset history - Clears the bots memory and resets the context to the default as configured in this script. (Always the first line of the '!history' output.)
            !usage - Prints token usage stats since the last reset."""
        )

    @staticmethod
    def handle_busy():
        return "I'm busy processing a previous request, please wait a moment and try again."
