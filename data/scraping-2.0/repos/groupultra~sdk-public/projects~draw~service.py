# service.py
import asyncio
import json
import copy
import re
import os
import random
import base64
from datetime import datetime
from io import BytesIO

from loguru import logger
from openai import AsyncOpenAI
from PIL import Image

from moobius import MoobiusService, MoobiusStorage


class DrawService(MoobiusService):
    def __init__(self, log_file="logs/service.log", error_log_file="logs/error.log", **kwargs):
        super().__init__(**kwargs)
        self.log_file = log_file
        self.error_log_file = error_log_file

        self.default_status = {
            "total_score": 0,
        }

        self.features = []

        self.about = '<about>'
        self.welcome = '<welcome>'
        self.image_dir = 'temp/'

        self.openai_client = None

    async def on_start(self):
        # ==================== load features and fill in the template ====================
        logger.add(self.log_file, rotation="1 day", retention="7 days", level="DEBUG")
        logger.add(self.error_log_file, rotation="1 day", retention="7 days", level="ERROR")

        self.openai_client = AsyncOpenAI()

        os.makedirs(self.image_dir, exist_ok=True)

        with open('resources/draw_features.json', 'r', encoding='utf-8') as f:
            self.features = json.load(f)

        with open('resources/about.txt', 'r', encoding='utf-8') as f:
            self.about = f.read()

        with open('resources/welcome.txt', 'r', encoding='utf-8') as f:
            self.welcome = f.read()

        # ==================== initialize the database ====================

        for channel_id in self.channels:
            self.bands[channel_id] = MoobiusStorage(self.service_id, channel_id, db_config=self.db_config)

            real_characters = await self.fetch_real_characters(channel_id)

            for character in real_characters:
                character_id = character.user_id
                self.bands[channel_id].real_characters[character_id] = character

                if character_id not in self.bands[channel_id].status:
                    self.bands[channel_id].status[character_id] = copy.deepcopy(self.default_status)
                else:
                    pass

            if 'prompt' not in self.bands[channel_id].current:
                self.bands[channel_id].current['prompt'] = 'Social'
            else:
                pass

            if 'image_url' not in self.bands[channel_id].current:
                self.bands[channel_id].current['image_url'] = 'https://frog.4fun.chat/dog-whistle/images/social-logo.png'
            else:
                pass

            if 'author' not in self.bands[channel_id].current:
                self.bands[channel_id].current['author'] = '🐱eow'
            else:
                pass
            
            # ====================== upload avatars ======================

            names = {
                'Painter': 'Picasso Sillica',
                'AI': 'Let me see!',
            }

            for name in names:
                if name not in self.bands[channel_id].virtual_characters:
                    logger.info(f'Uploading avatar {name}...')
                    file_name = f'resources/icons/{name}.jpg'
                    avatar_uri = self.http_api.upload_file(file_name)
                    self.bands[channel_id].avatars[name] = avatar_uri

                    character = self.http_api.create_service_user(self.service_id, names[name], name, avatar_uri, f'I am a {names[name]}')
                    self.bands[channel_id].virtual_characters[name] = character
                else:
                    pass

    @staticmethod
    def filter(content, prompt):
        safe_content = content.replace(prompt, '𠓗' * len(prompt))
        final_content = ''.join([(c if safe_content[i] != '𠓗' else '𠓗') for i, c in enumerate(content)])
        final_content = re.sub(r'𠓗+', '<b>[REDACTED]</b>', final_content)

        return final_content

    async def on_msg_up(self, msg_up):
        channel_id = msg_up.channel_id
        sender = msg_up.sender

        recipients = msg_up.recipients

        if msg_up.subtype == "text":
            raw_content = msg_up.content['text'].strip()
            content = raw_content.lower()

            if content == 'draw':
                say = 'Please describe what you want to draw with a word of no more than 20 characters. For example, try "draw banana".'
                await self._send_msg(channel_id, say, [sender], sent_by='Painter')

            elif content.startswith('draw '):
                prompt = content[5:].strip()

                if re.search(r'\s', prompt):
                    say = f'Please use just one word to describe what you want to draw. For example, try "draw banana" instead of "draw a banana".'
                    await self._send_msg(channel_id, say, [sender], sent_by='Painter')
                elif len(prompt) > 20:
                    say = f'Please describe what you want to draw with a word of no more than 20 characters.'
                    await self._send_msg(channel_id, say, [sender], sent_by='Painter')
                else:
                    say = 'I am drawing... please wait a moment...'
                    await self._send_msg(channel_id, say, [sender], sent_by='Painter')

                    try:
                        # image_url, revised_prompt = await self._draw(prompt, response_format='url')
                        image_url, revised_prompt = await self._draw(prompt, response_format='local')

                        last_prompt = self.bands[channel_id].current['prompt']
                        say = f'Last image: <b>{last_prompt}</b>'
                        await self._send_msg(channel_id, say, recipients, sent_by='Painter')

                        await asyncio.sleep(0.5)

                        self.bands[channel_id].current['prompt'] = prompt
                        self.bands[channel_id].current['image_url'] = image_url
                        nickname = self.bands[channel_id].real_characters[sender].user_context.nickname
                        self.bands[channel_id].current['author'] = sender

                        await self._send_playground_image(channel_id, image_url, recipients)
                        
                        say = f'I finished my drawing: {revised_prompt}'
                        await self._send_msg(channel_id, say, [sender], sent_by='Painter')

                        await asyncio.sleep(0.5)

                        say = f'{nickname} has finished a new drawing on the Stage. Please guess what they want to draw. Note: The image will expire in an hour. Please save it if you like it!'
                        await self._send_msg(channel_id, say, recipients, sent_by='Painter')

                        await asyncio.sleep(0.5)

                        await self._send_msg(channel_id, message_content=image_url, recipients=recipients, subtype='image', sent_by='Painter')

                    except Exception as e:
                        await self._send_msg(channel_id, f'Sorry! I cannot draw this!\n\n{e}', [sender], sent_by='Painter')
                    
            elif self.bands[channel_id].current['prompt'] in content:
                prompt = self.bands[channel_id].current['prompt']
                final_content = self.filter(content, prompt)
                
                new_recipients = [r for r in recipients if r != sender]

                await self._send_msg(channel_id, final_content, new_recipients, sent_by=sender, virtual=False)

                await asyncio.sleep(0.5)

                nickname = self.bands[channel_id].real_characters[sender].user_context.nickname
                
                if sender not in recipients:
                    recipients.append(sender)
                else:
                    pass
                
                await self._send_msg(channel_id, f'Congratulations, {nickname} guessed the drawing!', recipients, sent_by='Painter')

            else:
                await self.send(payload_type='msg_down', payload_body=msg_up)
        else:
            await self.send(payload_type='msg_down', payload_body=msg_up)

    async def _send_msg(self, channel_id, message_content, recipients, subtype='text', sent_by='Painter', virtual=True):
        """
        Send system message.
        """

        if virtual:
            sender = self.bands[channel_id].virtual_characters[sent_by].user_id
        else:
            sender = sent_by

        await self.send_msg_down(
            channel_id=channel_id,
            recipients=recipients,
            subtype=subtype,
            message_content=message_content,
            sender=sender
        )

    async def _send_playground_image(self, channel_id, image_path, recipients):
        """
        Send playground image.
        """
        content = {"path": image_path}
        await self.send_update_playground(channel_id, content, recipients)

    async def _see(self, image_url, prompt=None):
        """
        See the image.
        """

        prompt = prompt or 'What is in the image?'

        response = await self.openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        
                        {
                            "type": "image_url",
                            
                            "image_url": 
                            {
                                "url": image_url,
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )

        return response.choices[0].message.content

    async def _draw(self, prompt, response_format='url'):
        """
        Draw the image.
        """

        # 'revised_prompt', 'b64_json' (str or None if url), 'url'
        if response_format == 'url':
            response = await self.openai_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
                response_format="url"
            )

            image_url = response.data[0].url
            revised_prompt = response.data[0].revised_prompt

            return image_url, revised_prompt
        else:
            response = await self.openai_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
                response_format="b64_json"
            )

            image_b64 = response.data[0].b64_json
            revised_prompt = response.data[0].revised_prompt

            b64 = image_b64
            image = Image.open(BytesIO(base64.b64decode(b64)))
            file_name = f'{datetime.now().strftime("%Y%m%d%H%M%S")}_{random.randint(10000, 99999)}.png'
            file_path = os.path.join(self.image_dir, file_name)
            
            image.save(file_path)
            image_url = self.http_api.upload_file(file_path)

            return image_url, revised_prompt

    async def on_action(self, action):
        """
        Handle the received action.
        """
        sender = action.sender
        channel_id = action.channel_id

        if action.subtype == "fetch_userlist":
            real_characters = list(self.bands[channel_id].real_characters.values())
            virtual_characters = list(self.bands[channel_id].virtual_characters.values())
            user_list = virtual_characters + real_characters

            await self.send_update_user_list(channel_id, user_list, [sender])

        elif action.subtype == "fetch_features":
            await self.send_update_features(action.channel_id, self.features, [action.sender])

        elif action.subtype == "fetch_playground":
            image_url = self.bands[channel_id].current['image_url']
            await self._send_playground_image(channel_id, image_url, [sender])

            content = [
                {
                    "widget": "playground",
                    "display": "visible",
                    "expand": "true"
                }
            ]
            
            await self.send_update_style(channel_id, content, [sender])

        elif action.subtype == "join_channel":
            character = self.http_api.fetch_user_profile(sender)
            self.bands[channel_id].real_characters[sender] = character
            self.bands[channel_id].status[sender] = copy.deepcopy(self.default_status)

            real_characters = list(self.bands[channel_id].real_characters.values())
            virtual_characters = list(self.bands[channel_id].virtual_characters.values())
            user_list = virtual_characters + real_characters
            character_ids = list(self.bands[channel_id].real_characters.keys())

            await self.send_update_user_list(channel_id, user_list, character_ids)
            await self._send_msg(channel_id, f'{character.user_context.nickname} joined the band!', character_ids, sent_by='Painter')

            await asyncio.sleep(0.5)

            await self._send_msg(channel_id, self.welcome, [sender])

        elif action.subtype == "leave_channel":
            character = self.bands[channel_id].real_characters.pop(sender, None)

            self.bands[channel_id].status.pop(sender, None)

            real_characters = self.bands[channel_id].real_characters
            virtual_characters = self.bands[channel_id].virtual_characters
            user_list = list(virtual_characters.values()) + list(real_characters.values())
            character_ids = list(real_characters.keys())

            await self.send_update_user_list(channel_id, user_list, character_ids)
            await self._send_msg(channel_id, f'{character.user_context.nickname} left the band!', character_ids, sent_by='Painter')

        elif action.subtype == "fetch_channel_info":
            logger.info("fetch_channel_info")
            """
            await self.send_update_channel_info(channel_id, self.db_helper.get_channel_info(channel_id))
            """
        else:
            logger.warning("Unknown action subtype:", action.subtype)

    async def on_feature_call(self, feature_call):
        """
        Handle the received feature call.
        """
        sender = feature_call.sender
        channel_id = feature_call.channel_id
        feature_id = feature_call.feature_id

        if feature_id == "Ask":
            prompt = 'What is in the image?'
            image_url = self.bands[channel_id].current['image_url']

            try:
                await self._send_msg(channel_id, f'I am trying to tell you what is in this image, please wait...', [sender], sent_by='AI')
                answer = await self._see(image_url, prompt)
                answer = self.filter(answer, self.bands[channel_id].current['prompt'])

                await self._send_msg(channel_id, f'{answer}', [sender], sent_by='AI')
            except Exception as e:
                logger.warning(e)
                await self._send_msg(channel_id, f'Sorry! I cannot see the image!\n\n{e}', [sender], sent_by='AI')

        elif feature_id == "About":
            await self._send_msg(channel_id, self.about, [sender], sent_by='Painter')

        else:
            pass

    async def on_unknown_message(self, message_data):
        """
        Handle the received unknown message.
        """
        logger.warning("Received unknown message:", message_data)
