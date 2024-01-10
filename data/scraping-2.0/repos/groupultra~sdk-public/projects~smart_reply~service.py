# service.py
import json
from loguru import logger
from openai import AsyncOpenAI
from moobius import MoobiusService, MoobiusStorage
from pprint import pprint
import deepl
import os
import os
import json
import asyncio
from pprint import pprint
from typing import List, Dict
from openai import AsyncOpenAI
from deepl import Translator
import openai


class DemoService(MoobiusService):
    def __init__(self, log_file="logs/service.log", error_log_file="logs/error.log", **kwargs):
        super().__init__(**kwargs)
        self.personal_dialogues = {}
        self.channel_dialogues = {}
        logger.add(log_file, rotation="1 day", retention="7 days", level="DEBUG")
        logger.add(error_log_file, rotation="1 day", retention="7 days", level="ERROR")
        self.default_features = {}
        self.default_status = {}
        self.bands = {}
        self.PERSONAL_ASSISTANT = "Personal Assistant"
        self.GROUP_ASSISTANT = "Group Assistant"
        self.SIMPLE_ANSWER = "Simple Answer"
        self.model = 'gpt-3.5-turbo-1106'
        self.temperature = 0.5
        self.openai_client = None
        self.translator = None

        self.images = {
            self.PERSONAL_ASSISTANT: "resources/light.png",
            self.GROUP_ASSISTANT: "resources/light.png",
            self.SIMPLE_ANSWER: "resources/light.png"
        }

    @logger.catch
    async def get_answer_simple(self, prompt):
        try:
            completion = await self.openai_client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(e)

    async def on_start(self):
        """
        Called after successful connection to websocket server and service login success.
        """
        self.openai_client = AsyncOpenAI()
        self.translator = Translator(os.environ["DEEPL_TOKEN"])

        with open('resources/features.json', 'r') as f:
            self.default_features = json.load(f)

        for channel_id in self.channels:
            self.bands[channel_id] = MoobiusStorage(self.service_id, channel_id, db_config=self.db_config)
            real_characters = self.http_api.fetch_real_characters(channel_id, self.service_id)

            for character in real_characters:
                character_id = character.user_id
                self.bands[channel_id].real_characters[character_id] = character

                if character_id not in self.bands[channel_id].features:
                    self.bands[channel_id].features[character_id] = self.default_features
                else:
                    pass
                
                if character_id not in self.bands[channel_id].states:
                    self.bands[channel_id].states[character_id] = self.default_status
                else:
                    pass

            # DEMO: upload image
            for name in self.images:
                if name not in self.bands[channel_id].image_paths:
                    self.bands[channel_id].image_paths[name] = self.http_api.upload_file(self.images[name])
                else:
                    pass

            # DEMO: create virtual characters

            if self.PERSONAL_ASSISTANT not in self.bands[channel_id].virtual_characters:
                self.bands[channel_id].virtual_characters[self.PERSONAL_ASSISTANT] = self.http_api.create_service_user(
                    self.service_id, self.PERSONAL_ASSISTANT, self.PERSONAL_ASSISTANT, self.bands[channel_id].image_paths[self.PERSONAL_ASSISTANT], f'I am {self.PERSONAL_ASSISTANT}!'
                )
            else:
                pass

            if self.GROUP_ASSISTANT not in self.bands[channel_id].virtual_characters:
                self.bands[channel_id].virtual_characters[self.GROUP_ASSISTANT] = self.http_api.create_service_user(
                    self.service_id, self.GROUP_ASSISTANT, self.GROUP_ASSISTANT, self.bands[channel_id].image_paths[self.GROUP_ASSISTANT], f'I am {self.GROUP_ASSISTANT}!'
                )
            else:
                pass

            if self.SIMPLE_ANSWER not in self.bands[channel_id].virtual_characters:
                self.bands[channel_id].virtual_characters[self.SIMPLE_ANSWER] = self.http_api.create_service_user(
                    self.service_id, self.SIMPLE_ANSWER, self.SIMPLE_ANSWER, self.bands[channel_id].image_paths[self.SIMPLE_ANSWER], f'I am {self.SIMPLE_ANSWER}!'
                )
            else:
                pass

    async def on_msg_up(self, msg_up):
        logger.info("on_msg_up")
        logger.info(msg_up)
        if msg_up.subtype == "text":
            txt = msg_up.content['text']
            logger.info(txt)
            print(txt)
            channel_id = msg_up.channel_id
            sender = msg_up.sender
            recipients = msg_up.recipients
            logger.info(recipients)
            all_recipients = list(self.bands[channel_id].real_characters.keys())
            all_recipients_other_than_sender = all_recipients.copy()
            all_recipients_other_than_sender.remove(sender)
            personal_assistant_id = self.bands[channel_id].virtual_characters[self.PERSONAL_ASSISTANT].user_id
            logger.info(personal_assistant_id)
            group_assistant_id = self.bands[channel_id].virtual_characters[self.GROUP_ASSISTANT].user_id
            logger.info(group_assistant_id)
            simple_answer_id = self.bands[channel_id].virtual_characters[self.SIMPLE_ANSWER].user_id
            logger.info(simple_answer_id)
            
            if_personal_assistant = personal_assistant_id in recipients
            logger.info(if_personal_assistant)
            if_group_assistant = group_assistant_id in recipients
            logger.info(if_group_assistant)
            if_simple_answer = simple_answer_id in recipients
            logger.info(if_simple_answer)

            assistant_num = int(if_group_assistant) + int(if_personal_assistant) + int(if_simple_answer)
            logger.info(assistant_num)

            if ("?" in txt or "？" in txt) and set(all_recipients).issubset(set(recipients)) and assistant_num == 3:
                await self.one_time_qa(channel_id, txt, sender)
            elif set(all_recipients).isdisjoint(set(recipients)):
                logger.info("no recipient")
                if assistant_num == 1:
                    logger.info("one assistant")
                    if if_personal_assistant:
                        await self.personal_dialogue_qa(channel_id, txt, sender)
                    elif if_group_assistant:
                        await self.group_dialogue_qa(channel_id, txt, sender)
                    elif if_simple_answer:
                        await self.one_time_qa(channel_id, txt, sender)
                    else:
                        pass                    
                elif assistant_num > 1:
                    logger.info("more than one assistant")
                    await self.create_message(channel_id, "Please only choose one assistant.", [sender], sender=personal_assistant_id)
                else:
                    logger.info("no assistant")
                    if ("?" in txt or "？" in txt):
                        await self.one_time_qa(channel_id, txt, sender)
                    else:
                        await self.send(payload_type='msg_down', payload_body=msg_up)
            else:   
                await self.send(payload_type='msg_down', payload_body=msg_up)
        else:
            await self.send(payload_type='msg_down', payload_body=msg_up)

    async def one_time_qa(self, channel_id: str, question: str, sender: str):
        try:
            question_en = self.translator.translate_text(text=question, target_lang="EN-US").text
            logger.debug(question_en)
            answer = await self.get_answer_simple(question)
            logger.debug(answer)
            answer_en = self.translator.translate_text(text=answer, target_lang="EN-US").text
            logger.debug(answer_en)
            recipients_withour_sender = list(self.bands[channel_id].real_characters.keys())
            recipients_withour_sender.remove(sender)
            all_recipients = list(self.bands[channel_id].real_characters.keys())

            await self.create_message(channel_id, question, [sender], sender=sender)
            await self.create_message(channel_id, question, recipients_withour_sender, sender=sender)
            asyncio.sleep(1)
            await self.create_message(channel_id, answer, [sender], sender=self.bands[channel_id].virtual_characters[self.SIMPLE_ANSWER].user_id)
            await self.create_message(channel_id, answer, recipients_withour_sender, sender=self.bands[channel_id].virtual_characters[self.SIMPLE_ANSWER].user_id)

        except Exception as e:
            print(e)

    async def group_dialogue_qa(self, channel_id: str, question: str, sender: str):
        print("group_dialogue_qa")
        all_recipients = list(self.bands[channel_id].real_characters.keys())
        if channel_id not in self.channel_dialogues:
            self.channel_dialogues[channel_id] = []
            system_message = {
                "role": "system",
                "content": "You are a helpful assistant."
            }
            self.channel_dialogues[channel_id].append(system_message)
        
        user_message = {
            "role": "user",
            "content": question
        }

        self.channel_dialogues[channel_id].append(user_message)

        total_length = sum(len(message['content']) for message in self.channel_dialogues[channel_id])
        while total_length > 10000:
            self.channel_dialogues[channel_id].pop(1)
            total_length = sum(len(message['content']) for message in self.channel_dialogues[channel_id])

        response = await self.openai_client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=self.channel_dialogues[channel_id]
        )
        answer = response.choices[0].message.content
        print(answer)
        await self.create_message(channel_id, question, all_recipients, sender=sender)        
        await self.create_message(channel_id, answer, all_recipients, sender=self.bands[channel_id].virtual_characters[self.GROUP_ASSISTANT].user_id)

        response_message = {
            "role": "assistant",
            "content": answer
        }

        self.channel_dialogues[channel_id].append(response_message)



    async def personal_dialogue_qa(self, channel_id: str, question: str, sender: str):
        print("personal_dialogue_qa")
        if sender not in self.personal_dialogues:
            self.personal_dialogues[sender] = []  # 如果用户的对话历史不存在，则创建一个空列表
            system_message = {
                "role": "system",
                "content": "You are a helpful assistant."
            }
            self.personal_dialogues[sender].append(system_message)


        user_message = {
            "role": "user",
            "content": question
        }
        self.personal_dialogues[sender].append(user_message)
        
        total_length = sum(len(message['content']) for message in self.personal_dialogues[sender])
        while total_length > 10000:
            self.personal_dialogues[sender].pop(1)
            total_length = sum(len(message['content']) for message in self.personal_dialogues[sender])

        response = await self.openai_client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=self.personal_dialogues[sender]
        )
        answer = response.choices[0].message.content
        print(answer)

        await self.create_message(channel_id, answer, [sender], sender=self.bands[channel_id].virtual_characters[self.PERSONAL_ASSISTANT].user_id)

        response_message = {
            "role": "assistant",
            "content": answer
        }

        self.personal_dialogues[sender].append(response_message)

    async def on_fetch_user_list(self, action):
        await self.calculate_and_update_user_list_from_database(action.channel_id, action.sender)

    async def on_fetch_features(self, action):
        await self.send_features_from_database(action.channel_id, action.sender)

    async def on_fetch_playground(self, action):
        channel_id = action.channel_id
        sender = action.sender

        content = [
            {
                "widget": "playground",
                "display": "invisible",
                "expand": "false"
            }
        ]
        
        await self.send_update_style(channel_id, content, [sender])
        
    # DEMO: a typical join channel handler
    async def on_join_channel(self, action):
        character_id = action.sender
        channel_id = action.channel_id
        character = self.http_api.fetch_user_profile(character_id)
        nickname = character.user_context.nickname

        self.bands[action.channel_id].real_characters[character_id] = character
        self.bands[action.channel_id].features[character_id] = self.default_features
        self.bands[action.channel_id].states[character_id] = self.default_status

        real_characters = self.bands[channel_id].real_characters

        character_ids = list(real_characters.keys())
        personal_assistant = self.bands[channel_id].virtual_characters[self.PERSONAL_ASSISTANT]
        group_assistant = self.bands[channel_id].virtual_characters[self.GROUP_ASSISTANT]
        simple_answer = self.bands[channel_id].virtual_characters[self.SIMPLE_ANSWER]
        user_list = [personal_assistant, group_assistant, simple_answer]
        user_list.extend(list(real_characters.values()))
        
        await self.send_update_user_list(channel_id, user_list, character_ids)
        await self.send_features_from_database(channel_id, character_id)

    # DEMO: a typical leave channel handler
    async def on_leave_channel(self, action):
        character_id = action.sender
        channel_id = action.channel_id
        character = self.bands[action.channel_id].real_characters.pop(character_id, None)
        self.bands[action.channel_id].states.pop(character_id, None)
        self.bands[action.channel_id].features.pop(character_id, None)
        nickname = character.user_context.nickname

        real_characters = self.bands[channel_id].real_characters
        personal_assistant = self.bands[channel_id].virtual_characters[self.PERSONAL_ASSISTANT]
        group_assistant = self.bands[channel_id].virtual_characters[self.GROUP_ASSISTANT]
        simple_answer = self.bands[channel_id].virtual_characters[self.SIMPLE_ANSWER]
        user_list = [personal_assistant, group_assistant, simple_answer]
        user_list.extend(list(real_characters.values()))
        character_ids = list(real_characters.keys())

        await self.send_update_user_list(channel_id, user_list, character_ids)

    async def on_feature_call(self, feature_call):
        channel_id = feature_call.channel_id
        feature_id = feature_call.feature_id
        sender = feature_call.sender

        character = self.bands[channel_id].real_characters[sender]
        nickname = character.user_context.nickname
        recipients = list(self.bands[channel_id].real_characters.keys())
        
        if feature_id == "restart":
            # Clear user dialogue
            if sender in self.personal_dialogues:
                self.personal_dialogues[sender] = []
            await self.create_message(channel_id, "Your chat has been cleared.", [sender], sender=self.bands[channel_id].virtual_characters[self.PERSONAL_ASSISTANT].user_id)
        else:
            logger.warning(f"Unknown feature_id: {feature_id}")

    async def on_unknown_message(self, message_data):
        logger.warning(f"Received unknown message: {message_data}")
    
    # ==================== DEMO: Wand Event Listener ====================
    async def on_spell(self, spell):
        pass

    # ==================== helper functions ====================
    async def create_message(self, channel_id, content, recipients, subtype='text', sender=None):
        await self.send_msg_down(
            channel_id=channel_id,
            recipients=recipients,
            subtype=subtype,
            message_content=content,
            sender=sender or 'no_sender'
        )

    async def send_features_from_database(self, channel_id, character_id):
        feature_data_list = self.bands[channel_id].features.get(character_id, [])
        await self.send_update_features(channel_id, feature_data_list, [character_id])

    async def calculate_and_update_user_list_from_database(self, channel_id, character_id):
        real_characters = self.bands[channel_id].real_characters
        personal_assistant = self.bands[channel_id].virtual_characters[self.PERSONAL_ASSISTANT]
        group_assistant = self.bands[channel_id].virtual_characters[self.GROUP_ASSISTANT]
        simple_answer = self.bands[channel_id].virtual_characters[self.SIMPLE_ANSWER]
        user_list = [personal_assistant, group_assistant, simple_answer]
        user_list.extend(list(real_characters.values()))
        
        await self.send_update_user_list(channel_id, user_list, [character_id])
