# service.py
import json
from loguru import logger
from openai import AsyncOpenAI
from moobius import MoobiusService, MoobiusStorage
from notion_client import AsyncClient
from pprint import pprint
import deepl
import os


class DemoService(MoobiusService):
    def __init__(self, log_file="logs/service.log", error_log_file="logs/error.log", **kwargs):
        super().__init__(**kwargs)
        logger.add(log_file, rotation="1 day", retention="7 days", level="DEBUG")
        logger.add(error_log_file, rotation="1 day", retention="7 days", level="ERROR")
        self.default_features = {}
        self.default_status = {}
        self.bands = {}
        self.ALICE = "Alice"
        self.model = 'gpt-4-1106-preview'
        self.temperature = 0.5
        self.openai_client = None
        self.notion = None
        self.translator = None

        self.images = {
            self.ALICE: "resources/light.png"
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

    @logger.catch
    def get_notion_id(self, channel_id):
        try:
            with open('resources/notion.json', 'r') as f:
                notion_data = json.load(f)
                for item in notion_data:
                    if item["channel_id"] == channel_id:
                        return item["notion_id"]
                return None
        except Exception as e:
            print(e)
            return None    

    async def on_start(self):
        """
        Called after successful connection to websocket server and service login success.
        """
        # ==================== load features ====================
        self.openai_client = AsyncOpenAI()
        self.notion = AsyncClient(auth = os.environ["NOTION_TOKEN"])
        self.translator = deepl.Translator(os.environ["DEEPL_TOKEN"])
        
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

            if self.ALICE not in self.bands[channel_id].virtual_characters:
                self.bands[channel_id].virtual_characters[self.ALICE] = self.http_api.create_service_user(
                    self.service_id, self.ALICE, self.ALICE, self.bands[channel_id].image_paths[self.ALICE], f'I am {self.ALICE}!'
                )
            else:
                pass
                    

    async def on_msg_up(self, msg_up):
        if msg_up.subtype == "text":
            txt = msg_up.content['text']
            channel_id = msg_up.channel_id
            sender = msg_up.sender
            recipients = msg_up.recipients
            all_recipients = list(self.bands[channel_id].real_characters.keys())
            all_recipients_other_than_sender = all_recipients.copy()
            all_recipients_other_than_sender.remove(sender)
            #try:
            #    txt = self.translator.translate_text(text = txt, target_lang = "EN-US").text            
            #except Exception as e:
            #    print(e)
            if ("?" in txt or "？" in txt) and set(all_recipients).issubset(set(recipients)) or set(all_recipients).isdisjoint(set(recipients)):
                txt_en = self.translator.translate_text(text = txt, target_lang = "EN-US").text
                await self.create_message(channel_id, txt, [sender], sender = sender)
                await self.create_message(channel_id, txt_en, all_recipients_other_than_sender, sender = sender)
                answer = await self.get_answer_simple(txt)
                answer_en = self.translator.translate_text(text = answer, target_lang = "EN-US").text
                await self.create_message(channel_id, answer, [sender], sender = self.bands[channel_id].virtual_characters[self.ALICE].user_id)
                await self.create_message(channel_id, answer_en, all_recipients_other_than_sender, sender = self.bands[channel_id].virtual_characters[self.ALICE].user_id)
                #await self.create_message(channel_id, answer, all_recipients, sender = self.bands[channel_id].virtual_characters[self.ALICE].user_id)
                try:
                    #consider the 2000 length limit
                    txt_en_list = [txt_en[i:i+2000] for i in range(0, len(txt_en), 2000)]
                    answer_en_list = [answer_en[i:i+2000] for i in range(0, len(answer_en), 2000)]
                    txt_en_payload = []
                    answer_en_payload = []
                    for txt_en in txt_en_list:
                        txt_en_payload.append(
                            {
                                "text": {
                                    "content": txt_en
                                }
                            }
                        )
                    for answer_en in answer_en_list:
                        answer_en_payload.append(
                            {
                                "text": {
                                    "content": answer_en
                                }
                            }
                        )
                    notion_response = await self.notion.pages.create(
                        parent = {
                            "database_id": self.get_notion_id(channel_id)
                        },
                        properties = {
                            "Answer": {
                                "rich_text": answer_en_payload
                            },
                            "Question/Issue": {
                                "title": txt_en_payload
                            }
                        }
                    )
                    pprint(notion_response)
                except Exception as e:
                    print(e)                  
            else:
                # DEMO: send message to other recipients
                txt_en = self.translator.translate_text(text = txt, target_lang = "EN-US").text
                if sender in recipients:
                    recipients_other_than_sender = recipients.copy()
                    recipients_other_than_sender.remove(sender)
                    await self.create_message(channel_id, txt, [sender], sender=sender)
                    await self.create_message(channel_id, txt_en, recipients_other_than_sender, sender=sender)
                else:
                    await self.create_message(channel_id, txt_en, recipients, sender=sender)
                #await self.send(payload_type='msg_down', payload_body=msg_up)
        
        # DEMO: other message types. TODO: save to your disk
        else:   
            await self.send(payload_type='msg_down', payload_body=msg_up)

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
        real_characters = self.bands[action.channel_id].real_characters

        user_list = list(real_characters.values())
        character_ids = list(real_characters.keys())
        
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

        real_characters = self.bands[action.channel_id].real_characters
        user_list = list(real_characters.values())
        character_ids = list(real_characters.keys())

        await self.send_update_user_list(channel_id, user_list, character_ids)

    async def on_feature_call(self, feature_call):
        channel_id = feature_call.channel_id
        feature_id = feature_call.feature_id
        sender = feature_call.sender

        character = self.bands[channel_id].real_characters[sender]
        nickname = character.user_context.nickname
        recipients = list(self.bands[channel_id].real_characters.keys())
        
        if feature_id == "notion":
            notion_id = self.get_notion_id(channel_id)
            notion_link = f"https://moobius.notion.site/moobius/{notion_id}"
            await self.create_message(channel_id, notion_link, [sender], sender=self.bands[channel_id].virtual_characters[self.ALICE].user_id)
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
        alice = self.bands[channel_id].virtual_characters[self.ALICE]
        user_list = [alice]
        user_list.extend(list(real_characters.values()))
        
        await self.send_update_user_list(channel_id, user_list, [character_id])