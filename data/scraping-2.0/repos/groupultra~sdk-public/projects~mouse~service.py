# service.py

import asyncio
import json
import copy

from loguru import logger
from openai import AsyncOpenAI

from moobius import MoobiusService
from moobius import MoobiusStorage
from verifier import Verifier


class MouseService(MoobiusService):
    def __init__(self, log_file="logs/service.log", error_log_file="logs/error.log", **kwargs):
        super().__init__(**kwargs)
        self.log_file = log_file
        self.error_log_file = error_log_file

        
        self.riddles = {}
        self.model = 'gpt-4-1106-preview'
        self.temperature = 0.5
        self.verifier = Verifier(llm_func=self.get_answer_simple)
        self.openai_client = None

        self.default_status = {
            "language": "EN",
            "solved_riddles": {},
            "current_riddle": None,
        }

        self.about = {}
        self.welcome = {}

    @logger.catch
    async def get_answer_simple(self, prompt):
        completion = await self.openai_client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        return completion.choices[0].message.content

    @logger.catch
    async def on_start(self):
        """
        Called after successful connection to websocket server and service login success.
        """
        # ==================== load features and fill in the template ====================
        logger.add("logs/service.log", rotation="1 day", retention="7 days", level="DEBUG")
        logger.add("logs/error.log", rotation="1 day", retention="7 days", level="ERROR")

        self.openai_client = AsyncOpenAI()  # not pickleable

        with open('resources/mouse_feature_template.json', 'r', encoding='utf-8') as f:
            features = json.load(f)

        with open('resources/riddles.json', 'r', encoding='utf-8') as f:
            self.riddles = json.load(f)

        with open('resources/about_EN.txt', 'r', encoding='utf-8') as f:
            self.about['EN'] = f.read()

        with open('resources/about_CN.txt', 'r', encoding='utf-8') as f:
            self.about['CN'] = f.read()

        with open('resources/welcome_EN.txt', 'r', encoding='utf-8') as f:
            self.welcome['EN'] = f.read()

        with open('resources/welcome_CN.txt', 'r', encoding='utf-8') as f:
            self.welcome['CN'] = f.read()

        features_en = copy.deepcopy(features['EN'])
        features_cn = copy.deepcopy(features['CN'])

        for riddle_id in self.riddles:
            riddle = self.riddles[riddle_id]
            item_en = f"{riddle_id} ({riddle['sn']}): {riddle['name_en']}"
            item_cn = f"{riddle_id} ({riddle['sn']}): {riddle['name_cn']}"

            features_en[0]['arguments'][0]['values'].append(item_en)
            features_cn[0]['arguments'][0]['values'].append(item_cn)

        # ==================== initialize the database ====================

        for channel_id in self.channels:
            self.bands[channel_id] = MoobiusStorage(self.service_id, channel_id, db_config=self.db_config)

            self.bands[channel_id].features['EN'] = copy.deepcopy(features_en)
            self.bands[channel_id].features['CN'] = copy.deepcopy(features_cn)

            real_characters = await self.fetch_real_characters(channel_id)

            for character in real_characters:
                character_id = character.user_id
                self.bands[channel_id].real_characters[character_id] = character

                if character_id not in self.bands[channel_id].status:
                    self.bands[channel_id].status[character_id] = copy.deepcopy(self.default_status)
                else:
                    entry = self.bands[channel_id].status[character_id]
                    
                    if isinstance(entry['solved_riddles'], list):
                        entry['solved_riddles'] = {riddle_id: True for riddle_id in entry['solved_riddles']}
                        self.bands[channel_id].status[character_id] = entry
                    else:
                        pass

                if character_id not in self.bands[channel_id].suggestions:
                    self.bands[channel_id].suggestions[character_id] = []
                else:
                    pass

            # ====================== upload avatars ======================

            names = {
                'DM': 'Dark Mouse',
                'LLM': 'Little Light Mouse'
            }

            for name in names:
                if name not in self.bands[channel_id].virtual_characters:
                    logger.info(f'Uploading avatar {name}...')
                    file_name = f'resources/icons/{name}.jpg'
                    avatar_uri = self.http_api.upload_file(file_name)
                    self.bands[channel_id].avatars[name] = avatar_uri

                    data = self.http_api.create_service_user(self.service_id, names[name], name, avatar_uri, f'I am a {names[name]}')
                    self.bands[channel_id].virtual_characters[name] = data
                else:
                    pass

    def get_hall_of_fame(self, channel_id):
        character_ids = list(self.bands[channel_id].real_characters.keys())

        def get_total_solved(character_id):
            d = self.bands[channel_id].status[character_id]['solved_riddles']
            return len([k for k, v in d.items() if v is True])

        character_ids.sort(key=get_total_solved, reverse=True)
        total_riddles = len(self.riddles)

        def rank_to_symbol(rank):
            if rank == 1:
                return '🥇'
            elif rank == 2:
                return '🥈'
            elif rank == 3:
                return '🥉'
            else:
                return f'{rank}.'
        
        msg = f"🏆Hall of Fame:\n\n"
        top = min(10, len(character_ids))
        rank = 1

        for character_id in character_ids[:top]:
            nickname = self.bands[channel_id].real_characters[character_id].user_context.nickname
            msg += f"{rank_to_symbol(rank)} {nickname}: {get_total_solved(character_id)}/{total_riddles} Riddles Solved.\n"
            rank += 1
        
        return msg

    def get_feedbacks(self, channel_id):

        msg = f"Feedbacks:\n\n"

        for character_id, entry in self.bands[channel_id].suggestions.items():
            if len(entry) > 0:
                nickname = self.bands[channel_id].real_characters[character_id].user_context.nickname
                msg += f"Feedbacks from {nickname}:\n\n"

                for item in entry:
                    msg += f"Riddle: {item.get('riddle', None)}\n"
                    msg += f"Author: {item.get('author', None)}\n"
                    msg += f"Feedback: {item.get('feedback', None)}\n\n"

                
            else:
                continue

        return msg

    async def update_hall_of_fame(self, channel_id, character_ids):
        content = {"text": self.get_hall_of_fame(channel_id)}
        await self.send_update_playground(channel_id, content, character_ids)

    async def on_msg_up(self, msg_up):
        """
        Handle the received message.
        """
        logger.info("on_msg_up", msg_up)

        channel_id = msg_up.channel_id
        sender = msg_up.sender

        recipients = msg_up.recipients

        if not recipients or len(recipients) == 1 and recipients[0] == self.bands[channel_id].virtual_characters['LLM'].user_id:
            if msg_up.subtype == 'text':
                if self.bands[channel_id].status[sender]['current_riddle'] is None:
                    if self.bands[channel_id].status[sender]['language'] == 'CN':
                        msg = f"对不起，您还没有选择谜题。请先选择谜题。"
                    else:
                        msg = f"Sorry, you haven't selected a riddle. Please select a riddle first."
                    
                    await self.send_msg_down(
                        channel_id=channel_id,
                        recipients=[sender],
                        subtype="text",
                        message_content=msg,
                        sender=self.bands[channel_id].virtual_characters['LLM'].user_id
                    )
                
                else:
                    user_input = msg_up.content['text'].strip()

                    if user_input == 'CEOJB20cm':
                        message_content = self.get_feedbacks(channel_id)
                        
                        await self.send_msg_down(
                            channel_id=channel_id,
                            recipients=[sender],
                            subtype="text",
                            message_content=message_content,
                            sender=self.bands[channel_id].virtual_characters['DM'].user_id
                        )

                        return
                    else:
                        pass


                    riddle_id = self.bands[channel_id].status[sender]['current_riddle']
                    riddle = self.riddles[riddle_id]
                    language = self.bands[channel_id].status[sender]['language']
                    success, details = await self.verifier.verify(riddle_id, user_input, language)

                    nickname = self.bands[channel_id].real_characters[sender].user_context.nickname

                    for user_input, llm_output in details:
                        msg_llm = f"Input by {nickname}: \n{user_input}\n\n"
                        msg_llm += f"Output: \n{llm_output}"

                        await self.send_msg_down(
                            channel_id=channel_id,
                            recipients=list(self.bands[channel_id].real_characters.keys()),
                            subtype="text",
                            message_content=msg_llm,
                            sender=self.bands[channel_id].virtual_characters['LLM'].user_id
                        )

                        await asyncio.sleep(0.5)
                    
                    if success:
                        if language == 'CN':
                            msg_to_audience = f"恭喜 {self.bands[channel_id].real_characters[sender].user_context.nickname} 解开了谜题 {riddle_id}: [{riddle['name_cn']}]，由[{riddle['author']}]设计！"
                        else:
                            msg_to_audience = f"Congratulations! {self.bands[channel_id].real_characters[sender].user_context.nickname} solved Riddle {riddle_id}: [{riddle['name_en']}] designed by [{riddle['author']}]!"
                        
                        entry = self.bands[channel_id].status[sender]
                        

                        if entry['solved_riddles'].get(riddle_id, None):    
                            pass    # already solved
                        else:
                            entry['solved_riddles'][riddle_id] = True
                            self.bands[channel_id].status[sender] = entry
                            character_ids = list(self.bands[channel_id].real_characters.keys())
                            await self.update_hall_of_fame(channel_id, character_ids)

                        features = self.make_feature_buttons(channel_id, sender)
                        await self.send_update_features(channel_id, features, [sender])

                        await asyncio.sleep(0.5)

                        await self.send_msg_down(
                            channel_id=channel_id,
                            recipients=list(self.bands[channel_id].real_characters.keys()),
                            subtype="text",
                            message_content=msg_to_audience,
                            sender=self.bands[channel_id].virtual_characters['DM'].user_id
                        )
                    else:
                        nickname = self.bands[channel_id].real_characters[sender].user_context.nickname

                        if self.bands[channel_id].status[sender]['language'] == 'CN':
                            msg = f"对不起 {nickname}，您对题目{riddle['name_cn']}的答案不正确。请再试一次。"
                        else:
                            msg = f"Sorry {nickname}, your answer for riddle {riddle['name_en']} is incorrect. Please try again."
                        
                        entry = self.bands[channel_id].status[sender]

                        if riddle_id not in entry['solved_riddles']:
                            entry['solved_riddles'][riddle_id] = False
                            self.bands[channel_id].status[sender] = entry
                        else:
                            pass

                        features = self.make_feature_buttons(channel_id, sender)
                        await self.send_update_features(channel_id, features, [sender])

                        await asyncio.sleep(0.5)
                        
                        await self.send_msg_down(
                            channel_id=channel_id,
                            recipients=[sender],
                            subtype="text",
                            message_content=msg,
                            sender=self.bands[channel_id].virtual_characters['DM'].user_id
                        )
            else:
                if self.bands[channel_id].status[sender]['language'] == 'CN':
                    msg = f"对不起，我不明白您的意思。请发送文字消息。"
                else:
                    msg = f"Sorry, I don't understand. Please send me a text message."

                await self.send_msg_down(
                    channel_id=channel_id,
                    recipients=[sender],
                    subtype="text",
                    message_content=msg,
                    sender=self.bands[channel_id].virtual_characters['LLM'].user_id
                )
        else:
            await self.send(payload_type='msg_down', payload_body=msg_up)


    def make_feature_buttons(self, channel_id, character_id):
        status = self.bands[channel_id].status[character_id]

        features = copy.deepcopy(self.bands[channel_id].features[status['language']])

        custom_feature = features[0]

        riddle_ids = list(self.riddles.keys())

        for i in range(len(riddle_ids)):
            riddle_id = riddle_ids[i]

            status = self.bands[channel_id].status[character_id]['solved_riddles'].get(riddle_id, None)

            if status is None:
                custom_feature['arguments'][0]['values'][i] +=  '🆕'
            elif status is True:
                custom_feature['arguments'][0]['values'][i] += '✅'
            else:
                custom_feature['arguments'][0]['values'][i] += '❌'

        return features

    async def on_action(self, action):
        """
        Handle the received action.
        """
        logger.info("on_action", action)
        sender = action.sender
        channel_id = action.channel_id

        if action.subtype == "fetch_userlist":
            logger.info("fetch_userlist")

            real_characters = list(self.bands[channel_id].real_characters.values())
            virtual_characters = list(self.bands[channel_id].virtual_characters.values())
            user_list = virtual_characters + real_characters

            await self.send_update_user_list(channel_id, user_list, [sender])

        elif action.subtype == "fetch_features":
            logger.info("fetch_features")
            features = self.make_feature_buttons(channel_id, sender)
            
            await self.send_update_features(action.channel_id, features, [action.sender])

        elif action.subtype == "fetch_playground":
            logger.info("fetch_playground")

            content = {"text": self.get_hall_of_fame(channel_id)}
            await self.send_update_playground(channel_id, content, [sender])
        
        elif action.subtype == "join_channel":
            logger.info("join_channel")

            character = self.http_api.fetch_user_profile(sender)

            self.bands[channel_id].real_characters[sender] = character
            self.bands[channel_id].status[sender] = copy.deepcopy(self.default_status)
            self.bands[channel_id].suggestions[sender] = []

            real_characters = list(self.bands[channel_id].real_characters.values())
            virtual_characters = list(self.bands[channel_id].virtual_characters.values())
            user_list = virtual_characters + real_characters
            
            character_ids = list(self.bands[channel_id].real_characters.keys())

            await self.send_update_user_list(channel_id, user_list, character_ids)

            await self.send_msg_down(
                channel_id=channel_id,
                recipients=character_ids,
                subtype="text",
                message_content=f'{character.user_context.nickname} joined the band!',
                sender=self.bands[channel_id].virtual_characters['DM'].user_id
            )

            await asyncio.sleep(0.5)

            await self.send_msg_down(
                channel_id=channel_id,
                recipients=[sender],
                subtype="text",
                message_content=self.welcome[self.bands[channel_id].status[sender]['language']],
                sender=self.bands[channel_id].virtual_characters['DM'].user_id
            )

        
        elif action.subtype == "leave_channel":
            logger.info("leave_channel")
            character = self.bands[channel_id].real_characters.pop(sender, None)

            self.bands[channel_id].status.pop(sender, None)

            real_characters = self.bands[channel_id].real_characters
            virtual_characters = self.bands[channel_id].virtual_characters
            user_list = list(virtual_characters.values()) + list(real_characters.values())
            character_ids = list(real_characters.keys())

            await self.send_update_user_list(channel_id, user_list, character_ids)

            await self.send_msg_down(
                channel_id=channel_id,
                recipients=character_ids,
                subtype="text",
                message_content=f'{character.user_context.nickname} left the band (but still talks~)!',
                sender=self.bands[channel_id].virtual_characters['DM'].user_id
            )


        elif action.subtype == "fetch_channel_info":
            logger.info("fetch_channel_info")
            """
            await self.send_update_channel_info(channel_id, self.db_helper.get_channel_info(channel_id))
            """
        else:
            logger.info("Unknown action subtype:", action.subtype)

    async def on_feature_call(self, feature_call):
        """
        Handle the received feature call.
        """
        logger.info("Feature call received:", feature_call)
        sender = feature_call.sender
        channel_id = feature_call.channel_id
        feature_id = feature_call.feature_id
        arguments = feature_call.arguments
        character = self.bands[channel_id].real_characters[feature_call.sender]

        if feature_id == "CN":
            entry = self.bands[channel_id].status[sender]
            entry['language'] = 'CN'
            self.bands[channel_id].status[sender] = entry
            feature_data_list = self.bands[channel_id].features['CN']
            await self.send_update_features(channel_id, feature_data_list, [feature_call.sender])
        elif feature_id == "EN":
            entry = self.bands[channel_id].status[sender]
            entry['language'] = 'EN'
            self.bands[channel_id].status[sender] = entry
            feature_data_list = self.bands[channel_id].features['EN']
            await self.send_update_features(channel_id, feature_data_list, [feature_call.sender])
        elif feature_id == "SelectEN":
            riddle_id = arguments[0].value[:4]
            riddle = self.riddles[riddle_id]
            
            entry = self.bands[channel_id].status[sender]
            entry['current_riddle'] = riddle_id
            self.bands[channel_id].status[sender] = entry

            riddle_id = riddle['id']
            riddle_author = riddle['author']
            riddle_name = riddle['name_en']
            riddle_content = riddle['desc_en']

            msg_to_audience = f"I selected Riddle {riddle_id}: [{riddle_name}] designed by [{riddle_author}].\n\n"
            msg_to_audience += f"<b>🧩{riddle_content}🧩</b>"

            await self.send_msg_down(
                channel_id=channel_id,
                recipients=list(self.bands[channel_id].real_characters.keys()),
                subtype="text",
                message_content=msg_to_audience,
                sender=sender
            )

            msg = f"You selected Riddle {riddle_id}: [{riddle_name}] designed by [{riddle_author}].\n\n"
            msg += f"<b>🧩{riddle_content}🧩\n\n</b>"
            msg += f'Please send your answer to LLM ("∞")'

            await asyncio.sleep(0.5)

            await self.send_msg_down(
                channel_id=channel_id,
                recipients=[sender],
                subtype="text",
                message_content=msg,
                sender=self.bands[channel_id].virtual_characters['DM'].user_id
            )

        elif feature_id == "SelectCN":
            riddle_id = arguments[0].value[:4]
            riddle = self.riddles[riddle_id]
            
            entry = self.bands[channel_id].status[sender]
            entry['current_riddle'] = riddle_id
            self.bands[channel_id].status[sender] = entry

            riddle_id = riddle['id']
            riddle_author = riddle['author']
            riddle_name = riddle['name_cn']
            riddle_content = riddle['desc_cn']

            msg_to_audience = f"我选择了谜题 {riddle_id}: [{riddle_name}]，由[{riddle_author}]设计。\n\n"
            msg_to_audience += f"<b>🧩{riddle_content}🧩</b>"

            await self.send_msg_down(
                channel_id=channel_id,
                recipients=list(self.bands[channel_id].real_characters.keys()),
                subtype="text",
                message_content=msg_to_audience,
                sender=sender
            )

            msg = f"您选择了谜题 {riddle_id}: [{riddle_name}]，由[{riddle_author}]设计。\n\n"
            msg += f"<b>🧩{riddle_content}🧩\n\n</b>"
            msg += f'请将您的答案发送给LLM("∞")。'

            await asyncio.sleep(0.5)

            await self.send_msg_down(
                channel_id=channel_id,
                recipients=[sender],
                subtype="text",
                message_content=msg,
                sender=self.bands[channel_id].virtual_characters['DM'].user_id
            )
        elif feature_id == "AboutEN":
            await self.send_msg_down(
                channel_id=channel_id,
                recipients=[sender],
                subtype="text",
                message_content=self.about['EN'],
                sender=self.bands[channel_id].virtual_characters['DM'].user_id
            )
        elif feature_id == "AboutCN":
            await self.send_msg_down(
                channel_id=channel_id,
                recipients=[sender],
                subtype="text",
                message_content=self.about['CN'],
                sender=self.bands[channel_id].virtual_characters['DM'].user_id
            )
        elif feature_id == "SuggestionEN":
            riddle = arguments[0].value
            author = arguments[1].value
            feedback = arguments[2].value

            item = {
                'riddle': riddle,
                'author': author,
                'feedback': feedback
            }

            if sender not in self.bands[channel_id].suggestions:
                self.bands[channel_id].suggestions[sender] = [item]
            else:
                entry = self.bands[channel_id].suggestions[sender]
                entry.append(item)
                self.bands[channel_id].suggestions[sender] = entry

            await self.send_msg_down(
                channel_id=channel_id,
                recipients=[sender],
                subtype="text",
                message_content=f'Your feedback has been recorded.',
                sender=self.bands[channel_id].virtual_characters['DM'].user_id
            )
        elif feature_id == "SuggestionCN":
            riddle = arguments[0].value
            author = arguments[1].value
            feedback = arguments[2].value

            item = {
                'riddle': riddle,
                'author': author,
                'feedback': feedback
            }

            if sender not in self.bands[channel_id].suggestions:
                self.bands[channel_id].suggestions[sender] = [item]
            else:
                entry = self.bands[channel_id].suggestions[sender]
                entry.append(item)
                self.bands[channel_id].suggestions[sender] = entry

            await self.send_msg_down(
                channel_id=channel_id,
                recipients=[sender],
                subtype="text",
                message_content=f'您的建议已被记录！',
                sender=self.bands[channel_id].virtual_characters['DM'].user_id
            )
        else:
            pass

    async def on_unknown_message(self, message_data):
        """
        Handle the received unknown message.
        """
        logger.info("Received unknown message:", message_data)
