import os
import json
import discord
import asyncio
from typing import Union

from src import responses
from src.log import logger
from utils.message_utils import send_split_message

from dotenv import load_dotenv
from discord import app_commands

import openai

load_dotenv()

class aclient(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.current_channel = None
        self.activity = discord.Activity(type=discord.ActivityType.listening, name="/chat | /help")
        self.isPrivate = False
        self.is_replying_all = os.getenv("REPLYING_ALL")
        self.replying_all_discord_channel_id = os.getenv("REPLYING_ALL_DISCORD_CHANNEL_ID")
        self.openAI_API_key = os.getenv("OPENAI_API_KEY")
        self.openAI_gpt_engine = os.getenv("GPT_ENGINE")

        config_dir = os.path.abspath(f"{__file__}/../../")
        prompt_name = 'system_prompt.txt'
        prompt_path = os.path.join(config_dir, prompt_name)
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.starting_prompt = f.read()

        self.chat_model = os.getenv("CHAT_MODEL")
        if self.chat_model == "OFFICIAL":
            openai.api_key = self.openAI_API_key
        elif self.chat_model == "LOCAL":
            openai.api_base = "http://localhost:8000/v1"
            openai.api_key = "empty"
        self.message_queue = asyncio.Queue()

        self.chat_history = {}
        self.chat_history_lock = asyncio.Lock()



    async def process_messages(self):
        while True:
            if self.current_channel is not None:
                while not self.message_queue.empty():
                    async with self.current_channel.typing():
                        message, user_message = await self.message_queue.get()
                        try:
                            await self.send_message(message, user_message)
                        except Exception as e:
                            logger.exception(f"Error while processing message: {e}")
                        finally:
                            self.message_queue.task_done()
            await asyncio.sleep(1)

    async def set_chat_history(self, user: str, message: list):
        # print('set chat history')
        async with self.chat_history_lock:
            self.chat_history[user]=message

    async def get_chat_history(self, user: str):
        # print('get chat history')
        async with self.chat_history_lock:
            if user in self.chat_history:
                # print('user',user)
                return self.chat_history[user]
            else:
                # print('no user',user)
                return None

    async def clear_chat_history(self, user: str):
        # print('clear chat history')
        async with self.chat_history_lock:
            if user in self.chat_history:
                del self.chat_history[user]
                return True
            else:
                return False

    async def enqueue_message(self, message, user_message):
        await message.response.defer(ephemeral=self.isPrivate) if self.is_replying_all == "False" else None
        await self.message_queue.put((message, user_message))

    async def send_message(self, message, user_message):
        if self.is_replying_all == "False":
            user = message.user.id
        else:
            user = message.author.id
        user = str(user)
        try:

            if self.chat_model == "OFFICIAL":
                # print('official')
                response, history = await responses.official_handle_response(user_message, self, user, stream=True)
                end = ''
            elif self.chat_model == "LOCAL":
                response, history = await responses.local_handle_response(user_message, self, user, stream=True, rag=True)
                end = f"\n If you want me to coninue, use /chat continue.\n To help us improve, please rate this response using the reactions below(👍or👎)."
            # msg=await send_split_message(self, response, message)

            collected_messages = []
            buffer = f'> **{user_message}** - <@{str(user)}> \n\n'
            sent = await message.followup.send(buffer)
            sent = await message.followup.send('generating response...')
            msg = ''
            current_index = 1
            send_allowed = asyncio.Event()
            send_allowed.set()

            def on_send_done(task):
                nonlocal sent
                sent = task.result()  # get the result of the send task
                send_allowed.set()  # set send_allowed

            async for chunk in response:
                if self.chat_model == "OFFICIAL":
                    collected_messages.append(chunk['choices'][0]['delta'])
                    msg = ''.join([m.get('content', '') for m in collected_messages])
                elif self.chat_model == "LOCAL":
                    collected_messages.append(chunk['choices'][0]['delta'])
                    msg = ''.join([m.get('content', '') for m in collected_messages])
                if not send_allowed.is_set():
                    continue
                if not msg:
                    continue
                msg_split = await send_split_message(client, msg, message, send=False)
                index = len(msg_split)

                if index == 0:
                    continue
                send_allowed.clear()
                if index == current_index:
                    send_task = asyncio.create_task(sent.edit(content=msg_split[current_index - 1]))
                else:
                    while current_index < index:
                        send_task = asyncio.create_task(sent.edit(content=msg_split[current_index - 1]))
                        send_task = asyncio.create_task(message.followup.send(msg_split[current_index]))
                        current_index += 1
                send_task.add_done_callback(on_send_done)  # add callback
            await send_allowed.wait()  # wait for the last send to complete
            msg_split = await send_split_message(client, msg, message, send=False)
            index = len(msg_split)
            if index == current_index:
                sent = await sent.edit(content=msg_split[-1]+ end)
            else:
                sent = await message.followup.send(msg_split[-1]+ end)
            if self.chat_model == "OFFICIAL":
                role = ''.join([m.get('role', '') for m in collected_messages])
            elif self.chat_model == "LOCAL":
                role = ''.join([m.get('role', '') for m in collected_messages])
            assistance_message = {"role": role, "content": msg}
            history.append(assistance_message)
            await client.set_chat_history(user, history)

            await sent.add_reaction("👍")
            await sent.add_reaction("👎")
            if not os.path.exists("./chatlog.json"):
                with open("./chatlog.json", "w", encoding="utf-8") as f:
                    messages = {}
                    messages[sent.id] = {"message": user_message, "user": message.user.name, "response": msg, "reactions": {}}
                    json.dump(messages, f,indent=4,ensure_ascii=False)
            else:
                with open("./chatlog.json", "r+", encoding="utf-8") as f:
                    messages= json.load(f)
                    messages[sent.id] = {"message": user_message, "user": message.user.name, "response": msg, "reactions": {}}
                    f.seek(0)
                    json.dump(messages, f, indent=4,ensure_ascii=False)
                    f.truncate()
        except Exception as e:
            logger.exception(f"Error while sending : {e}")
            if self.is_replying_all == "True":
                await message.channel.send(
                    f"> **ERROR: Something went wrong, please try again later!** \n ```ERROR MESSAGE: {e}```")
            else:
                await message.followup.send(
                    f"> **ERROR: Something went wrong, please try again later!** \n ```ERROR MESSAGE: {e}```")

    # async def send_start_prompt(self):
    #     discord_channel_id = os.getenv("DISCORD_CHANNEL_ID")
    #     try:
    #         if self.starting_prompt:
    #             if (discord_channel_id):
    #                 logger.info(f"Send system prompt with size {len(self.starting_prompt)}")
    #                 response = ""
    #                 if self.chat_model == "OFFICIAL":
    #                     response = f"{response}{await responses.official_handle_response(self.starting_prompt, self)}"
    #                 elif self.chat_model == "LOCAL":
    #                     response = f"{response}{await responses.local_handle_response(self.starting_prompt, self)}"
    #                 channel = self.get_channel(int(discord_channel_id))
    #                 await channel.send(response)
    #                 logger.info(f"System prompt response:{response}")
    #             else:
    #                 logger.info("No Channel selected. Skip sending system prompt.")
    #         else:
    #             logger.info(f"Not given starting prompt. Skiping...")
    #     except Exception as e:
    #         logger.exception(f"Error while sending system prompt: {e}")


client = aclient()
