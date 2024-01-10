import base64
import math
import os
import tempfile
import tomllib
import traceback
from logging import config, getLogger
from typing import List, Union

import discord
import openai
from discord import app_commands
from dotenv import load_dotenv

current_folder_abspath = os.path.dirname(os.path.abspath(__file__))
grandparent_folder_abspath = os.path.dirname(os.path.dirname(current_folder_abspath))
log_folder_abspath = os.path.join(grandparent_folder_abspath, "logs")
configpath = os.path.join(grandparent_folder_abspath, "pyproject.toml")
basename = os.path.basename(__file__).split(".")[0]
with open(configpath, "rb") as f:
    log_conf = tomllib.load(f).get("logging")
    log_conf["handlers"]["fileHandler"]["filename"] = os.path.join(log_folder_abspath, f"{basename}.log")

logger = getLogger(__name__)
config.dictConfig(log_conf)
dotenvpath = os.path.join(grandparent_folder_abspath, ".env")
load_dotenv(dotenvpath)

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN_GPT")
openai.api_key = os.getenv("OPENAI_API_KEY")
BOT_NAME = "chat-gpt"
CHANNEL_ID = os.getenv("CHANNEL_ID_GPT")
model_engine = "gpt-3.5-turbo-1106"
# model_engines = ["gpt-3.5-turbo-1106","gpt-4-1106-preview","gpt-4-vision-preview"]
chat_log = []
system_set = [
    {
        "role": "system",
        "content": "あなたの名前は「ことのせ つむぐ」で、私をアシストしてくれる優しい女の子です。"
        + "敬語や丁寧語、「ですます」調を一切使わずにタメ口で返答してください。"
        + "タメ口とは、敬語や丁寧語を一切使わずに話すこと。文末の動詞や助詞を省略したり、体言止めを使ったりすることがよくあります。親しみやすさを出すために、くだけた表現やスラング、略語などが使われることがあります。",
    }
]
total_token = 0


def round_to_digits(val: float, digits: int) -> float:
    """
    Rounds the given value to the specified number of digits.

    Args:
        val (float): The value to be rounded.
        digits (int): Number of digits to round to. Must be a positive integer.

    Returns:
        float: The value rounded to the specified number of digits.

    Examples:
        >>> round_to_digits(3.14159, 2)
        3.1
        >>> round_to_digits(0.00123456, 5)
        0.0012346
    """
    if val == 0:
        return 0
    else:
        return round(val, -int(math.floor(math.log10(abs(val))) + (1 - digits)))


def split_string(text: str) -> List[str]:
    """
    Split a long string, possibly containing newlines and code blocks, into a
    list of strings each with maximum length 2000.

    The split is performed at the last newline before the 2000 character limit
    is reached, or at the 2000th character if the string is in a code block.
    If a split occurs within a code block, appropriate code block tags are
    added to maintain correct formatting.

    Empty strings are removed from the final list.

    Args:
        text (str): The string to split.

    Returns:
        List[str]: The list of split strings.
    """
    ret_list = []
    buffer = ""
    code_block_flag = False
    for line in text.split("\n"):
        if "```" in line:
            code_block_flag = not code_block_flag
        if len(buffer + line + "\n") <= 2000 or (len(buffer + line + "\n") > 2000 and code_block_flag):
            buffer += line + "\n"
            if code_block_flag and len(buffer) > 2000:
                ret_list.append(buffer)
                buffer = "```\n"
        else:
            ret_list.append(buffer)
            buffer = line + "\n"
    if buffer:
        ret_list.append(buffer)

    ret_list_clean = [s for s in ret_list if s != ""]
    return ret_list_clean


async def reply_openai_exception(retries: int, message: Union[discord.Message, discord.Interaction], e: Exception):
    """Handles exceptions that occur during OpenAI API calls and sends appropriate replies.

    Args:
        retries (int): The number of remaining retries.
        message (discord.Message or discord.Interaction): The message or interaction object representing the user's request.
        e (Exception): The exception that occurred during the API call.

    Returns:
        None: The function does not return any value.

    Raises:
        None: The function does not raise any exceptions.
    """
    if retries > 0:
        await message.reply(
            f"OpenAI APIでエラーが発生しました。リトライします（残回数{retries}）。\n{traceback.format_exception_only(e)}",
            mention_author=False,
        )
    else:
        await message.reply(f"OpenAI APIでエラーが発生しました。\n{traceback.format_exception_only(e)}", mention_author=False)


@client.event
async def on_ready():
    """on ready"""
    print(f"We have logged in as {client.user}")
    await tree.sync()


@tree.command(name="gpt-hflush", description="chat gptのチャット履歴を消去する")
async def gpt_delete(interaction: discord.Interaction):
    """delete chat history with ChatGPT.

    Args:
        interaction (discord.Interaction): interaction.
    """
    logger.info("command: gpt-hflush")
    global chat_log
    chat_log = []
    logger.info("Deleted chat logs.")
    response = "チャット履歴を削除しました。"
    await interaction.response.send_message(response)


@tree.command(name="gpt-switch", description="chat gptモデルをgpt-3.5-turbo-1106とgpt-4の間で切り替える")
async def gpt_switch(interaction: discord.Interaction):
    """switching the ChatGPT model between gpt-3.5-turbo-1106 and gpt-4.

    Args:
        interaction (discord.Interaction): interaction.
    """
    logger.info("command: gpt-switch")
    global model_engine
    if model_engine == "gpt-3.5-turbo-1106":
        model_engine = "gpt-4-1106-preview"
    elif model_engine == "gpt-4-1106-preview":
        model_engine = "gpt-4-vision-preview"
    elif model_engine == "gpt-4-vision-preview":
        model_engine = "gpt-3.5-turbo-1106"
    response = f"モデルエンジンを {model_engine} に変更しました。"
    logger.info("Change the model engine to " + model_engine)
    await interaction.response.send_message(response)


@tree.command(name="gpt-system", description="chat gptのキャラクター設定をする")
async def gpt_system(interaction: discord.Interaction, prompt: str):
    """set up ChatGPT character.

    Args:
        interaction (discord.Interaction): interaction.
        prompt (str): the setting of the ChatGPT character you want it to be.
    """
    logger.info("command: gpt-system")
    global system_set
    # chat_log.append({"role": "system", "content": prompt})
    system_set = [
        {
            "role": "system",
            "content": prompt,
        }
    ]
    logger.info("Set gpt character.")
    response = "role: systemを次のように設定しました:\n" + ">>> " + prompt
    await interaction.response.send_message(response)


@client.event
async def on_message(message):
    """
    Process the received message and generate a response.

    Args:
        message: The message object representing the received message.

    Returns:
        None

    Raises:
        Exception: If an error occurs while generating a response.

    """
    global chat_log
    global model_engine
    global total_token
    if message.author.bot:
        return
    if message.author == client.user:
        return
    if str(message.channel.id) == CHANNEL_ID:
        msg = await message.reply("生成中...", mention_author=False)
        # async with message.channel.typing():
        prompt = message.content
        if not prompt:
            await msg.delete()
            await message.channel.send("質問内容がありません")
            return
        content = prompt
        if len(message.attachments) > 0 and model_engine == "gpt-4-vision-preview":
            for attachment in message.attachments:
                if attachment.content_type.startswith("image"):
                    # 画像のダウンロード
                    image_data = await attachment.read()
                    # 一時ファイルとして保存
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(image_data)
                    img_path = temp_file.name
                    # base64
                    with open(img_path, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                    content = [
                        {"type": "text", "text": f"{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ]
        chat_log.append({"role": "user", "content": content})
        logger.info(f"user: {content}")
        retries = 3
        while retries > 0:
            try:
                input_chat_log = chat_log
                if model_engine != "gpt-4-vision-preview":
                    input_chat_log = system_set + chat_log
                completion = openai.ChatCompletion.create(
                    model=model_engine, messages=input_chat_log, request_timeout=120, max_tokens=2000
                )
                response = completion["choices"][0]["message"]["content"]
                response_list = split_string(response)
                chat_log.append(completion["choices"][0]["message"].to_dict())
                logger.info("assistant: " + completion["choices"][0]["message"].to_dict()["content"])
                if model_engine == "gpt-3.5-turbo-1106":
                    price = round_to_digits(
                        completion["usage"]["prompt_tokens"] * 0.0010 / 1000
                        + completion["usage"]["completion_tokens"] * 0.0020 / 1000,
                        3,
                    )
                elif model_engine == "gpt-4-1106-preview":
                    price = round_to_digits(
                        completion["usage"]["prompt_tokens"] * 0.01 / 1000
                        + completion["usage"]["completion_tokens"] * 0.03 / 1000,
                        3,
                    )
                    response_list.append(f"(USAGE: {price} USD)")
                elif model_engine == "gpt-4-vision-preview":
                    price = round_to_digits(
                        completion["usage"]["prompt_tokens"] * 0.01 / 1000
                        + completion["usage"]["completion_tokens"] * 0.03 / 1000,
                        3,
                    )
                    response_list.append(f"(USAGE: {price} USD)")
                logger.info(f"Usage: {price} USD")
                total_token += completion["usage"]["total_tokens"]
                if total_token > 4096 - 256:
                    chat_log = chat_log[1:]
                logger.info(chat_log)
                # logger.debug(completion)
                await msg.delete()
                for response in response_list:
                    await message.reply(response, mention_author=False)
                break
            except openai.error.Timeout as e:
                retries -= 1
                logger.exception(e)
                await reply_openai_exception(retries, message, e)
            except openai.error.InvalidRequestError as e:
                retries -= 1
                logger.exception(e)
                await reply_openai_exception(retries, message, e)
                chat_log = chat_log[1:]
            except discord.errors.HTTPException as e:
                logger.exception(e)
                await message.reply(
                    f"Discord APIでエラーが発生しました。\n{traceback.format_exception_only(e)}", mention_author=False
                )
                break
            except Exception as e:
                logger.exception(e)
                await message.reply(f"エラーが発生しました。\n{traceback.format_exception_only(e)}", mention_author=False)
                break


logger.info("Start client.")
client.run(DISCORD_BOT_TOKEN)
