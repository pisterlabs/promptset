import os
import pickle

import discord
import openai

import chatgpt_api
import chatgpt_settings

# 以下2つを自分のトークンに置換
openai.api_key = os.environ["OPENAI_API_KEY"]
TOKEN = os.environ["DISCODE_HORO_API_KEY"]

INIT_TOKEN = 1000
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
system_settings = chatgpt_settings.NEO_HORO_DISCODE_BOT_SETTINGS_V3


def init_save_object():
    save_object = {}
    save_object["tokens"] = []
    save_object["messages"] = []
    save_object["tokenlog"] = []
    save_object["total_tokens_history"] = [INIT_TOKEN]
    save_object["pop_tokens"] = 0
    return save_object


@client.event
async def on_message(message):
    if message.author.bot:
        return

    try:
        save_object = init_save_object()

        if os.path.isfile("./db.pkl"):
            with open("./db.pkl", "rb") as f:
                save_object = pickle.load(f)

        if message.content == "/reset":
            await message.channel.send("RESET")
            save_object = init_save_object()
            if os.path.isfile("db.pkl"):
                os.remove("db.pkl")

        else:
            bot_message = await message.channel.send("...🍎(ムシャムシャ")
            (
                new_message,
                save_object["messages"],
                tokens,
                tokenlog,
            ) = chatgpt_api.completion(
                message.content, system_settings, save_object["messages"]
            )
            save_object["tokens"].append(tokens)
            save_object["tokenlog"].append(tokenlog)

            if len(save_object["tokens"]) == 1:
                if sum(save_object["tokens"]) > INIT_TOKEN:
                    save_object["total_tokens_history"] = [sum(save_object["tokens"])]

            else:
                save_object["total_tokens_history"].append(
                    save_object["tokens"][-1] - save_object["tokens"][-2]
                )

            await bot_message.edit(content=new_message)

            while sum(save_object["total_tokens_history"]) > 3000:
                # print("delete messages")
                save_object["pop_tokens"] = save_object["tokens"].pop(0)
                save_object["total_tokens_history"].pop(1)
                save_object["messages"].pop(1)
                save_object["messages"].pop(1)

            if save_object["pop_tokens"] > 0:
                save_object["tokens"] = list(
                    map(lambda x: x - save_object["pop_tokens"], save_object["tokens"])
                )
                save_object["pop_tokens"] = 0

            # デバック用
            # print(len(save_object["messages"]))
            # print(save_object["tokenlog"])
            # print(save_object["tokens"])
            # print(sum(save_object["total_tokens_history"]))
            # print(save_object["total_tokens_history"])

            with open("./db.pkl", "wb") as f:
                pickle.dump(save_object, f)

    except Exception as e:
        # print(e)

        await message.channel.send(f"エラーが発生しました。プログラムをリセットします。 ```Error: {e}```")
        save_object = init_save_object()
        if os.path.isfile("db.pkl"):
            os.remove("db.pkl")


client.run(TOKEN)
