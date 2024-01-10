import discord
import re
import deepl
import openai  


# Intentsの設定
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.reactions = True 

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print('start.')

@client.event
async def on_message(message):
    # ボット自身のメッセージは無視
    if message.author == client.user:
        return

    # ボットがメンションされているかチェック
    msg = message.content
    print(msg)
    lang = "EN-US"
    translator = deepl.Translator(DEEPL_API_KEY) 
    result = translator.translate_text(msg, target_lang=lang)
    result_txt = result.text
    print(result_txt)



    text = result_txt

        # OpenAIにテキストを送信してレスポンスを取得
    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {
            "role": "system",
            "content": "nomal GPT4",
        },
        {
            "role": "user",
            "content": text,
            }
        ],
    )
    gpt3_response = response["choices"][0]["message"]["content"]
    print(gpt3_response)

    msg = gpt3_response
    print(gpt3_response)
    lang = "Ja"

    translator = deepl.Translator(DEEPL_API_KEY) 
    result = translator.translate_text(msg, target_lang=lang)
    result_txt = result.text

    await message.channel.send(result_txt)
    print(result_txt)
client.run(TOKEN)

