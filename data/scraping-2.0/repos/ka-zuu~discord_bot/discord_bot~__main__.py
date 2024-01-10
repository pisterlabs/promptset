# OpenAIとやり取りするDiscord Bot

from discord.ext import commands
import discord
import openai
import json
import requests

# JSONファイルから設定を読み込む
with open("config.json") as f:
    config = json.load(f)

# OpenAIの設定
openai.api_key = config["openai_api_key"]
MODEL = config["openai_model"]
PROMPT = config["openai_prompt"]

# Discord Botの設定
intents = discord.Intents.default()
intents.typing = False  # typingを受け取らないように
intents.message_content = True
TOKEN = config["discord_token"]

# Botをインスタンス化
bot = commands.Bot(
    command_prefix="$",  # $コマンド名　でコマンドを実行できるようになる
    case_insensitive=True,  # コマンドの大文字小文字を区別しない
    intents=intents,  # 権限を設定
)


# ログインしたらターミナルにログイン通知が表示される
@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")


# Botにメンションをした場合、OpenAIに問い合わせる
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if bot.user in message.mentions:
        async with message.channel.typing():
            response = await create_response_conversation(message)
        await message.reply(response)

    await bot.process_commands(message)


# サーバのグローバルIPアドレスを返すコマンド
@bot.command()
async def gip(ctx):
    try:
        global_ip = requests.get("https://api.ipify.org").text
        async with ctx.channel.typing():
            response = await create_response_infomation(f"自宅サーバのグローバルIPアドレスは{global_ip}です。")
        await ctx.send(response)
    except Exception as e:
        await ctx.send(f"エラーが発生しました: {e}")


# チャットをする関数
async def create_response_conversation(message):
    # 会話履歴を初期化
    conversations = [{"role": "system", "content": PROMPT}]
    # メッセージを会話履歴に追加
    if message.attachments:
        image_url = message.attachments[0].url
        conversations.insert(1, {"role": "user", "content": [{"type": "text", "text": message.content}, {"type": "image_url", "image_url": {"url": image_url}}]})
    else:
        conversations.insert(1, {"role": "user", "content": message.content})

    # メッセージが返信か再帰的に確認し、返信元のメッセージをすべて会話履歴に追加
    while message.reference:
        message = await message.channel.fetch_message(message.reference.message_id)
        # 返信がBotの場合はrole:assistant、ユーザーの場合はrole:userとして会話履歴に追加
        if message.author == bot.user:
            conversations.insert(1, {"role": "assistant", "content": message.content})
        else:
            conversations.insert(1, {"role": "user", "content": message.content})

    # OpenAIに問い合わせ
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=conversations,
        max_tokens=2048,
        temperature=0.8,
    )

    return response.choices[0]["message"]["content"]


# 特定の情報を返すために、返答を作成する関数
async def create_response_infomation(information):
    # 会話履歴を初期化
    conversations = [{"role": "system", "content": PROMPT}]
    # 指示を追記
    conversations.insert(
        1, {"role": "user", "content": f"{information}をあなたの言葉に言い換えてください"}
    )

    # OpenAIに問い合わせ
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=conversations,
        max_tokens=2048,
        temperature=1.0,
    )

    return response.choices[0]["message"]["content"]


bot.run(TOKEN)
