import discord
import re
import openai



# Discordの設定
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')

@client.event
async def on_message(message):
    # ボット自身のメッセージは無視
    if message.author == client.user:
        return

    # ボットがメンションされているかチェック
    if client.user in message.mentions:
        # メッセージを文字列として変数に格納
        mentioned_message = message.content
        text = mentioned_message
        text = re.sub(r'<@\d*> ', '', text)
        print(text)

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

        # OpenAIからのレスポンスの先頭にメッセージの送信者へのメンションを追加して、それをDiscordに送信
        mention = f"<@1155533312438317097>"
        await message.channel.send(f"{mention} {gpt3_response}")

# Discordのボットトークンをセットしてください
client.run(YOUR_DISCORD_TOKEN)