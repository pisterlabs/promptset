# discord.pyの大事な部分をimport

import discord
from discord.ext import commands, tasks
import os
import asyncio
import openai
from channel import Channel, Mode
from discord import app_commands
from judging_puns import scoring
import MeCab
import random
import datetime as dt
import re


from dotenv import load_dotenv
load_dotenv(".env")
m = MeCab.Tagger()

# デプロイ先の環境変数にトークンをおいてね
API_TOKEN = os.environ["DISCORD_BOT_TOKEN"]
openai.api_key = os.environ["OPENAI_API_KEY"]

# botのオブジェクトを作成(コマンドのトリガーを!に)
bot = commands.Bot(
    command_prefix="/",
    intents=discord.Intents.all(),
    application_id=os.environ["APPLICATION_ID"],
    # activity=discord.Game(name="XX") #"XXをプレイ中"にする

)

tree = bot.tree

rooms = os.environ["ROOM_ID"].split(",")

channels = {int(channel): Channel(int(channel)) for channel in rooms}


def is_question(message: discord.Message):
    # コマンドや二重スラッシュは無視
    if message.content[:2] == "//" or message.content[0] == "/":
        return False
    # チャンネルIDが該当しない場合の排除
    if message.channel.id not in channels:
        return False
    # 自分自身の排除
    if message.author.id == bot.user.id:
        return False
    # ボット許可
    if channels[message.channel.id].unconditional:
        return True
    # メッセージ主がボットなら排除
    if message.author.bot:
        return False

    return True


def has_twitter_link(txt: str) -> bool:
    try:
        if re.search(r"https?://twitter.com/\w+/status/\d+", txt):
            print(re.match(r"https?://twitter.com/\w+/status/\d+", txt))
            return True
        if re.search(r"https?://x.com/\w+/status/\d+", txt):
            print(re.match(r"https?://x.com/\w+/status/\d+", txt))
            return True
        return False
    except Exception as e:
        print(e)
        return False


@tree.command(name="join", description="臨時でチャンネルに参加するよ、しばらくたつと反応しなくなるよ")
async def join(interaction: discord.Interaction):
    await interaction.response.defer()
    if interaction.channel.id in channels:
        return await interaction.followup.send("既に参加しているよ")

    channels[interaction.channel.id] = Channel(
        interaction.channel.id, is_temporary=True)
    return await interaction.followup.send("こんちゃ！！")


@tree.command(name="bye", description="臨時で参加しているチャンネルから脱退するよ")
async def bye(interaction: discord.Interaction):
    await interaction.response.defer()
    if interaction.channel.id not in channels:
        return await interaction.followup.send("いないよ……")
    if channels[interaction.channel.id].is_temporary:
        del channels[interaction.channel.id]
        return await interaction.followup.send("ばいばい！！")
    else:
        return await interaction.followup.send("何らかの理由で退場できないよ！")


@tree.command(name="reset", description="そのチャンネルの会話ログをリセットするよ")
async def reboot(interaction: discord.Interaction):
    channels[interaction.channel.id].reset()
    await interaction.response.send_message("リセットしたよ！")


@tree.command(name="token", description="現在のトークン消費状況を表示するよ")
async def token(interaction: discord.Interaction):
    channel = channels[interaction.channel.id]
    await interaction.response.send_message(f"現在の利用しているトークンの数は{channel.get_now_token()}だよ！\n{channel.TOKEN_LIMIT}に達すると古いログから削除されていくよ！\n(base:{channel.base_token},history:{sum([x.token for x in channel.history])})")


@tree.command(name="history", description="現在残っている会話ログを表示するよ、結構出るよ")
async def talk_history(interaction: discord.Interaction):
    channel = channels[interaction.channel.id]
    text = ""
    if not channel.history:
        return await interaction.response.send_message("会話ログはまだないよ！")
    for msg in channel.history:
        c = msg.content[:20].replace('\n', '')
        text += f"{msg.token}:{c}{'...' if len(msg.content)>20 else ''}\n"

    await interaction.response.send_message(text)


@tree.command(name="generate", description="OpenAIのAPIにアクセスして画像を生成するよ")
@app_commands.describe(prompt="生成する画像を指定する文章を入力してね")
@app_commands.describe(size="生成する画像のサイズを指定するよ")
@app_commands.choices(
    size=[
        app_commands.Choice(name="1024x1024", value="1024x1024"),
        app_commands.Choice(name="1792x1024", value="1792x1024"),
        app_commands.Choice(name="1024x1792", value="1024x1792"),
    ]
)
async def generate(interaction: discord.Interaction, prompt: str, size: app_commands.Choice[str] | None = None):
    print(f"prompt:'{prompt}'")
    if prompt == "":
        await interaction.response.send_message("`/generate rainbow cat`のように、コマンドの後ろに文字列を入れてね！")
    else:
        #  考え中にする 送信するときはinteraction.followupを使う
        await interaction.response.defer()
        if size is None:
            size = "1024x1024"
        else:
            size = size.value
        try:
            response = openai.Image.create(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size=size
            )
            image_url = response['data'][0]['url']
            img: discord.Embed = discord.Embed(
                title=prompt[:255], color=0xffffff)
            img.set_image(url=image_url)
            print(image_url)
            await interaction.followup.send(content=prompt, embed=img)
        except Exception as e:
            await interaction.followup.send(f"エラーだよ！\n```{e}```")


@tree.command(name="normal", description="通常のChatGPTモードに切り替えるよ 会話ログは消えるよ")
async def normal(interaction: discord.Interaction):
    await interaction.response.defer()
    channel = channels[interaction.channel.id]
    if channel.mode == Mode.chatgpt:
        return await interaction.followup.send("既に現在ChatGPTモードです")
    else:
        channel.mode = Mode.chatgpt
        channel.reset()
        return await interaction.followup.send("ChatGPTモードに変更しました")


@tree.command(name="tsumugi", description="つむつむモードに切り替えるよ 会話ログは消えるよ")
async def tsumugi(interaction: discord.Interaction):
    await interaction.response.defer()
    channel = channels[interaction.channel.id]
    if channel.mode == Mode.tsumugi:
        return await interaction.followup.send("もうつむつむモードだよ")
    else:
        channel.mode = Mode.tsumugi
        channel.reset()
        return await interaction.followup.send("つむつむモードに変更したよ")


@tree.command(name="mecab", description="mecabの導入が出来ているかのテストコマンドだよ 形態素解析できるよ")
async def mecab(interaction: discord.Interaction, arg: str):
    await interaction.response.send_message(m.parse(arg))
    # await interaction.response.send_message("工事中！ごめんね！")


@tree.command(name="test", description="スラッシュコマンドが機能しているかのテスト用コマンドだよ")
async def test(interaction: discord.Interaction):
    print("ちゃろ～☆")
    await interaction.response.send_message("ちゃろ～☆")


@tree.command(name="allow", description="botとの会話を許可するよ、ボット相手に無限に話す可能性があるから注意！")
async def allow(interaction: discord.Interaction):
    channel = channels[interaction.channel.id]
    if channel.unconditional:
        return await interaction.response.send_message("既に許可されているよ")
    else:
        channel.set_unconditional(True)
        return await interaction.response.send_message("会話対象を無条件にしたよ")


@tree.command(name="disallow", description="ボットとの会話の許可を取り消すよ")
async def disallow(interaction: discord.Interaction):
    channel = channels[interaction.channel.id]
    if channel.unconditional:
        channel.set_unconditional(False)
        return await interaction.response.send_message("会話対象の設定を元に戻したよ")
    else:

        return await interaction.response.send_message("元々許可されていないよ")


@tree.command(name="dajare", description="ダジャレモードを切り替えるよ(WIP)")
async def dajare(interaction: discord.Interaction):
    channel = channels[interaction.channel.id]
    if channel.dajare:
        channel.dajare = False
        return await interaction.response.send_message("ダジャレモードをオフにしたよ")
    else:
        channel.dajare = True
        return await interaction.response.send_message("ダジャレモードをオンにしたよ")


@tree.command(name="minesweeper", description="マインスイーパーを生成するよ")
async def minesweeper(interaction: discord.Interaction, x: int = 10, y: int = 10, bomb: int = 10):
    field = [[0 for _ in range(x)] for _ in range(y)]
    zenkaku = ["０", "１", "２", "３", "４", "５", "６", "７", "８", "９"]
    if bomb > x * y:
        return await interaction.response.send_message("爆弾の数が多すぎるよ")
    pair = [(i, j) for i in range(x) for j in range(y)]

    for i in random.sample(pair, k=bomb):
        field[i[1]][i[0]] = 9

    for i in range(x):
        for j in range(y):
            if field[j][i] == 9:
                continue
            for ii in range(max(0, i - 1), min(x, i + 2)):
                for jj in range(max(0, j - 1), min(y, j + 2)):
                    if field[jj][ii] == 9:
                        field[j][i] += 1

    text = ""
    for i in range(x):
        text += "# "
        for j in range(y):
            if field[j][i] == 9:
                text += "|| Ｘ || "
            else:
                text += f"|| {zenkaku[field[j][i]]} || "
        text += "\n"
    if len(text) > 1000:
        return await interaction.response.send_message("フィールドが大きすぎるよ")
    await interaction.response.send_message(text)


@tree.command(name="user_info", description="ユーザーの情報を表示するよ")
async def user_info(interaction: discord.Interaction):
    user = interaction.user
    text = f"名前:{user.name}\n"
    text += f"ID:{user.id}\n"
    text += f"display_name:{user.display_name}\n"
    text += f"global_name:{user.global_name}\n"
    await interaction.response.send_message(text)


@tree.command(name="destruction", description="パラメータを破壊するよ")
@app_commands.describe(precense_penalty="すでに存在するワードへのペナルティ(-2.0<x<2.0)", frequency_penalty="頻度に対するペナルティ(-2.0<x<2.0)")
async def destruction(interaction: discord.Interaction, precense_penalty: float = -2.0, frequency_penalty: float = -2.0):
    if interaction.channel.id not in channels:
        return await interaction.response.send_message("いないよ……")
    channel = channels[interaction.channel.id]
    channel.precense_penalty = precense_penalty
    channel.frequency_penalty = frequency_penalty
    await interaction.response.send_message("破壊したよ")


@tree.command(name="regeneration", description="パラメータを戻すよ")
async def regeneration(interaction: discord.Interaction):
    if interaction.channel.id not in channels:
        return await interaction.response.send_message("いないよ……")
    channel = channels[interaction.channel.id]
    channel.precense_penalty = 0.0
    channel.frequency_penalty = 0.0
    await interaction.response.send_message("戻したよ")


@tree.command(name="secret", description="ひみつの鍵を入れられるよ")
@app_commands.describe(secret_key="ひみつの鍵")
async def secret(interaction: discord.Interaction, secret_key: str):
    if interaction.channel.id not in channels:
        return await interaction.response.send_message("いないよ……")
    channel = channels[interaction.channel.id]
    if channel.model != "gpt-3.5-turbo-0613":
        return await interaction.response.send_message("すでに切り替わっているよ!", ephemeral=True)
    if channel.secret_key_count <= 0:
        return await interaction.response.send_message("今日の分のチャンスがないよ！", ephemeral=True)
    if secret_key == os.environ["SECRET_KEY"]:
        channel.model = "gpt-4-0613"
        await interaction.response.send_message("gpt4に変更したよ！開発者の財布を破壊しよう！")
    else:
        channel.secret_key_count -= 1
        await interaction.response.send_message(f"鍵が違うよ！今日のチャンスはあと{channel.secret_key_count}回だよ", ephemeral=True)


@bot.event
# botの起動が完了したとき
async def on_ready():
    print("Logged in as")
    print(bot.user.name)
    print(bot.user.id)
    print("------")
    try:
        await tree.sync()
    except Exception as e:
        print(e)
    print("-synced-")
    await notif.start()


@tasks.loop(seconds=60)
async def notif():

    tz = dt.timezone(dt.timedelta(hours=9))
    now = dt.datetime.now(tz)
    if now.hour == 23 and now.minute == 59:
        notif_channels = os.environ["NOTIF_CHANNEL"].split(",")
        if not notif_channels:
            return
        for channel_id in notif_channels:
            channel = bot.get_channel(int(channel_id))
            await channel.send("今日はとっても楽しかったね。明日はも～っと楽しくなるよね。ねっハム太郎♪")

errmsg = "err:The server had an error processing your request."


@bot.event
async def on_message(message: discord.Message):
    # Twitterのリンクを含むメッセージのリンクをvxtwitterに置換してリプライ
    try:
        print(has_twitter_link(message.content))

    except Exception as e:
        print(e)
    if has_twitter_link(message.content):
        # twitter.comとx.comのリンクをピックアップして
        # vxtwitterのリンクに変換して改行区切りで返す
        links = re.findall(
            r"https?://(?:twitter|x)\.com/\w+/status/\d+", message.content)
        link = ""
        for l in links:
            if "twitter.com" in l:
                link += l.replace("https://twitter.com", "https://vxtwitter.com")
            else:
                link += l.replace("https://x.com", "https://vxtwitter.com")
        return await message.reply(link)

    elif not is_question(message):
        print("not question")
        return await bot.process_commands(message)

    channel = channels[message.channel.id]
    try:
        if channel.dajare:
            score, rep = scoring(message.content)
            if rep:
                channel.hiscore = max(channel.hiscore, score)
                rep += "\n現在のハイスコア:" + str(channel.hiscore)
                await message.channel.send(rep)
                return await bot.process_commands(message)

    except Exception as e:
        print(e)

    try:
        msg = await message.reply("考え中……")
        reply = ""
        chunk_size = 50
        next_chunk = chunk_size
        response = channel.send(
            message.author.display_name+' : '+message.content)
        for chunk in response:
            reply += chunk
            if len(reply) > next_chunk:
                try:
                    await msg.edit(content=reply)
                except Exception as e:
                    msg = await message.channel.send(reply)
                next_chunk += chunk_size
        try:
            all_token = channel.get_now_token()
            completion_token = channel.history[-1].token
            all_token -= completion_token
            if channel.model == "gpt-3.5-turbo-0613":
                prompt_cost = 0.0015
                completion_cost = 0.002
            elif channel.model == "gpt-4-0613":
                prompt_cost = 0.03
                completion_cost = 0.06
            else:
                price = 0
            prompt_price = prompt_cost*(all_token/1000)
            completion_price = completion_cost*(completion_token/1000)
            reply += f"\n\nlog_token: {all_token}x({prompt_cost}/1K)=${prompt_price:.4}\ncompletion_token: {completion_token}x({completion_cost}/1K)=${completion_price:.4}\n消費: ${prompt_price+completion_price:.4}"
            await msg.edit(content=reply)

        except Exception as e:
            msg = await message.channel.send(reply)

    # APIの応答エラーを拾う
    except openai.error.InvalidRequestError as e:
        reply = f"err:情報の取得に失敗したみたい\nもう一回試してみてね\n```{e}```"
    except openai.error.APIConnectionError as e:
        reply = f"err:OpenAIのAPIに接続できなかったみたい\nもう一回試してみてね\n```{e}```"
    except openai.error.APIError as e:
        reply = f"err:OpenAIのAPIに接続できなかったみたい\nもう一回試してみてね\n```{e}```"
    except Exception as e:
        reply = f"err:なにかエラーが起こってるみたい、なんかいろいろ書いとくから、開発者に見せてみて\n```{e}```"
    finally:
        if reply[:4] == "err:":
            channel.history.pop()
            await message.channel.send(reply)
    # コマンド側にメッセージを渡して終了
    await bot.process_commands(message)


async def main():
    # start the client
    async with bot:
        try:

            await bot.start(API_TOKEN)
        except KeyboardInterrupt:
            await bot.close()
        except Exception as e:
            print(e)

asyncio.run(main())
