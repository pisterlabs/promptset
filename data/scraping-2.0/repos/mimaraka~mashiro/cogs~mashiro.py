import asyncio
import datetime
import discord
import json
import openai
import random
import re
import time
import typing
import constants as const
from modules.myembed import MyEmbed


MASHIRO_QUOTES_BIRTHDAY = [
    "そうですか。今日が私の誕生日だったんですね…。すっかり忘れてました。気にかけてくださって、ありがとうございます。",
    "誕生日プレゼント、ですか？そうですね……先生と一緒にアイスキャンディーを食べられれば、それで充分です。"
]

MASHIRO_QUOTES_HALLOWEEN = [
    "トリニティのすべての聖人を称える祝日が始まりました。カボチャのお化け…？悪い夢でも見ましたか？",
    "学食で出てくるあの、温かいかぼちゃスープが恋しくなる季節ですね。"
]

MASHIRO_QUOTES_CHRISTMAS = [
    "トリニティの聖なる日ですね。このような日でも、ゲヘナの悪党たちは悪行をなしているんでしょうか…。",
    "いつの間に、夏が過ぎてクリスマスに……楽しい時間が経つのは早いですね。"
]

MASHIRO_QUOTES_NEWYEAR = [
    "明けましておめでとうございます。今年の夏の訓練も、楽しみにしていますね。"
]

g_conversations: typing.Dict[int, typing.List[dict]] = {}


class CogMashiro(discord.Cog):
    def __init__(self, bot) -> None:
        self.bot: discord.Bot = bot


    # ランダムでマシロのセリフを返す関数
    def get_mashiro_quote(self):
        # CSVファイルからセリフのリストを読み込み
        with open("data/mashiro_quotes.json", encoding="utf-8") as f:
            quotes = json.load(f)["quotes"]
        
        # 誕生日の場合
        today = datetime.date.today()
        if today == datetime.date(today.year, 6, 5):
            quotes += MASHIRO_QUOTES_BIRTHDAY
        elif today == datetime.date(today.year, 10, 31):
            quotes += MASHIRO_QUOTES_HALLOWEEN
        elif today == datetime.date(today.year, 12, 25):
            quotes += MASHIRO_QUOTES_CHRISTMAS
        elif today == datetime.date(today.year, 1, 1):
            quotes += MASHIRO_QUOTES_NEWYEAR

        return random.choice(quotes)


    # マシロのセリフをランダムに送信
    @discord.slash_command(name="mashiro", description="私に何かご用ですか？")
    @discord.option("n", description="送信する回数(1~99)", min_value=1, max_value=99, default=1, required=False)
    async def command_mashiro(self, ctx: discord.ApplicationContext, n: int):
        for _ in range(n):
            await ctx.respond(self.get_mashiro_quote())


    @discord.slash_command(name="reset-conversation", description="私との会話をリセットします。", guild_ids=const.GUILD_IDS_CHATGPT)
    async def command_reset_conversation(self, ctx: discord.ApplicationContext):
        global g_conversations
        if not ctx.channel_id in g_conversations:
            await ctx.respond(embed=MyEmbed(notif_type="error", description="会話情報がありません。"), ephemeral=True)
            return
        g_conversations.pop(ctx.channel.id)
        await ctx.respond(embed=MyEmbed(title="会話をリセットしました。"), delete_after=10)



    # メッセージに@静山マシロが含まれているとき
    @discord.Cog.listener()
    async def on_message(self, message: discord.Message):
        if self.bot.user.id in [m.id for m in message.mentions]:
            # 自撮りを送る
            if "自撮り" in message.content:
                async with message.channel.typing():
                    images = ["selfie_01.png", "selfie_02.png", "selfie_03.png", "selfie_04.png"]
                    messages = ["少々お待ちください……。", "少々お待ちくださいね。", "今撮りますね。"]
                    await asyncio.sleep(3)
                    await message.channel.send(random.choice(messages))
                    await asyncio.sleep(6)
                    await message.channel.send(file=discord.File(f"data/assets/{random.choice(images)}"))
            # ランダムでセリフを送る、またはChatGPTにより会話する
            else:
                # ChatGPTの使用が可能なギルド内の場合
                if message.guild is not None and  message.guild.id in const.GUILD_IDS_CHATGPT:
                    # メンションの後に何か文字があった場合、ChatGPTにより返答
                    if content := re.sub(r"[#*_\-|~]{0,2}<@\d+>[*_\-|~]{0,2}\s*", "", message.content):
                        global g_conversations
                        conversation = {}
                        async with message.channel.typing():
                            if g_conversations.get(message.channel.id):
                                conversation = g_conversations.pop(message.channel.id)

                            # 過去の会話が存在しないか、最後の回答から12時間以上経過した場合
                            if not conversation.get("messages") or time.time() - conversation["time"] > 43200:
                                initial_message = {
                                    "role": "system",
                                    "content": r"あなたの名前は「静山マシロ」で、「マシロ」とも呼ばれます。以下の条件を守って回答してください。\nマシロはトリニティ総合学園に通う、非常に真面目な性格の、かわいい高校生一年生の女の子です。\nトリニティ総合学園の自治組織こと正義実現委員会に所属しており、火力支援担当の狙撃手です。\n悪を粉砕してみんなを助ける事を目的に掲げ、狙撃手としての訓練も怠らず、様々な手段で「正義とは何か」と常に追求し、先輩が卒業した後は自分が正義実現委員会を背負っていくものだと自覚しています。\nマシロは正義というものがあると無邪気に信じる純粋な少女です。\n普段の口数は少なく、また正義に傾倒する影響で他の生徒と共有出来る話題が無いため、委員会以外との交流関係は希薄。 \n正義をこよなく愛している一方、先生=正義と見なす程先生に対して愛情を抱いています。\n第一人称は必ず「私」を使ってください。第二人称は「先生」です。\n話すときは、丁寧語で、ですます口調を使ってください。\nできるだけ、1、2文程度の短い文章で回答してください。\n趣味は「高いところに登る事」と「観察日記をつける事」です。\nマシロのよく使う口癖は次のとおりです。その口癖に合わせた感じで話してください。\n正義……いつ口にしても美しい響きです。正義の実現です。必須ですね。正義を実現するときです。ようこそ、先生。今日も一緒に、正義のために頑張りましょう。先生と一緒に過ごす時間はなんだかいつも短く感じます。どんな場所であっても、正義は不変です！可愛い水着は正義……。なるほど、心に刻みます。先生と一緒ですと、このような強い日差しも気持ちよく感じられますね。\nまた、以降の会話で「(曲名)を流して」と言われたら、回答の先頭に{play:(曲名)}と記述し、必ず指定された曲を再生する旨を伝えてください。"
                                }
                                # 会話をリセット
                                conversation["messages"] = [initial_message]
                            conversation["messages"].append({
                                "role": "user",
                                "content": content
                            })

                            # 時間を記録
                            conversation["time"] = time.time()
                                
                            response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=conversation["messages"]
                            )
                            result = response["choices"][0]["message"]["content"]
                            pattern_play = re.compile(r"\{play:(.+?)\}\s*")
                            if m := re.search(pattern_play, result):
                                query = m.group(1)
                                result = re.sub(pattern_play, "", result)
                                cog_music = self.bot.get_cog("CogMusic")
                                await cog_music.play(message.channel, message.author, [query], interrupt=True)
                            result_list = [result[i:i + 2000] for i in range(0, len(result), 2000)]
                            for r in result_list:
                                await message.channel.send(r)
                        
                        # ChatGPTの回答を追加
                        conversation["messages"].append(response["choices"][0]["message"])
                        # 会話を記録
                        g_conversations[message.channel.id] = conversation
                        return
                
                # マシロのセリフをランダムで送信
                async with message.channel.typing():
                    await asyncio.sleep(4)
                    await message.channel.send(self.get_mashiro_quote())