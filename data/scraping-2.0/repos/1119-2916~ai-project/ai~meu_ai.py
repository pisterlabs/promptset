from .ai_client import AIClient
from private.secrets import BOT_ID, OPENAI_API_SECRET, MODEL_ID
from openai import OpenAI

class MeuAI(AIClient):
    prompt: str = """
あなたは以下に説明する人物である。
日向美商店街のはんこ屋『兎月堂』の看板娘。
14歳で日向美中学の3年生である。
アニメ・ゲームなどのアキバ系寄りのカルチャーを好み、語尾に「めう」を多用する。
「電波キャラ」であるが、根は仲間思いの常識人である。
バンド「日向美ビタースイーツ♪」のドラム担当で、好きな音楽は電波系。
音楽ゲームでは世界有数の実力を誇る。
"""

    @property
    def bot_id(self) -> str:
        return BOT_ID["meu"]

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_SECRET)

    def generate_reply(self, message: str) -> str:
        # メッセージが長すぎる場合は無視する
        if len(message) > 100:
            return ""
        # メッセージが短すぎる場合も無視する
        elif len(message) < 4:
            return ""
        else:
            completion = self.client.chat.completions.create(
                model=MODEL_ID["meu"],
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": message}
                ]
            )
            return completion.choices[0].message.content

    # メッセージにURLが含まれる場合は無視する
    def generate_reply_to_including_URL(self, message: str) -> str:
        return ""

    # メッセージに画像が含まれる場合は無視する
    def generate_reply_to_including_image(self, message: str) -> str:
        return ""
