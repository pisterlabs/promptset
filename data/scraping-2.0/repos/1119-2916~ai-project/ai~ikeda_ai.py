from .ai_client import AIClient
from private.secrets import BOT_ID, OPENAI_API_SECRET, MODEL_ID
from openai import OpenAI

class IkedaAI(AIClient):
    prompt: str = """
下記に説明する人物として、短めの返答をせよ。
27歳の独身男性。名前は「いけだ」である。
日本の東京都町田市本町田出身である。
心優しく、遠慮がちであるが、気さくに会話する。
理系のオタクである。
"""

    @property
    def bot_id(self) -> str:
        return BOT_ID["ikeda"]

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
                model=MODEL_ID["ikeda"],
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
