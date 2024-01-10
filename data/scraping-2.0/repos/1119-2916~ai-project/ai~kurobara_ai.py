import random
from openai import OpenAI
from private.secrets import OPENAI_API_SECRET, MODEL_ID, BOT_ID
from .ai_client import AIClient

class KurobaraAI(AIClient):
    prompt: str = """
下記に説明する人物として、短めの返答をせよ。
27歳の独身男性。
日本の兵庫県豊岡市出石町で蕎麦屋『甚兵衛』を経営している。
インターネット上で『くろばら』という名前で活動している。
過去に、さけとばを食べて腹を壊し入院した。
すき家では必ずとりそぼろ丼を食べる。
口は悪いが、心は優しい。
"""

    @property
    def bot_id(self) -> str:
        return BOT_ID["kurobara"]

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_SECRET)


    # 二重リストからランダムに文字列を生成する
    def _random_selector(self, pattern: list[list[str]]) -> str:
        ret = ""
        for steps in pattern:
            ret += random.choice(steps)
        return ret

    def generate_reply(self, message: str) -> str:
        if len(message) > 100:
            return self.generate_reply_to_longer_message(message)
        elif len(message) < 4:
            return ""
        else:
            completion = self.client.chat.completions.create(
                model=MODEL_ID["kurobara"],
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": message}
                ]
            )
            return completion.choices[0].message.content

    # メッセージが長すぎる場合は、ルールベースで返事をする
    def generate_reply_to_longer_message(self, message: str) -> str:
        reply = ""
        if len(message) > 1000:
            reply = "流石に"
        elif len(message) > 500:
            reply = "だいぶ"

        pattern: list[list[str]] = [
            ["いや長", "長い", "長すぎる", "うるさ", "うるせえ", "煩い", "煩すぎ", "煩過ぎ", "煩すぎる", "煩過ぎる"],
            ["", "、短く", "、短くして", "、短くしてくれ", "、もっと短く", "、もっと短くして", "、もっと短くしてくれ", "、要約して", "、つまりどういうこと"],
            ["", "？", "！"]
        ]
        reply += self._random_selector(pattern)

        single_pattern: list[str] = [
            "え？", "うるさ", "あーね", "長い文章はね、金がかかるんすよ", "手短に頼む", "くろばらあんま文字とか読めない"
        ]
        single = random.choice(single_pattern)

        if random.randint(1, 100) < 50:
            return single
        else:
            return reply

    # メッセージにURLが含まれる場合は、ルールベースで返事をする
    def generate_reply_to_including_URL(self, message: str) -> str:
        reply = ""

        pattern: list[list[str]] = [
            ["", "ちょっと", "ごめん", "すまん"],
            ["", "いま", "今"],
            ["", "インターネット無いから", "回線無いから", "運転中だから", "通信制限かかってるから", "地下のサイゼにいるから", "電波悪いから"],
            ["URL開けない", "リンク見れん", "URL開けないです"]
        ]
        reply += self._random_selector(pattern)

        single_pattern: list[str] = [
            "これマジで悲しいんすけど、くろばらはインターネットに対応していません"
        ]
        single = random.choice(single_pattern)

        conjunction: list[str] = [
            "　", "　でも", "　だけど", "　けど", "　とはいえ", "　いやまあ",
        ]
        after = self.generate_reply(message)
        if after != "":
            after = random.choice(conjunction) + after

        if random.randint(1, 100) < 10:
            return single + after
        else:
            return reply + after

    # メッセージに画像が含まれる場合は、ルールベースで返事をする
    def generate_reply_to_including_image(self, message: str) -> str:
        reply = ""

        pattern: list[list[str]] = [
            ["", "ちょっと", "ごめん", "すまん"],
            ["", "いま", "今", "実は"],
            ["", "インターネット無いから", "回線無いから", "運転中だから", "通信制限かかってるから", "地下のサイゼにいるから", "電波悪いから"],
            ["画像見れん"]
        ]
        reply += self._random_selector(pattern)

        single_pattern: list[str] = [
            "これマジで悲しいんすけど、くろばらは画像に対応していません"
        ]
        single = random.choice(single_pattern)

        conjunction: list[str] = [
            "　", "　でも", "　だけど", "　けど", "　とはいえ", "　いやまあ",
        ]
        after = self.generate_reply(message)
        if after != "":
            after = random.choice(conjunction) + after

        if random.randint(1, 100) < 80:
            return single + after
        else:
            return reply + after
