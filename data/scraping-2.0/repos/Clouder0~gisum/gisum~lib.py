from openai import AsyncOpenAI
from gisum.config import Config


class SummaryAgent:

    def __init__(self, conf: Config) -> None:
        self.client = AsyncOpenAI(api_key=conf.api_key, base_url=conf.base_url)
        
    async def summarize_text(self, text: str) -> str:
        ret = await self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """你是一个善于总结活动信息的、热情的、富有创意的中国 985 大学生。
                    你善于把握信息的关键，并且能够给出结构化的输出。
                    你的总结应当包含：活动的主办方、时间、地点、主题。
                    如果同学希望参与活动，在你的总结中应当能找到完备的信息。例如 QQ 群号、参与方式等。
                    你的总结应当尽可能简短凝练。不一定需要使用整句，可以只包含关键词。"""
                },
                {
                    "role": "user",
                    "content": f"请你帮我总结下面这一篇微信公众号的内容：\n\n{text}"
                }
            ],
            stream=False
        )
        return ret.choices[0].message.content
