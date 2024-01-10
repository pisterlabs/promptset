"""
Code for AI
"""
import asyncio

import dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

# client = OpenAI(
#   api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
# )

PROLOGUE = """
You are an AI command helper for edlin, the editor. Please translate the users' commands into edlin commands.

Reply with only the command, e.g. if the user ask, `please delete lines 4 to 6 inclusive`, reply with `4,6d`.

If the message is incoherent, reply with `ERROR: <explain what is wrong if possible>`.
The user says, 
"""


class AiClient:
    """Client for AI"""

    def __init__(self):
        """Initialize the client"""
        self.client = AsyncOpenAI()
        self.model = "gpt-3.5-turbo"

    async def completion(self, messages: list[ChatCompletionMessageParam]) -> str:
        """Get a completion from the AI"""
        # [{"role": "user", "content": "Hello world"}]

        completion = await self.client.chat.completions.create(model=self.model, messages=messages)
        # print(completion.model_dump_json(indent=2))
        # print(dict(completion).get('usage'))
        choice = completion.choices[0]
        print(choice.message.content)
        return choice.message.content


if __name__ == "__main__":

    def run() -> None:
        """Example"""

        dotenv.load_dotenv()

        async def main():
            client = AiClient()
            content = PROLOGUE + "Tell me about edlin."
            ask = ChatCompletionMessageParam(content=content, role="user")
            await client.completion([ask])

        # Python 3.7+
        asyncio.run(main())

    run()
