import asyncio
import aiohttp
import revolt
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import logging
from revolt.enums import ChannelType


logger = logging.getLogger("revolt")
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


chat_open_ai = ChatOpenAI(model="gpt-4", temperature=0.5, streaming=True)

state: dict[str, list[BaseMessage]] = {}


async def process(history: list[BaseMessage], user_input: str) -> str:
    response = await chat_open_ai.agenerate(
        [history], callbacks=[StreamingStdOutCallbackHandler()]
    )

    text = response.generations[0][0].text

    history.append(AIMessage(content=text))

    return text


class Client(revolt.Client):
    def need_reply(self, message: revolt.Message) -> bool:
        message_user = self.get_user(message.author.id)
        is_response = False
        for user in message.mentions:
            if user.id == self.user.id:
                is_response = True

        if message.channel.channel_type == ChannelType.direct_message:
            is_response = True

        return message_user.id != self.user.id and is_response

    async def on_message(self, message: revolt.Message):
        if not self.need_reply(message):
            return

        content = message.content

        if message.author.id not in state:
            state[message.author.id] = []

        history = state[message.author.id]
        history.append(HumanMessage(content=content))

        while len(history) > 10:
            history.pop(0)

        resp = await process(history, message.content)
        await message.channel.send(
            resp,
        )


async def main():
    async with aiohttp.ClientSession() as session:
        client = Client(
            session,
            "Your Bot Token Here",
            api_url="Your API URL",
        )
        await client.start()


asyncio.run(main())
