
import asyncio
from reply_config import REPLY_TO, CONTEXT_MESSAGE
from websockets.server import serve
from llm.gpt4free_api import GPT4FreeAPI
from llm.openai_api import OpenAIAPI
from discord_messenger import DiscordMacroMessage, DiscordRequestMessenger
import json


def load_config() -> dict:
    with open("config.json", "r") as f:
        return json.load(f)

class AutoReply:
    def __init__(self, config: dict) -> None:
        super().__init__()
        self._messenger = None
        self._config : dict = config
        self._select_llm()
        print("Configuration loaded, AutoReply Server ready. Waiting for WebSocket connection...")
    
    def _select_llm(self) -> None:
        if self._config["OPENAI_API_KEY"] != "":
            print("OpenAI API Key found, using OpenAI API for responses")
            self._llm = OpenAIAPI(3, self._config["OPENAI_API_KEY"], CONTEXT_MESSAGE)
        else:
            print("Defaulting to GPT4Free API for responses")
            self._llm = GPT4FreeAPI(4)

    async def reply(self, websocket) -> None:
        print("WebSocket connection established. Initializing Discord Messenger...")
        if self._config["DISCORD_AUTHORIZATION"] != "":
            print("Discord Channel ID and Authorization found, using requests to send messages")
            self._messenger = DiscordRequestMessenger(self._config["DISCORD_AUTHORIZATION"])
        else:
            print("Defaulting to Discord Macro Message for sending messages, preparing to capture window position...")
            self._messenger = DiscordMacroMessage()
        print("Discord Messenger initialized. Waiting for messages...")
        async for message in websocket:
            msg_json = json.loads(message)
            msg_tuple = (msg_json["author"], msg_json["channel"])
            if msg_tuple not in REPLY_TO or msg_json["content"] == "":
                continue
            print(f"Received message: {message}")
            response: str = self._llm.get_response(msg_json["content"])
            print(f"Sending response: {response}")
            self._messenger.send_message(response, msg_json["channel"]) # Reply to the same channel
            await websocket.send("")

    
    
    async def start_service(self) -> None:
        async with serve(self.reply, self._config["WEBSOCKET_HOST"], self._config["WEBSOCKET_PORT"]):
            await asyncio.Future()

if __name__ == "__main__":
    replier = AutoReply(load_config())
    asyncio.run(replier.start_service())