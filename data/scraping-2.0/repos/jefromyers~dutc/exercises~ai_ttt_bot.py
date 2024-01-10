import logging
from configparser import ConfigParser
from dataclasses import dataclass, field
from datetime import datetime
from socket import AF_INET, SOCK_STREAM, socket

from openai import OpenAI
from rich.logging import RichHandler

level = logging.DEBUG

logging.basicConfig(
    level=level,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logging.getLogger("openai").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("ssl").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)


@dataclass
class Message:
    msg: str
    recieved: datetime
    action: str | None = None

    @classmethod
    def from_bytes(cls, msg: bytes):
        return cls(msg=msg.decode(), recieved=datetime.now())

    def __str__(self):
        return self.msg


@dataclass
class Bot:
    gpt: OpenAI
    conn: socket
    name: str = "Boty McBotterson"
    messages: list = field(default_factory=list)
    _model: str = "gpt-4"

    def _ask_gpt(self, msg: Message) -> str | None:
        answer = None
        prompt = f"""
                Your name is: {self.name} and your a tic tac toe bot, you're playing at the letter a on the board
                You've just been given the following message:
                {msg.msg}
                What is your action?

                Your goal is to get 3 a's in a row, column, or diagonal

                Please respond with Action: <action>
                Doing nothing is a valid action, in that case please resond: Action: None

                Make sure if its not your turn to respond with Action: None

                The board positions are numbered 1 to 9 from top left to bottom right
                1 | 2 | 3
                -----------
                4 | 5 | 6
                -----------
                7 | 8 | 9
                Do not pick a position that is already taken by a or b 

                Example:
                What is your name?
                Action: {self.name}

                Example:
                a |   | b
                -----------
                  | b |
                -----------
                  |   |
                Action: 7
                """
        completion = self.gpt.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self._model,
            # response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        if content.startswith("Action:"):
            answer = content.split("Action:")[1].strip()
            if answer.lower() == "none":
                answer = None

        return answer

    def decide_action(self) -> str | None:
        # We need to decide what to do with the message
        logger.info(f"Received:\n{self.messages[-1]}\n")
        action = self._ask_gpt(self.messages[-1])

        # if action:
        #     listen = input(
        #         f"Bot says: {action}\nShould we follow instructions (y or alternative)?: "
        #     )
        #     if listen.lower() != "y":
        #         action = listen
        return action

    @classmethod
    def from_socket(cls, conn: socket, name: str, gpt: OpenAI):
        return cls(conn=conn, name=name, gpt=gpt)

    def play(self):
        while True:
            # XXX: Blocks until we get a message
            msg = Message.from_bytes(self.conn.recv(1024))
            if msg:
                self.messages.append(msg)
                action = self.decide_action()
                if action:
                    self.conn.send(action.encode())


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 4227
    name = "Mr Roboty"

    config = ConfigParser()
    config.read("./data/ttt.config")
    api_key = config["TTT"]["API_KEY"]
    organization = config["TTT"]["ORGANIZATION"]
    gpt = OpenAI(api_key=api_key, organization=organization)

    client = socket(AF_INET, SOCK_STREAM)
    client.connect((host, port))
    bot = Bot.from_socket(client, name, gpt)
    bot.play()

    client.close()
