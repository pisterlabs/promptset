import asyncio
from dataclasses import dataclass

from tindermate.configuration import Configuration
from tindermate.conversation.agent import ConversationAgent
from tindermate.conversation.gpt import OpenAIAuthError
from tindermate.tinder.client import create_tinder_client
from tindermate.tinder.exception import TinderAuthError


class InvalidTokenError(Exception):
    pass


@dataclass
class Tokens:
    openai_token: str | None = Configuration.OPENAI_API_KEY
    tinder_token: str | None = Configuration.TINDER_AUTH_TOKEN

    @classmethod
    def load(cls) -> "Tokens":
        print("loading tokens from " + str(Configuration.TOKEN_FILE))
        lines = Configuration.TOKEN_FILE.read_text().splitlines()
        openai, tinder = None, None
        if len(lines) >= 2:
            openai, tinder = lines[:2]
        return Tokens(
            openai_token=openai or Configuration.OPENAI_API_KEY,
            tinder_token=tinder or Configuration.TINDER_AUTH_TOKEN,
        )

    def save(self) -> None:
        print("saving tokens to " + str(Configuration.TOKEN_FILE))
        lines = "\n".join([self.openai_token or "", self.tinder_token or ""])
        Configuration.TOKEN_FILE.write_text(lines)


async def validate_tokens(tokens: Tokens) -> None:
    if tokens.tinder_token is None:
        raise InvalidTokenError("Tinder token is not valid")
    if tokens.openai_token is None:
        raise InvalidTokenError("Open AI token is not valid")

    tinder = create_tinder_client(auth_token=tokens.tinder_token)
    agent = ConversationAgent(api_key=tokens.openai_token)

    try:
        await asyncio.gather(tinder.current_user_info(), agent.test_connection())
    except TinderAuthError as exc:
        print("Validation failed because because tinder token is invalid")
        raise InvalidTokenError("Tinder token is not valid") from exc
    except OpenAIAuthError as exc:
        print("Validation failed because because open AI token is invalid")
        raise InvalidTokenError("Open AI token is not valid") from exc
