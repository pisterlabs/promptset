import re
import yaml
from typing import Dict
from typing import List
from typing import Tuple
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import ChatMessage


RoleCodec = Dict[str, str]


class ConversationAsYAML:
    def __init__(self, messages: List[ChatMessage], ai_role: str) -> None:
        self.messages = messages  # Ordered newest first
        self.ai_role = ai_role
        self.role_encoder, self.role_decoder = self.get_role_codec(self.messages, self.ai_role)

    def yaml(self, max_tokens: int) -> str:
        msg_lines = []
        tokens_used = 0
        for msg in self.messages:
            content = self.encode_discord_mentions(msg.content)
            msg_line = f'@{self.role_encoder[msg.role]} said: {content}'
            msg_line_tokens = ChatOpenAI().get_num_tokens(f'- {msg_line}')

            if tokens_used + msg_line_tokens > max_tokens:
                break

            msg_lines += [msg_line]
            tokens_used += msg_line_tokens

        # Change message order from newest first to oldest first
        msg_lines.reverse()

        return yaml.dumps(
            {
                'participants': self.role_decoder,
                'messages': msg_lines,
            }
        )

    def decode_reply(self, content: str) -> str:
        return self.decode_discord_mentions(content)

    def encode_discord_mentions(self, content: str) -> str:
        discord_mentions_regex = re.compile(r'<@(\d+)>')

        def encode_mention(match):
            return f'@{self.role_encoder[match.group(1)]}'

        return discord_mentions_regex.sub(encode_mention, content)

    def decode_discord_mentions(self, content: str) -> str:
        encoded_mentions_regex = re.compile(r'@([0-9]+)[^0-9]?\b')

        def decode_mention(match):
            return f'<@{self.role_decoder[match.group(1)]}>'

        return encoded_mentions_regex.sub(decode_mention, content)

    def get_role_codec(self, messages: List[ChatMessage], ai_role: str) -> Tuple[RoleCodec, RoleCodec]:
        roles = {ai_role} | set([m.role for m in messages])

        encoder = {
            role: str(i)
            for i, role in enumerate(roles)
        }

        decoder = {v: k for k, v in encoder.items()}

        return encoder, decoder
