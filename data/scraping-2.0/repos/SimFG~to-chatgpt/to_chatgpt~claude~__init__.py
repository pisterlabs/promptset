import json
import os
from typing import Optional

import anthropic
from fastapi import Request

from to_chatgpt.common import BaseAdapter, convert_messages_to_prompt_without_role, to_chatgpt_response, \
    to_chatgpt_response_stream, logger

model_map = {
    "gpt-3.5-turbo": "claude-v1.3",
    "gpt-3.5-turbo-0301": "claude-v1.3",
    "gpt-4": "claude-v1.3-100k",
    "gpt-4-0314": "claude-v1.3-100k",
}


class ClaudeAdapter(BaseAdapter):
    client: Optional[anthropic.Client] = None

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def openai_to_claude_params(openai_params):
        model = model_map.get(openai_params["model"], "claude-v1.3-100k")
        messages = openai_params["messages"]

        prompt = convert_messages_to_prompt_without_role(messages)

        claude_params = {
            "model": model,
            "prompt": f"{anthropic.HUMAN_PROMPT} Start a conversation {prompt} {anthropic.AI_PROMPT}",
            "max_tokens_to_sample": 100000 if model == "claude-v1.3-100k" else 9016,
        }

        if openai_params.get("max_tokens"):
            claude_params["max_tokens_to_sample"] = openai_params["max_tokens"]

        if openai_params.get("stop"):
            claude_params["stop_sequences"] = openai_params.get("stop")

        if openai_params.get("temperature"):
            claude_params["temperature"] = openai_params.get("temperature")

        if openai_params.get("stream"):
            claude_params["stream"] = True

        return claude_params

    async def achat(self, request: Request):
        openai_params = await request.json()
        claude_params = self.openai_to_claude_params(openai_params)
        logger.info(f"claude_params:{claude_params}")

        headers = request.headers
        auth_header = headers.get("authorization", None)
        if auth_header:
            token = auth_header.split(" ")[1]
            client = anthropic.Client(token)
        else:
            if not self.client:
                self.client = anthropic.Client(os.getenv("ANTHROPIC_API_KEY", ""))
            client = self.client

        if claude_params.get("stream"):
            response = await client.acompletion_stream(**claude_params)
            last_response = ""
            async for data in response:
                response = data.get("completion", "")
                if data.get("stop"):
                    logger.info(f"claude_stream_response:{json.dumps(data)}")
                    yield to_chatgpt_response_stream("", "stop")
                    yield "[DONE]"
                else:
                    if len(last_response) < len(response):
                        yield to_chatgpt_response_stream(response[len(last_response):], None)
                        last_response = response
                    else:
                        yield to_chatgpt_response_stream("", None)
        else:
            response = await client.acompletion(**claude_params)
            logger.info(f"claude_response:{json.dumps(response)}")
            yield to_chatgpt_response(response.get("completion", ""))
