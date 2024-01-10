import os

import mlrun
from langchain.schema import messages_from_dict

from src.llmbot import AppConfig, build_agent, parse_agent_output

os.environ["OPENAI_API_KEY"] = mlrun.get_secret_or_env("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = mlrun.get_secret_or_env("OPENAI_API_BASE")


class QueryLLM:
    def __init__(self):
        config = AppConfig()
        self.agent = build_agent(config=config)

    def do(self, event):
        try:
            agent_resp = self.agent(
                {
                    "input": event.body["question"],
                    "chat_history": messages_from_dict(event.body["chat_history"]),
                }
            )
            event.body["output"] = parse_agent_output(agent_resp=agent_resp)
        except ValueError as e:
            response = str(e)
            if not response.startswith("Could not parse LLM output: `"):
                raise e
            event.body["output"] = response.removeprefix(
                "Could not parse LLM output: `"
            ).removesuffix("`")
        return event
