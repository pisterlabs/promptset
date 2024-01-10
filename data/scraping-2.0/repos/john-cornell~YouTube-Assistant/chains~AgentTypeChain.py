from typing import List
from dotenv import load_dotenv
from multiagent.query_type_agent import query_type_agent, query_type_agent_input
from typing import Any, Dict, List, Optional
import json

from multiagent.json_extractor import extract_json, get_error_json
from langchain.schema.language_model import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun
)

load_dotenv()

# chat = ChatAnthropic()

# agent = query_type_agent(llm=ChatAnthropic)

class AgentTypeChain(Chain):
    llm:BaseLanguageModel
    debug: bool

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    @property
    def input_keys(self) -> List[str]:
        return ["query"]

    @property
    def output_keys(self) -> List[str]:
        return ["response"]

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None
    ) -> Dict[str, str]:

        prompt_to_analyse = inputs["query"]

        if not prompt_to_analyse:
            raise ValueError("prompt_to_analyse not set")

        agent = query_type_agent(llm=self.llm)

        try:
            output = agent.run(
                input=query_type_agent_input(prompt_to_analyse)
            )
        except Exception as e:
            print(f"Agent error: {e}")
            return get_error_json(f"Agent error: {e}", output_key="process")

        output = extract_json(output, output_key="process")

        ouput_dict = json.loads(output)
        thinking = ouput_dict["thinking"]
        process = ouput_dict["process"]
        if self.debug:
            print(f"Thinking: {thinking}")
            print()
            print(f"Process: {process}")

        return {"response": json.loads(output)}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:

        prompt_to_analyse = inputs["query"]

        if not prompt_to_analyse or prompt_to_analyse.strip() == '':
            raise ValueError("prompt_to_analyse not set")

        agent = query_type_agent(llm=self.llm)
        output = agent.arun(
            input=query_type_agent_input(prompt_to_analyse)
        )

        return {"response": json.loads(output)}

    @property
    def _chain_type(self) -> str:
        return "Agent Type Chain"