from typing import Optional, Callable

from langchain.schema import HumanMessage, SystemMessage
from langchain.input import print_text

from llm_oracle import llm
from llm_oracle.agents.base import OracleAgent
from llm_oracle.agents.output import OUTPUT_PROMPT_P10, parse_p10_output, OUTPUT_PROMPT_LIKELY, parse_likely_output
from llm_oracle.markets.base import MarketEvent


PROMPT_SYSTEM_BASIC_AGENT_V1 = """
You are an expert analyst with tons of already knowledge. You already know everything and you never say "I don't know".

Given the users prediction question, answer to the best of your ability and follow their instructions.
"""

PROMPT_HUMAN_BASIC_AGENT_V1 = f"""
Predict the outcome of the following event.

{{event_text}}

Respond with what you already might know about this.

{OUTPUT_PROMPT_P10}
"""


class BasicAgentv1(OracleAgent):
    def __init__(
        self,
        verbose: Optional[bool] = True,
        model: Optional[llm.BaseChatModel] = None,
        output_callback: Optional[Callable[[str], None]] = None,
    ):
        self.model = model or llm.get_default_llm()
        self.verbose = verbose
        self.output_callback = output_callback

    def get_system_prompt(self) -> str:
        return PROMPT_SYSTEM_BASIC_AGENT_V1

    def get_human_prompt(self) -> str:
        return PROMPT_HUMAN_BASIC_AGENT_V1

    def parse_output(self, result: str) -> float:
        return parse_p10_output(result)

    def predict_event_probability(self, event: MarketEvent) -> float:
        event_text = event.to_text()
        if self.verbose:
            print_text(event_text + "\n", "green")
        human_message = self.get_human_prompt().format(event_text=event_text)
        result = self.model([SystemMessage(content=self.get_system_prompt()), HumanMessage(content=human_message)])
        if self.output_callback:
            self.output_callback(result.content)
        if self.verbose:
            print_text(result.content + "\n", "yellow")
        return self.parse_output(result.content)


PROMPT_HUMAN_BASIC_AGENT_V2 = f"""
Predict the outcome of the following event.

{{event_text}}

Respond with:
* What you learned might know about this.
* Arguments for the event
* Arguments against the event

{OUTPUT_PROMPT_P10}
"""


class BasicAgentv2(BasicAgentv1):
    def get_system_prompt(self) -> str:
        return PROMPT_SYSTEM_BASIC_AGENT_V1

    def get_human_prompt(self) -> str:
        return PROMPT_HUMAN_BASIC_AGENT_V2


PROMPT_HUMAN_BASIC_AGENT_V3 = f"""
Predict the outcome of the following event.

{{event_text}}

Respond with:
* What you learned might know about this.
* Arguments for the event
* Arguments against the event

{OUTPUT_PROMPT_LIKELY}
"""


class BasicAgentv3(BasicAgentv1):
    def get_system_prompt(self) -> str:
        return PROMPT_SYSTEM_BASIC_AGENT_V1

    def get_human_prompt(self) -> str:
        return PROMPT_HUMAN_BASIC_AGENT_V3

    def parse_output(self, result: str) -> float:
        return parse_likely_output(result)
