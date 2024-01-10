"""
GOALS: watch -> LM (few shot or feature generator) -> "watch target" t_w

TODO:
extra inputs (e.g. stroke count, distance)

"""
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from params_proto import PrefixProto
import pandas as pd
from dataclasses import dataclass
from typing import Literal, Union, List


class ModelArgs(PrefixProto):
    # https://platform.openai.com/docs/models
    model_name: Literal["gpt-4-1106-preview", "gpt-3.5-turbo-1106"] = "gpt-4-1106-preview"


@dataclass
class TextGenerator:
    dataset: Union[str, List[str]]
    model_name: str = ModelArgs.model_name
    conversation_history: list = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []

        if isinstance(self.dataset, str):
            self.df = pd.read_csv(self.dataset)
        else:
            self.df = [pd.read_csv(d) for d in self.dataset]

        self.agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0, model=self.model_name), self.df,
                                                   prefix="Remove any ` from the Action Input",
                                                   # agent_type=AgentType.OPENAI_FUNCTIONS,
                                                   agent_executor_kwargs={"handle_parsing_errors": True},
                                                   verbose=True)
        # self.agent = create_csv_agent(ChatOpenAI(temperature=0, model=self.model_name), self.dataset, verbose=True)

    def __call__(self, prompt, verbose=True):
        self.conversation_history.append(prompt)

        full_prompt = "\n".join(self.conversation_history)

        tool_input = {
            "input": {
                "name": "python",
                "arguments": full_prompt,
            }
        }

        out = self.agent.run(tool_input)

        # Append the response to the conversation history
        self.conversation_history.append(out)

        return out

    def reset_conversation(self):
        # Clear the conversation history
        self.conversation_history = []


if __name__ == '__main__':
    prompt = "This was a swimming workout. What genre of music would be good to listen to? Using all of the data provided, come to a conclusion. Summarize it as a playlist description, and only return the description."
    #gen = TextGenerator("../example_data/swim_merged.csv", ModelArgs.model_name)
    #lm_output = gen(prompt)
    #print(lm_output)
