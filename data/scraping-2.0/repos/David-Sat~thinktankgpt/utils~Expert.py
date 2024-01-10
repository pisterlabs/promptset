from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser, ChatMessage

from typing import Dict, Any, Callable
from utils.Worker import Worker

class Expert(Worker):
    def __init__(self, model: str, expert_instruction: Dict[str, str]) -> None:
        super().__init__(model=model)
        self.expert_instruction = expert_instruction
        self.system_prompts = self.config["expert"]['system_prompts']
        self.examples = self.config["expert"]['examples']
    
    
    def generate_argument(self, debate: Any, stream_handler: Callable) -> str:
        system_prompt = self.system_prompts["system1"].replace("##debate_topic##", debate.topic)
        system_prompt += "\n" + self.expert_instruction["instructions"]

        messages = [ChatMessage(role="system", content=system_prompt)]

        if len(debate.memory) > 0:
            messages.extend(debate.memory)
            reminder = f"Shortly engage with your adversaries as the {self.expert_instruction['role']}."
            messages.append(ChatMessage(role="system", content=reminder))

        debate_prompt = ChatPromptTemplate.from_messages(messages)

        chain = (
            debate_prompt
            | self.model
            | StrOutputParser()
        )

        config = {
            "callbacks": [stream_handler]
        }
        return chain.invoke(input={}, config=config)
