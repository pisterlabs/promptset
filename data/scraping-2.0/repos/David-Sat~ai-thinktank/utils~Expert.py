from typing import Dict, Any, Callable
from langchain.schema.messages import HumanMessage, AIMessage
from utils.Worker import Worker

class Expert(Worker):
    def __init__(self, model: str, expert_instruction: Dict[str, str]) -> None:
        super().__init__(model=model)
        self.expert_instruction = expert_instruction
        self.system_prompts = self.config["expert"]['system_prompts']
        self.examples = self.config["expert"]['examples']
    
    def format_debate_history(self, debate_history: list) -> str:
        """
        Formats the debate history into a coherent string.
        """
        history_text = ""
        for entry in debate_history:
            speaker = entry['role'].capitalize()
            history_text += f"{speaker}: {entry['content']}\n"

        print(history_text)
        return history_text

    def generate_argument(self, debate: Any, stream_handler: Callable) -> str:
        # Include the debate history in the human message
        debate_history_text = self.format_debate_history(debate.debate_history)

        system_prompt = self.system_prompts["system1"].replace("##debate_topic##", debate.topic)
        system_prompt += "\n" + self.expert_instruction["instructions"] + "\n" + debate_history_text

        messages = [HumanMessage(content=system_prompt)]

        config = {
            "callbacks": [stream_handler]
        }

        for chunk in self.model.stream(messages, config=config):
            pass
