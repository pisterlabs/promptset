from typing import Optional, List
import openai

from config.prompt import personal_copilot

class Approach:
    def __init__(
            self,  
            chatgpt_deployment: str
            ):
        self.chatgpt_deployment = chatgpt_deployment
        self.system_prompt = personal_copilot["1"]["system"]

    def run(self, history: List, name: Optional[str]=None) -> any:

        system_prompt = self.system_prompt
        conversation = [
            {"role": "system", "content": system_prompt}
            ] + history
        
        response = openai.ChatCompletion.create(
            model=self.chatgpt_deployment,
            temperature=0.2,
            messages = conversation)
        
        return {"data_points": "", "answer": response['choices'][0]['message']['content'], "thoughts": ""}
