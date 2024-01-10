import os
from typing import Dict, List

import openai


class ChatCompletion:
    model: str
    messages: List[Dict]
    # functions: List[T]  # TODO: Would be fun to implement this.
    # temperature: float
    # top_p: float
    # n: int
    # stream: bool
    # stop: str | List[str] | None
    # max_tokens: int
    # presence_penalty: int
    # frequency_penalty: int
    # logit_bias: Dict[str, float] | None
    # user: str | None
    
    def __init__(self, model:str='gpt-3.5-turbo',
                 system_content:str='You are a helpful assistant.',
                 messages:List[Dict]=[{}]) -> None:
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        self.model = model
        if messages == [{}]:
            self.messages = [{"role": "system", "content": f"{system_content}"}]
        else:
            self.messages = messages
    
    def post_user_content(self, user_content:str) -> str:
        self.messages.append({"role": "user", "content": f"{user_content}"})

        chat_completion_response = openai.ChatCompletion.create(
        model=self.model, 
        messages=self.messages)
    
        response = chat_completion_response['choices'][0]['message']['content'] # type: ignore
        self.messages.append({"role": "assistant", "content": f"{response}"})

        return response
    
    def get_messages(self) -> List[Dict]:
        return self.messages
