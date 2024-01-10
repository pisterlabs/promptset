from dotenv import load_dotenv
import os
from openai import OpenAI
import requests
from .templates import TEMPLATE

load_dotenv()
api_key = os.environ["OPENAI_KEY"]
org = os.environ["OPENAI_ORG"]

def create_prompt(inputs, test_inputs, config):
    return TEMPLATE.format(
        header=config.header.strip("\n"),
        inputs=inputs.strip("\n"),
        task=config.task,
        test_inputs=test_inputs.strip("\n"),
        output_format=config.output_format.strip("\n"),
    )


class Messages:
    def __init__(self, system="") -> None:
        self.messages = []
        self.responses = []
        if system != "":
            self.messages.append({"role": "system", "content": system}) 
    
    def append(self, message, user=True, system=False):
        if message.strip("\n").strip() == "":
            return
        if system:
            self.messages.append({"role": "system", "content": message.strip("\n")})
            return
        message = {"role": "user" if user else "assistant", "content": message.strip("\n")}
        self.messages.append(message)

    def get(self):
        return self.messages
    
    def __str__(self) -> str:
        ans = ""
        for message in self.messages:
            ans += f"{message['role']}: {message['content']}\n"
        return ans
        
def post_to_gpt(model_name, messages, nucleus_p=0.1, temperature=0.1, return_response=False):
    client = OpenAI(api_key=api_key, organization=org)

    response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    # temperature=temperature,
    max_tokens=1000,
    top_p=nucleus_p,
    )
    if return_response:
        return response
    return response.choices[0].message.content

if __name__ == "__main__":
    response = post_to_gpt("gpt-3.5-turbo-1106", "This is a test prompt.", return_response=True)
    print(response.model_dump_json())
    print("=====================================")
    print(response.choices[0].message.content)
    print("=====================================")
    print(len(response.choices))