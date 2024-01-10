

import openai
import os 
from dotenv import load_dotenv,find_dotenv
__ = load_dotenv(find_dotenv()) #read local .env file
openai.api_key=os.environ["OPENAI_API_KEY"]

def get_completion(prompt,model="gpt-3.5-turbo"):
    messages=[{"role":"user","content":prompt}]
    response=openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message['content']


text=f"""
Prompt engineering is a process of creating prompts that help large language models (LLMs) generate text,
translate languages, write different kinds of creative content, and answer your questions 
in an informative way. Prompts are essentially instructions that tell the LLM what to do. 
They can be as simple as a question or as complex as a set of instructions.

The goal of prompt engineering is to create prompts that are clear, concise, and effective.
The prompts should be easy for the LLM to understand, and they should result in the desired output."""

prompt_example=f"""
Summarize the text delimited by triple backticks 
into single sentence
```{text}```
"""
response=get_completion(prompt=prompt_example)
print(response)