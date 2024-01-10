from langchain.chat_models import ChatOpenAI
import yaml
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

    
def create_mnemonic(key: str, value: str) -> dict:
    chat = ChatOpenAI(temperature=.05)
    key = key.strip()
    value = value.strip()
    messages = [
    SystemMessage(
        content="You are Duolingo-GPT, and you help design memorable mnemonics for language learning."
    ),
    HumanMessage(
        content=
"""
Hey, I need you to create a memorable mnemonic between two words, (named "Key" and "Value"):

For example, if I to remember the Spanish phrase "de pie", which means "standing" (Key is "de pie", Value is "standing"), following mnemonic is perfect:

"Denzel Washington is standing in a freshly baked pie". 

Explanation: "de" reminds me of name "Denzel", and this reminds me of actor named Denzel Washington. "pie" reminds me of English word "pie". Taken together, denzel washington standing in a pie is very silly and so, very memorable.

Please create a mnemonic following this example, and print the output in the standard YAML format for me:

mnemonic: <mnemonic> 
explanation: <mnemonic explanation>
"""
),
    AIMessage(
        content=
f"""
Sure thing. Provide a "Key" and "Value" words and the associated word, and I will respond in that YAML format.
"""
    ),
    HumanMessage(
        content=
f"""
"Key" is "{key}", "Value" is "{value}"
"""
    )
]
    return yaml.safe_load(chat(messages=messages).content)