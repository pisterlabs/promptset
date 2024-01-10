import os
import re
import subprocess
from typing import Dict, List

# load env variables
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)

load_dotenv()


def extract_code_from_response(response: str) -> Dict[str, List[str]]:
    code_blocks = re.findall(r"```(?:python|bash)?\s*[\s\S]*?```", response)

    extracted_code = {"python": [], "bash": []}

    for code_block in code_blocks:
        code_type, code = re.match(
            r"```(python|bash)?\s*([\s\S]*?)```", code_block
        ).groups()
        code_type = code_type or "python"
        extracted_code[code_type].append(code.strip())

    return extracted_code


sys_message = """
you are a natural language to eth agent. 

think about how to solve the given problem using web3.py. 

my wallet is already setup and libaries are installed. and connected using infura

return single a python function called command to complete the task like,

do not import any aditional libaries

```python

from web3 import Web3
import json

# Connect to web3 using Infura
infura_url = "https://goerli.infura.io/v3/f4149201e122477882ce3ec91ed8a37b"
web3 = Web3(Web3.HTTPProvider(infura_url))

#load the wallet data from the file
with open("wallet.json", "r") as infile:
    wallet_data = json.load(infile)
    
private_key = wallet_data["private_key"]

# Set the account address and private key
account = Account.from_key(private_key)

def function_to_complete_command():
```

return a full python file that can be run from main.py, including the example code above.

"""

from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

chat = ChatOpenAI(temperature=0)

system_message_prompt = SystemMessagePromptTemplate.from_template(sys_message)

human_template = "{text}"

human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

llm = OpenAI(temperature=0)
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory(),
)

conversation.predict(input="Hi there!")

# get a chat completion from the formatted messages
# resp = chat(chat_prompt.format_prompt(text="stake my eth").to_messages())


# code = extract_code_from_response(resp.content)

with open("test3.py", "w") as f:
    f.write("\n".join(code["python"]))


# read the file back in then run it using subprocess
with open("test3.py", "r") as f:
    code = f.read()

# run the code
subprocess.run(["python", "test3.py"], capture_output=True)
