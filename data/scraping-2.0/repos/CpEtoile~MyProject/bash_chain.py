from langchain_experimental.llm_bash.base import LLMBashChain
from langchain.llms import OpenAI

import os
openai_api_key = os.environ.get('openai_api_key')

llm = OpenAI(openai_api_key=openai_api_key, temperature=0)


text = "Please write a bash script removes a folder named test-folder and save the bash codes in a file called script.sh"

bash_chain = LLMBashChain.from_llm(llm, verbose=False)

answer = bash_chain.run(text)

print(answer)
