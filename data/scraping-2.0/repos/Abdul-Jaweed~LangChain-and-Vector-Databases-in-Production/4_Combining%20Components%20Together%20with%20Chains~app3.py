## Sequential Chain

# Another helpful feature is using a sequential chain that concatenates multiple chains into one. The following code shows a sample usage.


from langchain.chains import SimpleSequentialChain
from langchain import PromptTemplate, OpenAI, LLMChain

import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

llm = OpenAI(
    openai_api_key=apikey,
    model="text-davinci-003",
    temperature=0
)

overall_chain = SimpleSequentialChain(
    chains=[
        chain_one,
        chain_two
    ]
)


# The SimpleSequentialChain will start running each chain from the first index and pass its response to the next one in the list.

