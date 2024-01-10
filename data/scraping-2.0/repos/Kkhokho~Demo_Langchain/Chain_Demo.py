# Import schema for chat messages and ChatOpenAI in order to query chat models GPT-3.5-turbo or GPT-4
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.vectorstores import FAISS
import os

# Set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = "API_KEY"

# Import prompt and define PromptTemplate

prompt_first = PromptTemplate(
    input_variables=["name"],
    template="""You are very skilled at searching for information about NFTs. 
Let's gather information about {name} NFTs on rarible, niftygateway, binance, Opensea.""",
)

prompt_second = PromptTemplate(
    input_variables=["name"],
    template="Retrieve specific values, owner, and creator information of {name} NFTs.",
    )

# Function to run language model with PromptTemplate
llm = OpenAI()

# Import LLMChain and define chain with language model and prompt as arguments.
chain_first = LLMChain(llm=llm, prompt=prompt_first)
chain_second = LLMChain(llm=llm, prompt=prompt_second)
# Run the chain only specifying the input variable.
overall_chain = SimpleSequentialChain(chains=[chain_first, chain_second], verbose=True)
explanation = overall_chain.run("Nexian Gem")
# Print the chain's output
# print(explanation)