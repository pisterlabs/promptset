import os
import openai
from langchain.llms import AzureOpenAI
from dotenv import load_dotenv
from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain


load_dotenv()

openai.api_type = os.environ.get("OPENAI_API_TYPE")
openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.api_base = os.environ.get("OPENAI_API_BASE")
openai.api_version = os.environ.get("OPENAI_API_VERSION")

llm = AzureOpenAI(deployment_name="text-davinci-003", model_name="text-davinci-003")
# take inputs from user and return the answer until the user says "bye" or "quit" or "exit"
print("Chatbot: Hi, I am a chatbot. I can suggest the name of a company and slogan based the product.\n") 

prompt_product = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
print("Tell me the name of the product? (type 'bye' to exit)")
user_input = input("You: ")
if user_input.lower() in ["bye", "quit", "exit"]:
    print("Chatbot: Bye")
# Product chain
chain_product = LLMChain(llm=llm, prompt=prompt_product)
companyName = chain_product.run(user_input)
print(companyName)

#Slogan
prompt_catchphrase = PromptTemplate(
    input_variables=["company_name"],
    template="Write a catchphrase for the following company: {company_name}",
)
chain_catchphrase = LLMChain(llm=llm, prompt=prompt_catchphrase)

overall_chain = SimpleSequentialChain(chains=[chain_product,chain_catchphrase], verbose=False)

output = overall_chain.run(user_input)
print(output)