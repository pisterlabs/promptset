#Bring in dependencies
from langchain.llms import VertexAI
from langchain.chains import APIChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.api.prompt import API_RESPONSE_PROMPT
from langchain.chains import LLMChain

llm = VertexAI(model_name="text-bison@001")

#oas = "https://api-dev.apigee-west.com/retail/v1/oas"
oas = input("Enter the URI to your OAS: ")
#apikey = input("Enter your API Key: ")
prompt = input("Enter your prompt: ")

chain = APIChain.from_llm_and_api_docs(
    llm, 
    oas,
    #headers=apikey,
    verbose=True,
    limit_to_domains=None
    #limit_to_domains=["https://api-dev.apigee-west.com"],
)

response = chain.run(prompt)
print(response)
