from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import os
import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

llm = OpenAI(temperature = 0.9)

prompt = PromptTemplate(
    input_variables=["place"],
    template="Where is {place} city?",
)

chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run("Ha Noi"))