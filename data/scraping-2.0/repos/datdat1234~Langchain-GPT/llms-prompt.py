from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import os
import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

llm = OpenAI(temperature = 0.9)

prompt = PromptTemplate(
    input_variables=["place"],
    template="Where is {place} city?",
)

print(prompt.format(place="Ha Noi"))

print(llm(prompt.format(place="Ha Noi")))