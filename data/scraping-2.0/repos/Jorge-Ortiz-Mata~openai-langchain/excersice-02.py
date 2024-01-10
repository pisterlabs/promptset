from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

def generate_pet_name(operation_type):
  llm = OpenAI(temperature=0.8)

  prompt_template = PromptTemplate(
    input_variables=["operation_type"],
    template="What is the result of {operation_type}?"
  )


  name_chain = LLMChain(llm=llm, prompt=prompt_template)

  response=name_chain({"operation_type": operation_type})

  return response

print(generate_pet_name("5*20"))
