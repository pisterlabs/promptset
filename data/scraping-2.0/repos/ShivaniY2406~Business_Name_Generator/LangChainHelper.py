from secret_key import openapi_key
import os
os.environ['OPENAI_API_KEY']=openapi_key


def generate_name(type):
   from langchain import OpenAI
   llm = OpenAI(temperature=0.7)
   from langchain.prompts import PromptTemplate
   from langchain.chains import LLMChain
   Business_name_Template = PromptTemplate(  input_variables=['type'],template="Please suggest one good business name for {type}")

   Services_Provided_Template = PromptTemplate(
    input_variables=['business_name'],
    template="Please suggest all services along with price for {business_name}"
    )

   business_name_chain = LLMChain(llm=llm,prompt=Business_name_Template,output_key="business_name")
   services_provided_chain = LLMChain(llm=llm, prompt=Services_Provided_Template,output_key="services")

   from langchain.chains import SequentialChain

   schain = SequentialChain(
    chains=[business_name_chain,services_provided_chain],
    input_variables=['type'],
    output_variables=['business_name','services']
     )

   response=schain({'type':type})
   return response

if __name__=="__main__":
    print(generate_name("Teaching"))
