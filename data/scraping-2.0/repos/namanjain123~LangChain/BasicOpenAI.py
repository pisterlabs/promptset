from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
def callprompt_print(prompt,llm):
    print(llm(prompt))
    return "Result Printed"
#Model Called
llm = OpenAI(temprature=0)
#did to make more consistent and less creative for temprature#did to make more consistent and less creative for temprature
prompt=input("Enter your prompt: \n")
print(callprompt_print(prompt,llm))

