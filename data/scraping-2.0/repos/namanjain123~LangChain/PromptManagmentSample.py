from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
llm = OpenAI(temperature=0.9)
def setupprompttemp(inputvariable,template):
    prompt=PromptTemplate(
        input_variables=inputvariable,
        template=template
    )
    return prompt
#Set the prompt
prompts=setupprompttemp(["data"],"You have new {data} in this dataset what to do next to make a sql squery")
print(prompts.format(data="name"))
chain = LLMChain(llm=llm, prompt=prompts)
try:
    print(chain.run("you have data"))
except:
    print("error")
