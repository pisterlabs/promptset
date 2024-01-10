from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(temperature = 0.9)

template = 'How to make pasta using {ingredients}?'

prompt = PromptTemplate{ 
    input_variables = [ 'ingredients' ],
    template = template
}

from langchain.chains import LLMChain

chain = LLMChain(llm = llm, prompt =prompt)

print(chain.run ( 'White Sauce' ))