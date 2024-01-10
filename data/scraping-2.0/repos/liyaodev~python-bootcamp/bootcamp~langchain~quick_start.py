# -*- coding: utf-8 -*-

from langchain.llms import OpenAI

import os
# os.environ['OPENAI_API_KEY'] = ''

llm = OpenAI(temperature=0.9)

text = 'Python是不是最好的语言？'
print(llm(text))

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=['language'],
    template='{language}是不是最好的语言？'
)

print(prompt.format(language='Python'))

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=['language'],
    template='{language}是不是最好的语言？'
)
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

chain.run('Python')


# os.environ['SERPAPI_API_KEY'] = ''

from langchain.agents import (
    load_tools,
    initialize_agent,
)
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
tools = load_tools(['serpapi', 'llm-math'], llm=llm)
agent = initialize_agent(tools, llm, agent='zero-shot-react-description', verbose=True)
agent.run('Python 是不是最好的语言？')


from langchain import OpenAI, ConversationChain

llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)
conversation.predict(input='Python 是不是最好的语言？')
conversation.predict(input='Python 语言怎么样？')
