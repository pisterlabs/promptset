from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
# from langchain.chains import SequentialChain

import os
os.environ["OPENAI_API_KEY"] = "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"


llm = OpenAI(temperature=0.7)

# Chain 1: Tell me about celebrity
first_input_prompt = PromptTemplate(
    input_variables = ['name'],
    template = "Tell me about celebrity {name}"
)
chain1 = LLMChain(
    llm=llm,
    prompt=first_input_prompt,
    output_key='person'
)

# Chain 2: celebrity DOB
second_input_prompt = PromptTemplate(
    input_variables = ['person'],
    template = "when was {person} born"
)
chain2 = LLMChain(
    llm=llm,
    prompt=second_input_prompt,
    output_key='dob'
)

# Chain 3: 5 major events on that day
third_input_prompt = PromptTemplate(
    input_variables = ['dob'],
    template = "Mention 5 major events happened around {dob} in the world"
)
chain3 = LLMChain(
    llm=llm,
    prompt=third_input_prompt,
    output_key='description'
)

#combining chains
from langchain.chains import SequentialChain
celebrity_chain = SequentialChain(
    chains=[chain1,chain2,chain3],
    input_variables=['name'],
    output_variables=['person','dob','description']
)
data = celebrity_chain({'name':"MS Dhoni"})
print("Name:", data['name'])
print("Date of Birth:", data['dob'])
print("Description:")
print(data['person'])
print("Historical Events:")
print(data['description'])