
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

my_model_id = 'WizardLM/WizardCoder-15B-V1.0'
llm = HuggingFacePipeline.from_model_id(model_id = my_model_id, task = "text-generation", model_kwargs={ 'max_new_tokens': 10000, 'temperature': 0.2, 'do_sample': True, 'top_k': 15, 'top_p': 0.95})

template = """ 
Write a Python code to: {goal}
Code: Please write Python code to achieve the goal. Comment out contents which are not Python code 
"""

prompt = PromptTemplate(template=template, input_variables=["goal"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

goal = "calculate the sum from 1 to 10."

print(llm_chain.run(goal))

