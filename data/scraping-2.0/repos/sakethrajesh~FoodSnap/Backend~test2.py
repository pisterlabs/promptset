import requests

# make a post request to test endpoint that uploads a file
url = "localhost/api/generateRecipe" 
r = requests.post(url, files={})
print(r.json())

# from langchain.llms import OpenAI
# from langchain import PromptTemplate, LLMChain
# import os


# template = """Question: {question} 
#         Answer: Let's think step by step."""

# prompt = PromptTemplate(template=template, input_variables=["question"])

# llm = OpenAI(openai_api_key='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')

# llm_chain = LLMChain(prompt=prompt, llm=llm)

# detections = ''

# question = f"remove the non foods from this ${detections}"

# thing = llm_chain.run(question)

# print(thing)