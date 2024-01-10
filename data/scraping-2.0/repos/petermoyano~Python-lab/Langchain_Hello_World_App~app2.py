from langchain.llms import OpenAI

llm = OpenAI(model_name="text-davinci-003")
response = llm("Explain machine learning in one paragraph")
print(response)