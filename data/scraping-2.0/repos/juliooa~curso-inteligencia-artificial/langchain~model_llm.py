from langchain.llms.openai import OpenAI

llm = OpenAI(openai_api_key='TU_API_KEY')
result = llm.predict("La mejor forma de empezar el día es ")
print(result)