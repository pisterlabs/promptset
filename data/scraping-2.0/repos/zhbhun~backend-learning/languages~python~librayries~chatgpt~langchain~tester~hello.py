from langchain.llms import OpenAI


text = "What would be a good company name for a company that makes colorful socks?"

print(text)

llm = OpenAI(temperature=0.9)

print(llm(text))
