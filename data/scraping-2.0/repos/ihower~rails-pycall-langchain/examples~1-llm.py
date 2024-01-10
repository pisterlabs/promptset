# https://python.langchain.com/en/latest/getting_started/getting_started.html


from langchain.llms import OpenAI

llm = OpenAI(temperature=0.9,
             model_name="text-davinci-003",
             openai_api_key="sk-your-key-here")

text = "What would be a good company name for a company that makes colorful socks?"

print( llm(text) )