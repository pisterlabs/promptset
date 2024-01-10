from langchain.llms import OpenLLM

llm = OpenLLM(server_url='http://localhost:3002')

llm("What is the difference between a duck and a goose? And why there are so many Goose in Canada?")
