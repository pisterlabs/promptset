from langchain.llms import VertexAI

llm = VertexAI(model_name="gemini-pro")
print(llm("What are some of the pros and cons of Python as a programming language?"))
