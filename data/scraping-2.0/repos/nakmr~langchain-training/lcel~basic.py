from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.llms import OpenAI

prompt = ChatPromptTemplate.from_template("telll me a short joke about {topic}")
model = ChatOpenAI()
output_parser = StrOutputParser()

# chain = prompt | model | output_parser

# print(chain.invoke({"topic": "accenture"}))

prompt_value = prompt.invoke({"topic": "accenture"})
# print(prompt_value)

message = model.invoke(prompt_value)
print(message)

# llm = OpenAI(model="gpt-3.5-turbo-instruct")
# result = llm.invoke(prompt_value)
# print(result)

output = output_parser.invoke(message)
print(output)