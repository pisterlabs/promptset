from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

OPENAI_API_KEY = "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"

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
model = ChatOpenAI(api_key=OPENAI_API_KEY)
output_parser = StrOutputParser()

chain = prompt | model | output_parser

rslt = chain.invoke({"topic": "ice cream"})

prompt_value = prompt.invoke({"topic": "ice cream"})

llm = OpenAI(model="gpt-3.5-turbo-instruct", api_key=OPENAI_API_KEY)
modle_value = llm.invoke(prompt_value)

output_value = output_parser.invoke(modle_value)

print(output_value)