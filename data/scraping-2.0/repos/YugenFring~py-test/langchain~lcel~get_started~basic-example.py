import os

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

os.environ['OPENAI_API_KEY'] = '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'

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
model = ChatOpenAI()
llm = OpenAI(model="gpt-3.5-turbo-instruct")
output_parser = StrOutputParser()

chat_chain = prompt | model | output_parser
llm_chain = prompt | llm | output_parser

if __name__ == "__main__":
    #print(chat_chain.invoke({"topic": "ice cream"}))

    # Prompt
    print("1===========================================")
    prompt_value = prompt.invoke({"topic": "ice cream"})
    print(prompt_value)
    print(prompt_value.to_messages())
    print(prompt_value.to_string())

    # Model
    print("2===========================================")
    message = model.invoke(prompt_value)
    print(message)

    string = llm.invoke(prompt_value)
    print(string)

    # Output parser
    print("3===========================================")
    message_parser = output_parser.invoke(message)
    print(message_parser)

    string_parser = output_parser.invoke(string)
    print(string_parser)