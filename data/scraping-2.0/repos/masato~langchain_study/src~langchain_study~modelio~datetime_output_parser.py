from langchain.output_parsers import DatetimeOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI

output_parser = DatetimeOutputParser()

chat = ChatOpenAI(model="gpt-3.5-turbo")

prompt = PromptTemplate.from_template("{product}のリリース日を教えて")

result = chat.invoke(
    [
        HumanMessage(content=prompt.format(product="iPhone8")),
        HumanMessage(content=output_parser.get_format_instructions()),
    ],
)

output = output_parser.invoke(result)
print(output)
