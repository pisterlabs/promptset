from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.schema import HumanMessage

output_parser = CommaSeparatedListOutputParser()

chat = ChatOpenAI(
    model="gpt-3.5-turbo",
)

print(output_parser.get_format_instructions()),
result = chat(
    [
        HumanMessage(content="Apple開発した製品を3つ列挙してください."),
        HumanMessage(content=output_parser.get_format_instructions()),
    ],
)

output = output_parser.parse(result.content)

for item in output:
    print(item)
