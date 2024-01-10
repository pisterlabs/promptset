
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

llm = OpenAI(openai_api_key="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")

llm.predict("What would be a good company name for a company that makes colorful socks?")


chat = ChatOpenAI(openai_api_key="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",temperature=0)
chat.predict_messages([HumanMessage(content="Translate this sentence from English to French. I love programming.")])

human_message = "Translate from English to Frech: I love playing Tennis"
result = chat([HumanMessage(content = human_message)])
print(result.content)

prompt = "Suggest me a good name for an ice cream parlour that is located on a beach!"
print(llm(prompt))

print("done")