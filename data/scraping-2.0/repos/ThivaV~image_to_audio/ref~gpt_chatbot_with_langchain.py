from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a story teller. "
                "You can generate a story based on a simple narrative, "
                "the story be no more than 20 words."
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}")
    ]
)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
resp = llm(chat_template.format_messages(text="white wool shirt"))
print(resp.content)