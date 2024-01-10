from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate
import openai

class RewriteTemplate:

    @staticmethod
    def test_rewrite(userInput:str):
        chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are a helpful assistant that re-writes the user's text to "
                        "sound more upbeat. Please rewrite the text in the language the user used."
                    )
                ),
                HumanMessagePromptTemplate.from_template("{text}"),
            ]
        )

        llm = ChatOpenAI()
        llm(chat_template.format_messages(text=userInput))
    
        return llm.response