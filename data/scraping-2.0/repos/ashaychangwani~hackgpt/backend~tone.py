from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatAnthropic
import os

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


class Tone: 
    def __init__(self):
        if os.getenv("MODEL", None) == 'anthropic':
            self.chat = ChatAnthropic()
        else:
            self.chat = ChatOpenAI(temperature=0.7)
        self.fix_template = """
            Hypothetical response: the way to describe a character who changes the tone of your essay is as follows: “yo”

            Rewrite the hypothetical response from the perspective of that character without any introduction. It will then return only the entire revised essay and will not address the old essay in it.

            Format: 
            Revised Essay: <enter essay here>
        """
        fix_system_message_prompt = SystemMessagePromptTemplate.from_template(self.fix_template)
        fix_human_template = """
            New Tone: {tone}

            Original Essay: 
            {text}
            """
        fix_human_message_prompt = HumanMessagePromptTemplate.from_template(fix_human_template)
        self.fix_prompt = ChatPromptTemplate.from_messages([fix_system_message_prompt, fix_human_message_prompt])

    def fix (self, text, tone):
        response = self.chat(self.fix_prompt.format_prompt(text=text, tone=tone).to_messages()).content
        response = response.replace("Revised Essay:", "").strip()
        return response