from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback

import os

os.environ["OPENAI_API_KEY"] = "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"


class SuggestAgent:

    def __init__(self):
        self.chat = ChatOpenAI(temperature=1.0, n=3)
        self.template = (
            "You are a helpful assistant that rephrases the following sentence to make it sound more fluent and easy to understand for students, make it more formal sounding"
        )
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(self.template)
        self.human_template = "{sentence}"
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(self.human_template)
        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )

    def get_suggestions(self, text):
        generations = self.chat.generate([self.chat_prompt.format_prompt(sentence=text).to_messages()]).generations
        text_generations = []
        for generation in generations[0]:
            text_generations.append(generation.text)
        return text_generations

if __name__ == "__main__":
    sa = SuggestAgent()
    with get_openai_callback() as cb:
        suggestions = sa.get_suggestions("That thingy called photosynthesis what it do ?")
        print(suggestions)
        print(cb)