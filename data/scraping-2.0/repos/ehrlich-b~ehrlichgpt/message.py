from langchain.chains import OpenAIModerationChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (AIMessagePromptTemplate,
                                    HumanMessagePromptTemplate)

from utils import escape_prompt_content


class Message:
    CENSORED = '[Text removed due to content policy violation]'
    def __init__(self, sender, content, timestamp, gpt_version_requested=3, at_mentioned=False):
        self.sender = sender
        self.content = content
        self.token_count = 0
        self.timestamp = timestamp
        self.gpt_version_requested = gpt_version_requested
        self.at_mentioned = at_mentioned

    def get_prompt_template(self):
        if self.sender == "ai":
            message_prompt = AIMessagePromptTemplate(
                prompt=PromptTemplate(
                    template=escape_prompt_content(self.content),
                    input_variables=[],
                )
            )
        else:  # sender == "ai"
            message_prompt = HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    template=escape_prompt_content(self.sender + ": " + self.content),
                    input_variables=[],
                )
            )

        return message_prompt

    def get_number_of_tokens(self):
        if self.token_count == 0 and self.content != "":
            llm = ChatOpenAI()
            self.token_count = llm.get_num_tokens(self.content)
        return self.token_count

    @staticmethod
    def violates_content_policy(text):
        moderation_chain = OpenAIModerationChain(error=True)
        try:
            moderation_chain.run(text)
            return False
        except:
            return True

