from langchain.chains import LLMChain, SequentialChain
from langchain import PromptTemplate
from prompt_templates import reply_system_template, reply_humam_template

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

class Replier():
    '''
    The information updater to update thoughts, status, and relationship.
    It is a wrapper function that sequentially calls the specific updater.
    '''
    def __init__(self, llm, agent_name, sender) -> None:
        self.llm = llm
        self.agent_name = agent_name
        self.sender = sender

    async def reply(self, report, message, context) -> str:
        system_message_prompt = SystemMessagePromptTemplate.from_template(reply_system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(reply_humam_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        reply_chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        reply = await reply_chain.arun(agent_name=self.agent_name,
                                 sender=self.sender,
                                 message=message,
                                 report=report,
                                 context=context)
        return reply
