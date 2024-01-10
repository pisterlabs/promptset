import logging
from langchain.agents import AgentExecutor
from langchain.agents.openai_assistant import OpenAIAssistantRunnable

from tools.email_assistant.send import Email_SendEmail
from tools.form_assistant.order import (
    WorkOrderForm,
    PurchaseOrderForm,
)

#from app.langchain.assistants.tools.sevdesk_assistant.sevDesk_mock import sevDesk_Rechnungen

logger = logging.getLogger(__name__)

class FormAssistant:
    def __init__(self):
        self.tools = [
            Email_SendEmail(),
            WorkOrderForm(),
            PurchaseOrderForm(),
        ]
        self.assistant = OpenAIAssistantRunnable.create_assistant(
            name="filling form assistant",
            instructions="You are the construction companies Bot. Allow client to fill out & send forms much more quickly and easily",
            tools=self.tools,
            model="gpt-3.5-turbo-1106",
            as_agent=True,
        )

    def ask__question(self, question):
        logger.info(question)
        agent_executor = AgentExecutor(agent=self.assistant, tools=self.tools)
        return agent_executor.invoke({"content": question})
