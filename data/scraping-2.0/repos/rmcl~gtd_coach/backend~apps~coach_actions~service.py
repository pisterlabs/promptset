from django.conf import settings
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

from user_conversations.service import get_user_conversation_service
from coach_actions.actions import TOOL_ACTIONS

class CoachActionService:
    """Register all available coach actions."""

    SYSTEM_PROMPT = """You are very powerful assistant, with access to the users 'Get Things Done' (GTD) lists. """

    def __init__(self, settings: dict):
        self.settings = settings

        self._openai_api_key = settings.get('OPENAI_API_KEY', None)
        if not self._openai_api_key:
            raise Exception('OPENAI_API_KEY not found in settings')

    def get_chat_history(self, target_number_id : str) -> ConversationBufferMemory:
        """Build conversation history for a target number."""
        history = ChatMessageHistory()

        conversation_service = get_user_conversation_service()

        messages = conversation_service.get_messages_for_target_number(
            target_number_id=target_number_id,
            limit=15)

        for message in messages:
            if message['direction'] == 'inbound':
                history.add_user_message(message['content'])
            else:
                history.add_ai_message(message['content'])

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=history
        )

        return memory

    def get_agent(self, target_number_id : str):
        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=self._openai_api_key)

        chat_memory = self.get_chat_history(target_number_id)

        system_message = SystemMessage(
            content=self.SYSTEM_PROMPT)
        prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=system_message)

        agent = OpenAIFunctionsAgent(llm=llm, tools=TOOL_ACTIONS, prompt=prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=TOOL_ACTIONS,
            verbose=True,
            memory=chat_memory)
        return agent_executor

    def invoke(self, target_number_id : str, new_message : str) -> str:
        agent = self.get_agent(target_number_id)
        result = agent.run(new_message)
        return result

def get_gtd_coach_service():
    """Get the coach service."""
    return CoachActionService(settings=settings.GTD_COACH)
