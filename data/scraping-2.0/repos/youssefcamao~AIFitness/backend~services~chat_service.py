from langchain.chat_models import ChatOpenAI
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.prompts import ChatPromptTemplate
from ..models.chat_session import ChatSession, User
from ..config import settings
from ..llm import prompts
from ..llm.title_creator import TitleLlm
from ..models.chat_session import ChatSession
from beanie import PydanticObjectId
from typing import List, Dict
import asyncio
import uuid


class ChatService:
    def __init__(self, user: User):
        self.chat_model = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY, model_name=settings.MODEL_NAME)
        set_llm_cache(InMemoryCache())
        self.title_creator = TitleLlm()
        self.user = user

    async def get_session_fromId(self, session_id: str) -> ChatSession | None:
        for session in self.user.session_list:
            if session.session_id == session_id:
                return session
        return None

    async def handle_chat(self, session_id: str, user_message: str) -> str:
        session = await self.get_session_fromId(session_id)
        if session is None:
            raise ValueError("Session not found")

        await self.run_chat(session, user_message)
        await self.user.save()
        return session.get_latest_response()

    async def create_new_session(self, initial_message: str) -> ChatSession:
        session = ChatSession(session_id=str(uuid.uuid4()))
        await asyncio.gather(
            self.run_chat(session, initial_message),
            self.__setup_title(session, initial_message)
        )
        self.user.session_list.append(session)
        await self.user.save()
        return session

    async def get_all_sessions(self) -> List[ChatSession]:
        return self.user.session_list

    async def delete_session(self, session_id: str) -> Dict[str, str]:
        self.user.session_list = [
            s for s in self.user.session_list if s.session_id != session_id]
        await self.user.save()
        return {"message": "Session deleted"}

    async def run_chat(self, chat_session: ChatSession, user_message):
        chat_session.add_user_message(user_message)
        chat_prompt = chat_session.get_chat_prompt_template(prompts.SYS_PROMPT)
        formatted_messages = chat_prompt.format()
        response = await self.chat_model.ainvoke(
            formatted_messages)
        chat_session.add_ai_message(response.content.replace('AI:', ''))

    async def __setup_title(self, session: ChatSession, initial_message: str):
        title = await self.title_creator.generate_summary(initial_message)
        session.session_title = title
