import datetime
import json
import typing
import uuid
from uuid import UUID

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tracers import ConsoleCallbackHandler
from starlette.responses import StreamingResponse

from app.models import ChatSession
from app.repositories import TenantRepository, ChatbotRepository
from app.repositories.aimodels import SettingsRepository
from app.repositories.document import DocumentRepository
from app.repositories.session import SessionRepository
from app.schemas.requests.chat import ChatCompletionRequest
from app.schemas.responses.documents import DocumentResponse
from app.schemas.responses.session import SessionResponse
from core.chains import DefaultChainFactory
from core.exceptions import NotFoundException


def _get_default_config():
    return {
        'callbacks': [ConsoleCallbackHandler()],
    }


class SessionController:
    def __init__(self,
                 session_repository: SessionRepository,
                 document_repository: DocumentRepository,
                 tenant_repository: TenantRepository,
                 chatbot_repository: ChatbotRepository,
                 settings_repository: SettingsRepository):
        self.session_repository = session_repository
        self.document_repository = document_repository
        self.tenant_repository = tenant_repository
        self.chatbot_repository = chatbot_repository
        self.settings_repository = settings_repository

    def stream_chatbot_response(self,
                                session_id: UUID,
                                chat_completion_request: ChatCompletionRequest) -> StreamingResponse:
        session = self.session_repository.get_by_id(session_id=session_id)
        if session is None:
            raise NotFoundException("Session not found")

        return StreamingResponse(
            self._stream_chatbot_response(session, chat_completion_request),
            media_type="text/event-stream")

    async def _stream_chatbot_response(self,
                                       session: ChatSession,
                                       chat_completion_request: ChatCompletionRequest) -> typing.AsyncIterable[str]:
        # creating chain
        chain_factory = DefaultChainFactory(document_repository=self.document_repository,
                                            session=session)
        chain = chain_factory.create_chain()

        # chain created - updating session message history
        # update history and update ttl
        session.message_history.append(HumanMessage(content=chat_completion_request.human_input))
        self.session_repository.add(session=session)

        ai_response = ""
        async for chunk in chain.astream(
                {"human_input": chat_completion_request.human_input},
                config=_get_default_config()
        ):
            ai_response += chunk
            yield chunk

        # chain finished - updating session message history
        # update history and update ttl
        session.message_history.append(AIMessage(content=ai_response))
        self.session_repository.add(session=session)

        # saving sources used
        self.session_repository.set_session_sources(session_id=session.session_id,
                                                    sources=chain_factory.sources)

    def get_session_last_source(self, user_id: int, session_id: UUID):
        session = self.session_repository.get_by_id(session_id=session_id)
        if session is None:
            raise NotFoundException("Session not found")

        if session.user_id != user_id:
            raise NotFoundException("Session not found")

        sources = self.session_repository.get_session_sources(session_id=session.session_id)
        if sources is None:
            return []

        return [DocumentResponse(**source) for source in sources]

    async def create_session(self,
                             tenant_id: str,
                             chatbot_id: UUID) -> SessionResponse:
        # retrieving tenant
        tenant = await self.tenant_repository.get_by_id(tenant_id=tenant_id)
        if tenant is None:
            raise NotFoundException("Tenant not found")

        # retrieving chatbot
        chatbot = await self.chatbot_repository.get_by_id(
            tenant_id=tenant_id,
            chatbot_id=chatbot_id
        )
        if chatbot is None:
            raise NotFoundException("Chatbot not found")

        if chatbot.tenant_id != tenant.id:
            raise NotFoundException("Chatbot not found in this tenant")

        # retrieving ai model settings
        embedding_model = self.settings_repository.get_embedding_model()
        if embedding_model is None:
            raise NotFoundException("AI Model settings not found")

        # creating session
        session = self.session_repository.add(
            ChatSession(
                session_id=uuid.uuid4(),
                tenant_id=tenant_id,
                chatbot_id=chatbot_id,
                embedding_model=embedding_model,
                chat_model=chatbot.chatbot_model(),
                chatbot_filter=None if chatbot.chatbot_filter is None else json.load(chatbot.chatbot_filter),
                message_history=[SystemMessage(content=chatbot.system_prompt)],
                created_at=datetime.datetime.now()
            )
        )

        return SessionResponse(session_id=session.session_id,
                               created_at=session.get_created_at_string())

    def delete_session(self, session_id: UUID):
        session = self.session_repository.get_by_id(session_id=session_id)
        if session is not None:
            self.session_repository.delete(session_id=session_id)
        else:
            raise NotFoundException("Session not found")
