import logging
from datetime import datetime
from typing import List

from google.cloud.firestore_v1.types import firestore
from langchain.schema import BaseMessage, _message_to_dict, HumanMessage, AIMessage

from config import Config
from models.bookmark import VectorStoreBookmark, VectorStoreBookmarkMetadata
from models.chat import ConversationMessage
from utils.db import async_firebase_app

log = logging.getLogger(__name__)


class ChatHistoryService:
    def __init__(self, x_uid: str):
        self.db = async_firebase_app
        self.config = Config()
        self.x_uid = x_uid

    def __get_user_document(self):
        if self.config.environment == 'production':
            return self.db.collection('users').document(self.x_uid)
        else:
            return self.db.collection('test_users').document(self.x_uid)

    def get_conversation_document(self, conversation_id: str):
        user_doc_ref = self.__get_user_document()
        return user_doc_ref.collection('conversations').document(conversation_id)

    async def get_chat_history(self, conversation_id: str) -> List[ConversationMessage]:
        conversation_doc_ref = self.get_conversation_document(conversation_id)
        sorted_collection = conversation_doc_ref.collection('messages').order_by('timestamp')
        history = [ConversationMessage.parse_obj(mes.to_dict()) async for mes in sorted_collection.stream()]
        if not history:
            try:
                doc = await conversation_doc_ref.get()
                doc = doc.to_dict()
                return [
                    ConversationMessage(
                        message=_message_to_dict(HumanMessage(content=doc.get('question'))),
                        used_context=[],
                        timestamp=doc.get('timestamp'),
                    ),
                    ConversationMessage(
                        message=_message_to_dict(AIMessage(content=doc.get('answer'))),
                        used_context=[VectorStoreBookmarkMetadata(url=url, title='', id='') for url in doc.get('context_urls', [])],
                        timestamp=doc.get('timestamp'),
                    )
                ]
            except Exception as e:
                log.warning(f'Could not get conversation {conversation_id}: {e}')
                return []
        return history

    async def add_chat_message(self, conversation_id: str, message: BaseMessage, used_context: List[VectorStoreBookmark] = None):
        conversation_doc_ref = self.get_conversation_document(conversation_id)
        conversation_doc = await conversation_doc_ref.get()
        if not conversation_doc.exists:
            raise Exception(f'Conversation {conversation_id} does not exist')
        if not conversation_doc.get('title'):
            await conversation_doc_ref.update({
                'title': message.content[:250] + '...' if len(message.content) > 250 else message.content
            })
        await conversation_doc_ref.collection('messages').add(ConversationMessage(
            message=_message_to_dict(message),
            timestamp=int(datetime.now().timestamp()),
            used_context=[bookmark.metadata.dict() for bookmark in used_context] if used_context else None
        ).dict())

    async def get_conversations(self):
        user_doc_ref = self.__get_user_document()
        conversations_ref = user_doc_ref.collection('conversations')
        conversations = [
            {
                'id': conversation.id,
                'title': conversation.to_dict().get('title') or conversation.to_dict().get('question'), # backwards compatibility
            } async for conversation in conversations_ref.order_by('timestamp', direction="DESCENDING").stream()
        ]

        return conversations
