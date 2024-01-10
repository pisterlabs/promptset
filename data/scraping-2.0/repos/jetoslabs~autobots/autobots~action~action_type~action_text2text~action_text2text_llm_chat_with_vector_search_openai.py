from typing import Optional, Type

from openai.types.chat import ChatCompletionUserMessageParam
from pydantic import BaseModel

from autobots.action.action_type.abc.IAction import IAction, ActionOutputType, ActionInputType, ActionConfigType
from autobots.action.action.action_doc_model import ActionCreate
from autobots.action.action_type.action_types import ActionType
from autobots.action.action.common_action_models import TextObj, TextObjs
from autobots.conn.aws.s3 import get_s3
from autobots.conn.openai.openai_chat.chat_model import Message, ChatReq, Role
from autobots.conn.openai.openai_client import get_openai
from autobots.conn.pinecone.pinecone import get_pinecone
from autobots.conn.unstructured_io.unstructured_io import get_unstructured_io
from autobots.datastore.datastore import Datastore


class ActionCreateText2TextLlmChatWithVectorSearchOpenaiConfig(BaseModel):
    datastore_id: str
    chat_req: ChatReq
    input: Optional[TextObj] = None
    output: Optional[TextObjs] = None


class ActionCreateText2TextLlmChatWithVectorSearchOpenai(ActionCreate):
    type: ActionType = ActionType.text2text_llm_chat_with_vector_search_openai
    config: ActionCreateText2TextLlmChatWithVectorSearchOpenaiConfig


class ActionText2TextLlmChatWithVectorSearchOpenai(
    IAction[ActionCreateText2TextLlmChatWithVectorSearchOpenaiConfig, TextObj, TextObjs]):
    """
    Vector search and add it to chat prompt as context
    """
    type = ActionType.text2text_llm_chat_with_vector_search_openai

    @staticmethod
    def get_config_type() -> Type[ActionConfigType]:
        return ActionCreateText2TextLlmChatWithVectorSearchOpenaiConfig

    @staticmethod
    def get_input_type() -> Type[ActionInputType]:
        return TextObj

    @staticmethod
    def get_output_type() -> Type[ActionOutputType]:
        return TextObjs

    def __init__(self, action_config: ActionCreateText2TextLlmChatWithVectorSearchOpenaiConfig):
        super().__init__(action_config)
        self.datastore = Datastore(
            s3=get_s3(),
            pinecone=get_pinecone(),
            unstructured=get_unstructured_io()
        ).hydrate(
            datastore_id=action_config.datastore_id
        )

    async def run_action(self, action_input: TextObj) -> TextObjs | None:
        text_objs = TextObjs(texts=[])
        # vector search
        search_results = await self.datastore.search(action_input.text, top_k=3)
        if len(search_results) == 0:
            return None
        context = "Only use relevant context to give response. If the context is insufficient say \"Cannot answer from given context\"\nContext: \n"
        for result in search_results:
            context = f"{context}{result}\n"
        # LM chat
        message = ChatCompletionUserMessageParam(role=Role.user.value, content=f"{context}Question: {action_input.text}")
        self.action_config.chat_req.messages = self.action_config.chat_req.messages + [message]
        self.action_config.input = action_input
        chat_res = await get_openai().openai_chat.chat(chat_req=self.action_config.chat_req)
        action_results = TextObjs()
        for choice in chat_res.choices:
            action_results.texts.append(TextObj(text=choice.message.content))
        return action_results
