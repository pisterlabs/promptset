"""
Copyright (c) VKU.NewEnergy.

This source code is licensed under the Apache-2.0 license found in the
LICENSE file in the root directory of this source tree.
"""
from enum import Enum
import os
from typing import Dict, Type, Union
from langchain.llms import BaseLLM, type_to_cls_dict
from langchain.chat_models import ChatOpenAI

class BaseConstants:
    ROOT_PATH = os.path.abspath(os.path.join(__file__, "../.."))

class IngestDataConstants(BaseConstants):
    CHUNK_SIZE = 8000
    CHUNK_OVERLAP = 100
    VECTORSTORE_FOLDER = 'vectorstores/'
    MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
    ALLOWED_EXTENSIONS = ["pdf", "json"]
    TEMP_UPLOADED_FOLDER = 'tmp/uploaded/'

class LangChainOpenAIConstants(BaseConstants):
    type_to_cls_dict_plus: Dict[str, Type[Union[BaseLLM, ChatOpenAI]]] = {k: v for k, v in type_to_cls_dict.items()}
    type_to_cls_dict_plus.update({"chat_openai": ChatOpenAI})
    AGENT_SYSTEM_PROMPT_CONTENT = (
        "Do your best to answer the questions. "
        "Feel free to use any tools available to look up relevant information, only if necessary."
    )
    KNOWLEDGE_BASE_RETRIEVER_NAME: str = "knowledge_base_retriever"
    KNOWLEDGE_BASE_RETRIEVER_DESCRIPTION: str = (
        "Searches and returns documents in the knowledge base"
    )

class ErrorChatMessageConstants(str, Enum):
    VI = "Xin lỗi, tôi chưa thể trả lời câu hỏi này ngay bây giờ, vui lòng thử lại sau ít phút hoặc báo cáo sự cố cho nhóm phát triển, xin cảm ơn!"
    EN = "Sorry, I can't answer this question right now, please try again in a few minutes or report the issue to the development team, thank you!"