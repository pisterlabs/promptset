from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from services.qdrant.qdrant import QdrantClient
from settings.config import override_settings, get_settings
from settings.override_config import get_override_settings
from settings.config import Settings
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from qdrant_client import QdrantClient as qdc
from services.qunda.enums import OpenAIModels
import logging
from typing import Tuple


class Qunda:
    vector_store = None
    qa_client = None

    def __init__(self, settings: Settings, vector_db_client: qdc, model: str):
        collection_name = settings.qdrant_collection_name
        open_ai_base_url = settings.open_ai_base
        if open_ai_base_url is None:
            embeddings = OpenAIEmbeddings(
                openai_api_key=settings.openai_api_key,
            )
        else:
            embeddings = OpenAIEmbeddings(
                openai_api_key=settings.openai_api_key,
                openai_api_base=open_ai_base_url,
                deployment=settings.open_ai_deployment_name,
                api_version=settings.open_ai_api_version,
                openai_api_type=settings.open_ai_type,
                model=settings.open_ai_model_name,
            )

        self.vector_store = Qdrant(
            client=vector_db_client,
            collection_name=collection_name,
            embeddings=embeddings,
        )
        if model == OpenAIModels.davinci.value:
            llm = OpenAI(openai_api_key=settings.openai_api_key)
        elif model in [OpenAIModels.chatgpt35turbo, OpenAIModels.chatgpt4]:
            llm = ChatOpenAI(openai_api_key=settings.openai_api_key, model_name=model)
        else:
            # Default cheaper option
            llm = OpenAI(openai_api_key=settings.openai_api_key)

        self.qa_client = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True,
        )

    async def ask(self, question: str) -> Tuple[str, list[dict]]:
        response = self.qa_client({"query": question})

        source_metadata = list(doc.metadata for doc in response["source_documents"])

        seen_file_names = set()
        clean_metadata_list = list()
        for metadata in source_metadata:
            if not metadata:
                continue
            if metadata.get("filename") in seen_file_names:
                continue
            seen_file_names.add(metadata.get("filename"))
            clean_metadata_list.append(metadata)

        return response["result"], clean_metadata_list


async def ask(user_email: str, question: str, model: str) -> Tuple[str, list[dict]]:
    qdrant_client = QdrantClient()
    settings = override_settings(get_settings(), get_override_settings(user_email))
    qdrant_client.connect(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
    )
    qunda_client = Qunda(
        settings=settings,
        vector_db_client=qdrant_client.client,
        model=model,
    )
    try:
        answer, source_metadata = await qunda_client.ask(
            question=question,
        )
        return answer, source_metadata
    except Exception as e:
        logging.error(f"[Qunda] {e}")
        return "Sorry, I couldn't find an answer to your question."
