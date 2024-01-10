import logging
from os import unlink
from pathlib import Path
from string import Formatter
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Iterable, List, Optional, Protocol

from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from team_red.llm import build_llm
from team_red.models.qa import QAConfig
from team_red.transport import (
    DocumentSource,
    FileTypes,
    PromptConfig,
    QAAnswer,
    QAFileUpload,
    QAQuestion,
)
from team_red.utils import build_retrieval_qa

if TYPE_CHECKING:
    from langchain.document_loaders.base import BaseLoader


_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class VectorStore(Protocol):
    def merge_from(self, vector_store: "VectorStore") -> None:
        pass

    def save_local(self, path: str) -> None:
        pass

    def add_texts(
        self,
        texts: Iterable[str],
    ) -> List[str]:
        pass

    def search(self, query: str, search_type: str, k: int) -> List[Document]:
        pass


class QAService:
    def __init__(self, config: QAConfig) -> None:
        self._config = config
        self._llm = build_llm(config.model)
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self._config.embedding.model.name,
            model_kwargs={"device": self._config.device},
        )
        self._vectorstore: Optional[VectorStore] = None
        self._database: Optional[BaseRetrievalQA] = None
        self._fact_checker_db: Optional[BaseRetrievalQA] = None
        if (
            config.embedding.db_path
            and Path(config.embedding.db_path, "index.faiss").exists()
        ):
            _LOGGER.info(
                "Load existing vector store from '%s'.", config.embedding.db_path
            )
            self._vectorstore = FAISS.load_local(
                config.embedding.db_path, self._embeddings
            )

    def db_query(self, question: QAQuestion) -> List[DocumentSource]:
        if not self._vectorstore:
            return []
        return [
            DocumentSource(
                content=doc.page_content,
                name=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page", 1),
            )
            for doc in self._vectorstore.search(
                question.question,
                search_type=question.search_strategy,
                k=question.max_sources,
            )
        ]

    def query(self, question: QAQuestion) -> QAAnswer:
        if not self._database:
            if not self._vectorstore:
                msg = "No vector store initialized! Upload documents first."
                _LOGGER.error(msg)
                return QAAnswer(status=404, error_msg=msg)
            self._database = self._setup_dbqa(self._config.model.prompt)

        response = self._database({"query": question.question})
        answer = QAAnswer(answer=response["result"])
        if self._config.features.return_source:
            for doc in response.get("source_documents", []):
                answer.sources.append(
                    DocumentSource(
                        content=doc.page_content,
                        name=doc.metadata.get("source", "unknown"),
                        page=doc.metadata.get("page", 1),
                    )
                )
        if self._config.features.fact_checking.enabled is True:
            if not self._fact_checker_db:
                self._fact_checker_db = self._setup_dbqa_fact_checking(
                    self._config.features.fact_checking.model.prompt
                )
            response = self._fact_checker_db({"query": response["result"]})
            for doc in response.get("source_documents", []):
                answer.sources.append(
                    DocumentSource(
                        content=doc.page_content,
                        name=doc.metadata.get("source", "unknown"),
                        page=doc.metadata.get("page", 1),
                    )
                )
        _LOGGER.debug("\n==== Answer ====\n\n%s\n===============", answer)
        return answer

    def set_prompt(self, config: PromptConfig) -> PromptConfig:
        self._config.model.prompt = config
        self._database = self._setup_dbqa(self._config.model.prompt)
        return self._config.model.prompt

    def get_prompt(self) -> PromptConfig:
        return self._config.model.prompt

    def add_file(self, file: QAFileUpload) -> QAAnswer:
        documents: Optional[List[Document]] = None
        file_path = Path(file.name)
        try:
            f = NamedTemporaryFile(dir=".", suffix=file_path.suffix, delete=False)
            f.write(file.data)
            f.flush()
            f.close()
            file_type = FileTypes(file_path.suffix[1:])
            loader: BaseLoader
            if file_type == FileTypes.TEXT:
                loader = TextLoader(f.name)
            elif file_type == FileTypes.PDF:
                loader = PyPDFLoader(f.name)
            documents = loader.load()
            # source must be overriden to not leak upload information
            # about the temp file which are rather useless anyway
            for doc in documents:
                doc.metadata["source"] = file_path.name
        except BaseException as err:
            _LOGGER.error(err)
        finally:
            if f:
                unlink(f.name)
        if not documents:
            _LOGGER.warning("No document was loaded!")
            return QAAnswer(error_msg="No document was loaded!", status=500)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._config.embedding.chunk_size,
            chunk_overlap=self._config.embedding.chunk_overlap,
        )
        texts = text_splitter.split_documents(documents)
        if self._vectorstore is None:
            _LOGGER.info("Create new vector store from document.")
            self._vectorstore = FAISS.from_documents(texts, self._embeddings)
        else:
            _LOGGER.info("Adding document to existing vector store.")
            tmp = FAISS.from_documents(texts, self._embeddings)
            self._vectorstore.merge_from(tmp)
        if self._config.embedding.db_path:
            self._vectorstore.save_local(self._config.embedding.db_path)
        return QAAnswer()

    def _setup_dbqa(self, prompt: PromptConfig) -> BaseRetrievalQA:
        if "context" not in prompt.parameters:
            _LOGGER.warning(
                "Prompt does not include '{context}' variable."
                "It will be appened to the prompt."
            )
            prompt.text += "\n\n{context}"
        _LOGGER.info(
            "\n===== Setup dbqa with prompt ====\n\n%s\n\n====================",
            prompt.text,
        )
        qa_prompt = PromptTemplate(
            template=prompt.text,
            input_variables=prompt.parameters,
        )
        dbqa = build_retrieval_qa(
            self._llm,
            qa_prompt,
            self._vectorstore,
            self._config.embedding.vector_count,
            self._config.features.return_source,
        )

        return dbqa

    def _setup_dbqa_fact_checking(self, prompt: PromptConfig) -> BaseRetrievalQA:
        _LOGGER.info("Setup fact checking...")
        if "context" not in prompt.parameters:
            _LOGGER.warning(
                "Prompt does not include '{context}' variable."
                "It will be appened to the prompt."
            )
            prompt.text += "\n\n{context}"
        fact_checking_prompt = PromptTemplate(
            template=prompt.text,
            input_variables=prompt.parameters,
        )
        dbqa = build_retrieval_qa(
            self._llm,
            fact_checking_prompt,
            self._vectorstore,
            self._config.embedding.vector_count,
            self._config.features.return_source,
        )

        return dbqa


# # Process source documents
# source_docs = response["source_documents"]
# for i, doc in enumerate(source_docs):
#     _LOGGER.debug(f"\nSource Document {i+1}\n")
#     _LOGGER.debug(f"Source Text: {doc.page_content}")
#     _LOGGER.debug(f'Document Name: {doc.metadata["source"]}')
#     _LOGGER.debug(f'Page Number: {doc.metadata.get("page", 1)}\n')
#     _LOGGER.debug("=" * 60)

# _LOGGER.debug(f"Time to retrieve response: {endQA - startQA}")

# if CONFIG.features.fact_checking:
#     startFactCheck = timeit.default_timer()
#     dbqafact = setup_dbqa_fact_checking()
#     response_fact = dbqafact({"query": response["result"]})
#     endFactCheck = timeit.default_timer()
#     _LOGGER.debug("Factcheck:")
#     _LOGGER.debug(f'\nAnswer: {response_fact["result"]}')
#     _LOGGER.debug("=" * 50)

#     # Process source documents
#     source_docs = response_fact["source_documents"]
#     for i, doc in enumerate(source_docs):
#         _LOGGER.debug(f"\nSource Document {i+1}\n")
#         _LOGGER.debug(f"Source Text: {doc.page_content}")
#         _LOGGER.debug(f'Document Name: {doc.metadata["source"]}')
#         _LOGGER.debug(f'Page Number: {doc.metadata.get("page", 1)}\n')
#         _LOGGER.debug("=" * 60)

#     _LOGGER.debug(f"Time to retrieve fact check: {endFactCheck - startFactCheck}")
