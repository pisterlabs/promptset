# Commented out IPython magic to ensure Python compatibility.
# %%shell
# pip install --quiet chromadb gradio langchain loguru pymupdf pydantic sentence-transformers

# Commented out IPython magic to ensure Python compatibility.
# %%shell
# CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python
# #!CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python


# Commented out IPython magic to ensure Python compatibility.
# %%shell
# rm -rf cache data
# mkdir data cache
# cd cache; wget https://huggingface.co/TheBloke/OpenOrca-Zephyr-7B-GGUF/resolve/main/openorca-zephyr-7b.Q5_K_M.gguf


from abc import ABC, abstractmethod
from dataclasses import dataclass
import gc
import hashlib
import urllib
from uuid import UUID, uuid1, uuid4
from enum import Enum
from loguru import logger
from pathlib import Path
import shutil
import tqdm
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

# from langchain import PromptTemplate
from langchain import hub
from langchain.callbacks.manager import CallbackManager, CallbackManagerForLLMRun
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.base import Chain
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms.base import LLM
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


from typing import Optional

from llama_cpp import Llama

from loguru import logger

import fitz

import numpy as np

import pandas as pd

from pydantic import BaseModel, DirectoryPath, Extra, Field, validator

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from typing_extensions import Literal

"""## Config"""


class DocumentExtension(str, Enum):
    pdf = "pdf"


class DocumentPathSettings(BaseModel):
    doc_path: Union[DirectoryPath, str]
    exclude_paths: List[Union[DirectoryPath, str]] = Field(default_factory=list)
    scan_extensions: List[DocumentExtension]
    additional_parser_settings: Dict[str, Any] = Field(default_factory=dict)
    passage_prefix: str = ""
    label: str = ""  # Optional label, will be included in the metadata


class EmbeddingModelType(str, Enum):
    huggingface = "huggingface"
    instruct = "instruct"
    sentence_transformer = "sentence_transformer"


class EmbeddingModel(BaseModel):
    type: EmbeddingModelType
    model_name: str
    additional_kwargs: dict = Field(default_factory=dict)


class EmbeddingsConfig(BaseModel):
    embedding_model: EmbeddingModel = EmbeddingModel(
        type=EmbeddingModelType.sentence_transformer, model_name="all-MiniLM-L12-v2"
    )
    embeddings_path: Union[DirectoryPath, str]
    document_settings: List[DocumentPathSettings]
    chunk_sizes: List[int] = [1024]


class LlamaModelConfig(BaseModel):
    model_path: Union[Path, str]
    prompt_template: str
    model_init_params: dict = {}
    model_kwargs: dict = {}


def create_uuid() -> str:
    return str(uuid4())


class SemanticSearchOutput(BaseModel):
    chunk_link: str
    chunk_text: str
    metadata: dict


class ResponseModel(BaseModel):
    id: UUID = Field(default_factory=create_uuid)
    question: str
    response: str
    average_score: float
    semantic_search: List[SemanticSearchOutput] = Field(default_factory=list)
    hyde_response: str = ""


class ReplaceOutputPath(BaseModel):
    substring_search: str
    substring_replace: str


class SemanticSearchConfig(BaseModel):
    search_type: Literal["mmr", "similarity"]
    replace_output_path: List[ReplaceOutputPath] = Field(default_factory=list)
    max_k: int = 15
    max_char_size: int = 2048
    query_prefix: str = ""


class Config(BaseModel):
    cache_folder: Path = Path("/Users/alleon_g/code/punfinder/cache")
    embeddings: EmbeddingsConfig
    semantic_search: SemanticSearchConfig
    llm: LlamaModelConfig


llama_model_config = LlamaModelConfig(
    model_path=Path(
        "/Users/alleon_g/code/punfinder/cache/openorca-zephyr-7b.Q5_K_M.gguf"
    ),
    prompt_template="<|system|>\n{context}</s>\n<|user|>\n{question}</s>\n<|assistant|>",
)

document_settings = DocumentPathSettings(
    doc_path=Path("/Users/alleon_g/code/punfinder/sample_data/"),
    scan_extensions=[DocumentExtension.pdf],
)
embeddings_config = EmbeddingsConfig(
    embeddings_path=Path("/Users/alleon_g/code/punfinder/data/"),
    document_settings=[document_settings],
)
semantic_search_config = SemanticSearchConfig(search_type="mmr")

config = Config(
    embeddings=embeddings_config,
    semantic_search=semantic_search_config,
    llm=llama_model_config,
)


class Document(BaseModel):
    page_content: str
    metadata: dict = Field(default_factory=dict)


class PDFSplitter:
    def __init__(self, chunk_overlap: int = 200) -> None:
        self.chunk_overlap = chunk_overlap

    def split_document(
        self, document_path: Union[str, Path], max_size: int, **kwargs
    ) -> List[dict]:
        logger.info(f"Partitioning document: {document_path}")

        all_chunks = []
        splitter = CharacterTextSplitter(
            separator="\n",
            keep_separator=True,
            chunk_size=max_size,
            chunk_overlap=self.chunk_overlap,
        )

        doc = fitz.open(document_path)
        current_text = ""
        for page in doc:
            text = page.get_text("block")

            if len(text) > max_size:
                all_chunks.append(
                    {"text": current_text, "metadata": {"page": page.number}}
                )
                chunks = splitter.split_text(text)
                for chunk in chunks:
                    logger.info(
                        f"Flushing chunk. Length: {len(chunk)}, page: {page.number}"
                    )
                    all_chunks.append(
                        {"text": chunk, "metadata": {"page": page.number}}
                    )
                current_text = ""

            elif len(current_text + text) >= max_size:
                if current_text != "":
                    all_chunks.append(
                        {"text": current_text, "metadata": {"page": page.number}}
                    )
                logger.info(
                    f"Flushing chunk. Length: {len(current_text)}, page: {page.number}"
                )
                current_text = text

            # Otherwise, add element's text to current chunk, without re-assigning the page number
            else:
                current_text += text

        # Filter out empty docs
        all_chunks = [
            chunk for chunk in all_chunks if chunk["text"].strip().replace(" ", "")
        ]
        return all_chunks


class DocumentSplitter:
    def __init__(self, config: Config) -> None:
        self._splitter_conf = {
            "pdf": PDFSplitter(chunk_overlap=200).split_document,
        }
        self.document_path_settings = config.embeddings.document_settings
        self.chunk_sizes = config.embeddings.chunk_sizes

    def get_hashes(self) -> pd.DataFrame:
        hash_filename_mappings = []
        logger.info(f"Scanning hashes of the existing files.")

        for setting in self.document_path_settings:
            docs_path = Path(setting.doc_path)
            exclusion_paths = [str(e) for e in setting.exclude_paths]

            for scan_extension in setting.scan_extensions:
                extension = scan_extension.value

                # Create a list of document paths to process. Filter out paths in the exclusion list
                paths = [
                    p
                    for p in list(docs_path.glob(f"**/*.{extension}"))
                    if (not self.is_exclusion(p, exclusion_paths)) and (p.is_file())
                ]
                hashes = [
                    {"filename": str(path), "filehash": get_md5_hash(path)}
                    for path in paths
                ]
                hash_filename_mappings.extend(hashes)
        return pd.DataFrame(hash_filename_mappings)

    def split(
        self, restrict_filenames: Optional[List[str]] = None
    ) -> Tuple[List[Document], pd.DataFrame, pd.DataFrame]:
        """Splits documents based on document path settings

        Returns:
            List[Document]: List of documents
        """
        all_docs = []

        # Maps between file name and it's hash
        hash_filename_mappings = []

        # Mapping between hash and document ids
        hash_docid_mappings = []

        for setting in self.document_path_settings:
            passage_prefix = setting.passage_prefix
            docs_path = Path(setting.doc_path)
            documents_label = setting.label
            exclusion_paths = [str(e) for e in setting.exclude_paths]

            for scan_extension in setting.scan_extensions:
                extension = scan_extension.value
                for chunk_size in self.chunk_sizes:  # type: ignore
                    logger.info(f"Scanning path for extension: {extension}")

                    # Create a list of document paths to process. Filter out paths in the exclusion list
                    paths = [
                        p
                        for p in list(docs_path.glob(f"**/*.{extension}"))
                        if not self.is_exclusion(p, exclusion_paths)
                    ]

                    # Used when updating the index, we don't need to parse all files again
                    if restrict_filenames is not None:
                        logger.warning(
                            f"Restrict filenames specificed. Scanning at most {len(restrict_filenames)} files."
                        )
                        paths = [p for p in paths if str(p) in set(restrict_filenames)]

                    splitter = self._splitter_conf[extension]

                    # Get additional parser setting for a given extension, if present
                    additional_parser_settings = setting.additional_parser_settings.get(
                        extension, dict()
                    )

                    (
                        docs,
                        hf_mappings,
                        hd_mappings,
                    ) = self._get_documents_from_custom_splitter(
                        document_paths=paths,
                        splitter_func=splitter,
                        max_size=chunk_size,
                        passage_prefix=passage_prefix,
                        label=documents_label,
                        **additional_parser_settings,
                    )

                    logger.info(f"Got {len(docs)} chunks for type: {extension}")
                    all_docs.extend(docs)
                    hash_filename_mappings.extend(hf_mappings)
                    hash_docid_mappings.extend(hd_mappings)

        all_hash_filename_mappings = pd.DataFrame(hash_filename_mappings)
        all_hash_docid_mappings = pd.concat(hash_docid_mappings, axis=0)

        return all_docs, all_hash_filename_mappings, all_hash_docid_mappings

    def is_exclusion(self, path: Path, exclusions: List[str]) -> bool:
        """Checks if path has parent folders in list of exclusions

        Args:
            path (Path): _description_
            exclusions (List[str]): List of exclusion folders

        Returns:
            bool: True if path is in list of exclusions
        """

        exclusion_paths = [Path(p) for p in exclusions]
        for ex_path in exclusion_paths:
            if ex_path in path.parents:
                logger.info(
                    f"Excluding path {path} from documents, as path parent path is excluded."
                )
                return True
        return False

    def _get_documents_from_custom_splitter(
        self,
        document_paths: List[Path],
        splitter_func,
        max_size,
        passage_prefix: str,
        label: str,
        **additional_kwargs,
    ) -> Tuple[List[Document], List[dict], List[pd.DataFrame]]:
        """Gets list of nodes from a collection of documents

        Examples: https://gpt-index.readthedocs.io/en/stable/guides/primer/usage_pattern.html
        """

        all_docs = []

        # Maps between file name and it's hash
        hash_filename_mappings = []

        # Mapping between hash and document ids
        hash_docid_mappings = []

        if passage_prefix:
            logger.info(f"Will add the following passage prefix: {passage_prefix}")

        for path in document_paths:
            logger.info(
                f"Processing path using custom splitter: {path}, chunk size: {max_size}"
            )

            # docs_data = splitter_func(text, max_size)
            filename = str(path)
            additional_kwargs.update({"filename": filename})
            docs_data = splitter_func(path, max_size, **additional_kwargs)
            file_hash = get_md5_hash(path)

            path = urllib.parse.quote(str(path))  # type: ignore
            logger.info(path)

            docs = [
                Document(
                    page_content=passage_prefix + d["text"],
                    metadata={
                        **d["metadata"],
                        **{
                            "source": str(path),
                            "chunk_size": max_size,
                            "document_id": str(uuid1()),
                            "label": label,
                        },
                    },
                )
                for d in docs_data
            ]
            all_docs.extend(docs)

            # Add hash to filename mapping and hash to doc ids mapping
            hash_filename_mappings.append(dict(filename=filename, filehash=file_hash))

            df_hash_docid = (
                pd.DataFrame()
                .assign(docid=[d.metadata["document_id"] for d in docs])
                .assign(filehash=file_hash)
            )

            hash_docid_mappings.append(df_hash_docid)

        logger.info(f"Got {len(all_docs)} nodes.")
        return all_docs, hash_filename_mappings, hash_docid_mappings


HASH_BLOCKSIZE = 65536


def get_md5_hash(file_path: Path) -> str:
    hasher = hashlib.md5()

    with open(file_path, "rb") as file:
        buf = file.read(HASH_BLOCKSIZE)
        while buf:
            hasher.update(buf)
            buf = file.read(HASH_BLOCKSIZE)

    return hasher.hexdigest()


"""## Sparse Embeddings"""


def split(iterable: List, chunk_size: int):
    """Splits a list to chunks of size `chunk_size`"""

    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]


"""## Vector Store Functions"""


def get_embedding_model(config: EmbeddingModel):
    """Loads an embedidng model

    Args:
        config (EmbeddingModel): Configuration for the embedding model

    Raises:
        TypeError: if model is unsupported
    """

    # logger.info(f"Embedding model config: {config}")
    # model_type = MODELS.get(config.type, None)

    # if model_type is None:
    #     raise TypeError(f"Unknown model type. Got {config.type}")

    return SentenceTransformerEmbeddings(
        model_name=config.model_name, **config.additional_kwargs
    )


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


class VectorStoreChroma:
    def __init__(self, persist_folder: str, config: Config):
        self._persist_folder = persist_folder
        self._config = config
        self._embeddings = get_embedding_model(config.embeddings.embedding_model)
        self.batch_size = 200

        self._retriever = None
        self._vectordb = None

    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = self._load_retriever()
        return self._retriever

    @property
    def vectordb(self):
        if self._vectordb is None:
            self._vectordb = Chroma(
                persist_directory=self._persist_folder,
                embedding_function=self._embeddings,
            )
        return self._vectordb

    def unload(self):
        self._vectordb = None
        self._retriever = None

        gc.collect()

    def create_index_from_documents(
        self,
        all_docs: List[Document],
        clear_persist_folder: bool = True,
    ):
        if clear_persist_folder:
            pf = Path(self._persist_folder)
            if pf.exists() and pf.is_dir():
                logger.warning(f"Deleting the content of: {pf}")
                shutil.rmtree(pf)

        logger.info("Generating and persisting the embeddings..")

        vectordb = None
        for group in tqdm.tqdm(
            chunker(all_docs, size=self.batch_size),
            total=int(len(all_docs) / self.batch_size),
        ):
            ids = [d.metadata["document_id"] for d in group]
            if vectordb is None:
                vectordb = Chroma.from_documents(
                    documents=group,  # type: ignore
                    embedding=self._embeddings,
                    ids=ids,
                    persist_directory=self._persist_folder,  # type: ignore
                )
            else:
                vectordb.add_texts(
                    texts=[doc.page_content for doc in group],
                    embedding=self._embeddings,
                    ids=ids,
                    metadatas=[doc.metadata for doc in group],
                )
        logger.info("Generated embeddings. Persisting...")
        if vectordb is not None:
            vectordb.persist()
        vectordb = None

    def _load_retriever(self, **kwargs):
        # vectordb = Chroma(persist_directory=self._persist_folder, embedding_function=self._embeddings)
        return self.vectordb.as_retriever(**kwargs)

    # def add_documents(self, docs: List[Document]):
    #     """Adds new documents to existing vectordb

    #     Args:
    #         docs (List[Document]): List of documents
    #     """

    #     logger.info(f"Adding embeddings for {len(docs)} documents")
    #     # vectordb = Chroma(persist_directory=self._persist_folder, embedding_function=self._embeddings)
    #     for group in tqdm.tqdm(
    #         chunker(docs, size=self.batch_size), total=int(len(docs) / self.batch_size)
    #     ):
    #         ids = [d.metadata["document_id"] for d in group]
    #         self.vectordb.add_texts(
    #             texts=[doc.page_content for doc in group],
    #             embedding=self._embeddings,
    #             ids=ids,
    #             metadatas=[doc.metadata for doc in group],
    #         )
    #     logger.info("Generated embeddings. Persisting...")
    #     self.vectordb.persist()

    # def delete_by_id(self, ids: List[str]):
    #     logger.warning(f"Deleting {len(ids)} chunks.")
    #     # vectordb = Chroma(persist_directory=self._persist_folder, embedding_function=self._embeddings)
    #     self.vectordb.delete(ids=ids)
    #     self.vectordb.persist()

    def get_documents_by_id(self, document_ids: List[str]) -> List[Document]:
        """Retrieves documents by ids

        Args:
            document_ids (List[str]): list of document ids

        Returns:
            List[Document]: list of documents belonging to document_ids
        """

        results = self.retriever.vectorstore.get(
            ids=document_ids, include=["metadatas", "documents"]
        )  # type: ignore
        docs = [
            Document(page_content=d, metadata=m)
            for d, m in zip(results["documents"], results["metadatas"])
        ]
        return docs

    def similarity_search_with_relevance_scores(
        self, query: str, k: int, filter: Optional[dict]
    ) -> List[Tuple[Document, float]]:
        # If there are multiple key-value pairs, combine using AND rule - the syntax is chromadb specific
        if isinstance(filter, dict) and len(filter) > 1:
            filter = {"$and": [{key: {"$eq": value}} for key, value in filter.items()]}
            print("Filter = ", filter)

        return self.retriever.vectorstore.similarity_search_with_relevance_scores(
            query, k=self._config.semantic_search.max_k, filter=filter
        )


class CustomLlamaLangChainModel(LLM):
    @classmethod
    def from_parameters(cls, model_path, model_init_kwargs, model_kwargs, **kwargs):
        cls.model = Llama(model_path=str(model_path), **model_init_kwargs)
        cls.model_kwargs = model_kwargs
        cls.model_path = model_path
        cls.streaming = True
        return cls(**kwargs)

    def __del__(self):
        self.model.__del__()

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        if self.streaming:
            # If streaming is enabled, we use the stream
            # method that yields as they are generated
            # and return the combined strings from the first choices's text:
            combined_text_output = ""
            for token in self.stream(prompt=prompt, stop=stop, run_manager=run_manager):
                combined_text_output += token["choices"][0]["text"]
            return combined_text_output
        else:
            result = self.model(prompt=prompt, **self.model_kwargs)
            return result["choices"][0]["text"]

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_path": self.model_path}, **self.model_kwargs}

    def stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> Generator[Dict, None, None]:
        """Yields results objects as they are generated in real time.

        BETA: this is a beta feature while we figure out the right abstraction.
        Once that happens, this interface could change.

        It also calls the callback manager's on_llm_new_token event with
        similar parameters to the OpenAI LLM class method of the same name.

        Args:
            prompt: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            A generator representing the stream of tokens being generated.

        Yields:
            A dictionary like objects containing a string token and metadata.
            See llama-cpp-python docs and below for more.

        Example:
            .. code-block:: python

                from langchain.llms import LlamaCpp
                llm = LlamaCpp(
                    model_path="/path/to/local/model.bin",
                    temperature = 0.5
                )
                for chunk in llm.stream("Ask 'Hi, how are you?' like a pirate:'",
                        stop=["'","\n"]):
                    result = chunk["choices"][0]
                    print(result["text"], end='', flush=True)

        """
        result = self.model(prompt=prompt, stream=True, **self.model_kwargs)
        for chunk in result:
            token = chunk["choices"][0]["text"]
            log_probs = chunk["choices"][0].get("logprobs", None)
            if run_manager:
                run_manager.on_llm_new_token(
                    token=token, verbose=self.verbose, log_probs=log_probs
                )
            yield chunk


splitter = DocumentSplitter(config)
all_docs, all_hash_filename_mappings, all_hash_docid_mappings = splitter.split()

persist_folder: str = "/Users/alleon_g/code/punfinder/db/"
vs = VectorStoreChroma(persist_folder, config)
vs.create_index_from_documents(all_docs=all_docs)

vs._load_retriever()

query = "My coffee is cold, what should I do?"

max_k = 5
res = vs.similarity_search_with_relevance_scores(query, k=max_k, filter=None)

retrieved_doc_ids = [r[0].metadata["document_id"] for r in res]

relevant_docs = None
if retrieved_doc_ids:
    relevant_docs = vs.get_documents_by_id(document_ids=list(retrieved_doc_ids))

context = ""
for doc in relevant_docs:
    context += f"ON PAGE {doc.metadata['page']}\n{doc.page_content}\n\n"

logger.info(f"Context: {context}")

model_path = "/Users/alleon_g/code/punfinder/cache/openorca-zephyr-7b.Q5_K_M.gguf"

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path=model_path,
    temperature=0,
    max_tokens=1024,
    n_ctx=4096,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    # template="<|system|>\n{context}</s>\n<|user|>\n{question}</s>\n<|assistant|>",
    template="<|system|>\nYou are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\n</s>\n<|user|>\n{question}</s>\n<|assistant|>",
    # template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\nQuestion: {question}\n\nContext: {context}\n\nAnswer:",
)

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""

template = """
<|system|>
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}
</s>
<|user|>
{question}</s>
<|assistant|>
"""

rag_prompt_custom = PromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


retriever = vs._load_retriever()

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt_custom  # prompt_template
    | llm
    | StrOutputParser()
)

rag_chain.invoke(query)
