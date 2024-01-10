"""Langchain functionality"""

# these three lines swap the stdlib sqlite3 lib with the pysqlite3 package
__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import glob
import os
import re
import subprocess
from typing import Any, Optional, Type
from urllib.parse import urljoin, urlsplit

import git
import requests
from bs4 import BeautifulSoup
from langchain.chains import ReduceDocumentsChain, RetrievalQAWithSourcesChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources.base import BaseQAWithSourcesChain
from langchain.chains.qa_with_sources.map_reduce_prompt import (
    EXAMPLE_PROMPT,
    QUESTION_PROMPT,
)
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, WebBaseLoader
from langchain.document_loaders.base import BaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationSummaryMemory
from langchain.schema import BasePromptTemplate
from langchain.schema.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.vectorstore import VectorStore
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult
from langchain_core.prompts import PromptTemplate
from streamlit.delta_generator import DeltaGenerator


def get_links(url: str):
    """Get all unique links from a web page, excluding the anchors (#) and other websites (sith different netloc)."""
    website_netloc = urlsplit(url).netloc
    links = [url]
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for i, link in enumerate(soup.find_all('a', href=True)):
                absolute_url = urljoin(url, link['href'])
                # filter out links to other websites, anchors and duplicates
                if website_netloc in absolute_url and "#" not in absolute_url and absolute_url not in links:
                    links.append(absolute_url)
    except Exception as e:
        print(f"Error parsing {url}: {e}", file=sys.stderr)

    return links


class RepoRagChatAssistant:
    """Assistant for retrieval augmented generation powered chat with sources from a repository and web documentation."""

    SUCCESS_MSG = "SUCCESS"
    DB_FOLDER = "chroma_db"

    def __init__(self, streamlit_output_placeholder: DeltaGenerator):
        self.streamlit_output_placeholder = streamlit_output_placeholder
        self.model = None
        self.assistant_identity = None
        self.db = None
        self.retriever = None
        self.memory = None
        self.partial_steps_llm = None
        self.combine_llm = None
        self.qa_chain = None
        self.repo_local_path = None
        self.repo_url = None
        self.documentation_url = None
        self.num_documents_to_retrieve = None
        self.output_callback = None
        self.embedding = None
        self.last_formatted_message = None

    @property
    def db_path(self) -> str:
        """Local path to the vector database."""
        return os.path.join(self.repo_path, RepoRagChatAssistant.DB_FOLDER)

    @property
    def repo_path(self) -> str:
        """Local path to the repository."""
        assert self.repo_url and self.repo_local_path, "First set repo_url and repo_local_path."
        return os.path.join(self.repo_local_path, self.repo_url.split("/")[-1])

    def __call__(self, prompt: str) -> dict[str, Any]:
        assert self.qa_chain, "First load the model and the repository."
        result = self.qa_chain(prompt)
        self.last_formatted_message = self.output_callback.final_message
        return result

    def load_model(
        self,
        openai_api_key: str,
        model: str,
        assistant_identity: str,
        temperature: float = 0.1,
        max_tokens_per_generation: int = 500,
    ) -> str:
        """Create assistant's LLMs, memory and if possible combine to the whole chain.

        This method can be called in 2 different cases:
        - 1) model (LLMs+memory) doesn't exist:
            -> create new LLMs & memory
        - 2) model (LLMs+memory) exists:
            -> create new LLMs, keep the same memory - don't erase previous conversation
        In any case when retriever already exists - create the chain (assistant will be ready to use through __call__)
        """
        self.model = model
        self.assistant_identity = assistant_identity
        self.output_callback = StreamingOutCallbackHandler(self.streamlit_output_placeholder)
        self.embedding = OpenAIEmbeddings(openai_api_key=openai_api_key, disallowed_special=())

        self.combine_llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens_per_generation,
            streaming=True,
            callbacks=[self.output_callback],
        )
        self.partial_steps_llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens_per_generation,
        )
        # creates memory with the new partial_steps_llm, on reload;, if memory already exists, keep the internal buffer
        self._create_memory(keep_buffer=True)

        self.qa_chain = None  # loading new model invalidates the qa_chain

        # if the repo/vector DB retriever is already loaded, we can create new qa_chain
        if self.retriever:
            self._create_qa_chain()

        return RepoRagChatAssistant.SUCCESS_MSG

    def load_repo(
        self,
        repo_url: str,
        repo_local_path: str,
        documentation_url: Optional[str] = None,
        num_documents_to_retrieve: int = 4,
    ) -> str:
        """Load the repository and documentation and create the vector DB.

        Executes one of the following (both require OpenAI embeddings - for this the model has to be loaded):
        1a) If there is existing Chroma DB at the correct location - just load that
        1b) Create Chroma DB with new document chunks from scratch
            1) CLone or (pull existing) repo
            2) Parse and chunk the documents from the local repo and from the online documentation
            3) Add the chunks to the DB
            4) Persist the DB
        2) Creates retriever
        3) if the LLMs exist creates the whole chain (assistant will be ready to use through __call__), if memory
           existed it is erased
        """
        if not self.embedding:
            return "Embeddings not initialized. First load the model."
        elif repo_url == self.repo_url:
            return "This repository is already loaded."

        self.repo_url = repo_url
        self.repo_local_path = repo_local_path
        self.documentation_url = documentation_url

        if os.path.exists(self.db_path) and "chroma.sqlite3" in os.listdir(self.db_path):
            # repository was already cloned and DB was persisted - load existing DB
            # TODO: check the DB is up to date (was created from the up to date repo, contains documentation if provided)
            self.db = Chroma(
                embedding_function=self.embedding,
                persist_directory=self.db_path,
                collection_metadata={"hnsw:space": "cosine"},
            )
        else:
            # clone/update repo and create new DB with embeddings
            result = self._clone_or_update_git_repo()
            if result != RepoRagChatAssistant.SUCCESS_MSG:
                self.repo_url, self.repo_local_path = None, None
                return result
            self.db = self._create_db()

        self.create_retriever(num_documents_to_retrieve)

        # loading new repo/DB invalidates the qa_chain and memory
        self.qa_chain, self.memory = None, None

        # if the model is already loaded, we can create new memory and qa_chain
        if self.combine_llm and self.partial_steps_llm:
            self._create_memory()
            self._create_qa_chain()

        return RepoRagChatAssistant.SUCCESS_MSG

    def similarity_search_with_score(self, question: str) -> list[tuple[float, str]]:
        if self.db:
            return self.db.similarity_search_with_score(question, k=self.num_documents_to_retrieve)

    def _clone_or_update_git_repo(self) -> str:
        """Clones or updates the repository from self.repo_url to self.repo_path."""
        if os.path.exists(self.repo_path):
            try:
                repo = git.Repo(self.repo_path)
                repo.remotes.origin.pull()
            except git.GitError as e:
                return f"Error while pulling the repository {self.repo_url}: {str(e)}"
        else:
            try:
                self._clone_git_repo_in_subprocess()
                return RepoRagChatAssistant.SUCCESS_MSG
            except Exception as e:
                return f"Error while cloning the repository {self.repo_url}: {str(e)}"

    def _clone_git_repo_in_subprocess(self) -> None:
        """Clones the repository in a subprocess and updates the Streamlit progressbar - useful for large repositories."""
        self.streamlit_output_placeholder.progress(0, text="Cloning repo:")
        with subprocess.Popen(
            [
                "git",
                "clone",
                "--progress",  # necessary to force the output of the git command to stderr
                self.repo_url,
                self.repo_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # git clone --progress outputs to stderr, redirect to stdout
        ) as p:
            for line in iter(lambda: p.stdout.read1().decode('utf-8'), ""):
                receiving_objects_token = "Receiving objects"
                if line.startswith(receiving_objects_token):
                    line = line.split(receiving_objects_token)[-1]  # take just the last update
                    current, total = re.findall(r'\((\d+)/(\d+)\)', line)[0]
                    msg = f"Cloning repo: {receiving_objects_token}{line}"
                    self.streamlit_output_placeholder.progress(int(current) / int(total), text=msg)

    def _create_db(self, persist: bool = True) -> VectorStore:
        py_files = glob.glob(f"{self.repo_path}/**/*.py", recursive=True)  # get all .py files in the repo
        md_files = glob.glob(f"{self.repo_path}/**/*.md", recursive=True)
        documentation_links = get_links(self.documentation_url) if self.documentation_url else []
        chunks = []
        progressbar_current = 0
        long_step = 10
        progressbar_total = len(py_files) + len(md_files) + len(documentation_links) * long_step + long_step
        progress_msg = "##### Initial creation of the vector database - might take couple of minutes:"

        def load_and_split(
            files: list[str], language: Language, loader: Type[BaseLoader] = TextLoader, progress_step: int = 1
        ) -> list[Document]:
            """Loads and splits files into chunks using specific language separator; updates the global progressbar."""
            if not files:
                return []

            nonlocal progress_msg
            nonlocal progressbar_current
            documents = []
            file_type = "online documentation" if loader is WebBaseLoader else f"{language.value} files"

            for i, file in enumerate(files):
                progressbar_current += progress_step
                msg = f"{progress_msg}\n\n- {i}/{len(files)} Loading and splitting {file_type}..."
                self.streamlit_output_placeholder.progress(progressbar_current / progressbar_total, text=msg)
                try:
                    documents.append(loader(file).load()[0])
                except Exception as e:
                    print(f"Error loading {file}: {e}. \nSkipping the file.", file=sys.stderr)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=RecursiveCharacterTextSplitter.get_separators_for_language(language),
            )
            progress_msg = msg
            return splitter.split_documents(documents)

        # Load and split to chunks documents from various sources:
        chunks.extend(load_and_split(py_files, Language.PYTHON))
        chunks.extend(load_and_split(md_files, Language.MARKDOWN))
        chunks.extend(load_and_split(documentation_links, Language.PYTHON, WebBaseLoader, long_step))

        # create new DB with embeddings
        progressbar_current += long_step
        msg = f"{progress_msg}\n\n- Creating the vector database..."
        self.streamlit_output_placeholder.progress(progressbar_current / progressbar_total, text=msg)
        db = Chroma.from_documents(
            chunks,
            self.embedding,
            persist_directory=self.db_path,
            collection_metadata={"hnsw:space": "cosine"},
        )
        if persist:
            self.streamlit_output_placeholder.markdown(msg)
            db.persist()

        self.streamlit_output_placeholder.markdown("")  # clear the output
        return db

    def create_retriever(self, num_documents_to_retrieve):
        self.num_documents_to_retrieve = num_documents_to_retrieve
        # search_type="mmr" Max. Marginal Relevance - optimizes for similarity to query and diversity among documents
        self.retriever = self.db.as_retriever(
            search_type="similarity", search_kwargs={"k": self.num_documents_to_retrieve}
        )

    def _create_qa_chain(self) -> None:
        # initialize the output_callback with the current repo info
        repo = git.Repo(self.repo_path)
        self.output_callback.repo_branch = repo.active_branch.name  # used to create source links for the output
        self.output_callback.repo_url = self.repo_url
        self.output_callback.repo_path = self.repo_path

        self.qa_chain = RepoRetrievalQAWithSourcesChain.from_llms(
            assistant_identity=self.assistant_identity,
            question_llm=self.partial_steps_llm,
            combine_llm=self.combine_llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
        )

    def _create_memory(self, keep_buffer: bool = False) -> None:
        self.memory = ConversationSummaryMemory(
            llm=self.partial_steps_llm,
            memory_key="chat_history",
            output_key="answer",
            human_prefix="User",
            ai_prefix="Assistant",
            buffer=self.memory.buffer if (self.memory and keep_buffer) else "",  # reuse the existing memory content
        )


class StreamingOutCallbackHandler(StreamingStdOutCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    SOURCES_IDENTIFIER = "\nSOURCES:"
    SOURCES_MSG = "Source documents:"

    def __init__(self, streamlit_output_placeholder: DeltaGenerator):
        self.streamlit_output_placeholder = streamlit_output_placeholder
        self.repo_url = None
        self.repo_path = None
        self.repo_branch = None
        self.final_message = ""
        self._message_buffer = ""
        self._source_buffer = ""
        self._sources = []
        self._parsing_message = True

    @property
    def _formatted_sources(self) -> str:
        """String in streamlit markdown format with sources in a bullet list with clickable links."""
        sources_bullet_list = "\n\n - ".join([f"[{source}]({source})" for source in self._sources])
        sources_bullet_list = f"\n\n - {sources_bullet_list}" if sources_bullet_list else ""
        return sources_bullet_list

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Parses each new incoming token and prints formatted message to the streamlit output placeholder.

        Runs on new LLM token. Only available when streaming is enabled.
        Identified document sources are transformed into URLs and formatted as a bullet list.
        """
        if self._parsing_message:
            self._message_buffer += token
            if StreamingOutCallbackHandler.SOURCES_IDENTIFIER in self._message_buffer:
                self._message_buffer = self._message_buffer.replace(StreamingOutCallbackHandler.SOURCES_IDENTIFIER, "")
                self._parsing_message = False
        else:  # parsing sources
            if token == ",":  # append complete source url - change local repo path to url
                # TODO: fix the "/blob/": this will probably only work on Github
                self._sources.append(
                    self._source_buffer.lstrip().replace(self.repo_path, f"{self.repo_url}/blob/{self.repo_branch}")
                )
                self._source_buffer = ""
            else:  # continue parsing single source
                self._source_buffer += token

        streamed_message = (
            self._message_buffer + f"\n\n{StreamingOutCallbackHandler.SOURCES_MSG}"
            if (self._sources or self._message_buffer)
            else ""
        )
        streamed_message += self._formatted_sources
        streamed_message += f"\n\n - {self._source_buffer}" if self._source_buffer else ""
        self.streamlit_output_placeholder.markdown(streamed_message)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        # source still in buffer:
        if self._source_buffer:
            self._sources.append(
                self._source_buffer.lstrip().replace(self.repo_path, f"{self.repo_url}/blob/{self.repo_branch}")
            )

        if self._sources:
            self.final_message = (
                self._message_buffer + f"\n\n{StreamingOutCallbackHandler.SOURCES_MSG} {self._formatted_sources}"
            )
        self.streamlit_output_placeholder.markdown(self.final_message)

        # reset internal variables for the next run
        self._message_buffer = ""
        self._source_buffer = ""
        self._sources = []
        self._sources_links = []
        self._parsing_message = True


repo_combine_prompt_template = """{assistant_identity}\n
Given the chat history, the question and the extracted parts of multiple source files with programming code, all from one repository, create a final answer with references ("SOURCES"). 
If there is no relevant information in the sources summaries, look for it in the last CHAT HISTORY.
If you still don't know the answer, just say that you don't know. Don't try to make up an answer.
If the QUESTION is not an actual question, just a statement, just answer accordingly, e.g.: "Ok.", "Got it.", "Thank you." or "Sure.", ignore the given extracted content.
ALWAYS return a "SOURCES" part in your answer.

CHAT HISTORY: The user asks which classes are available that are derived from BaseMessage. The assistant responds with: "The available child classes of BaseMessage are: HumanMessage, SystemMessage and AIMessage."
=========
QUESTION: Which class is used for holding the parameters for listing tunes?
=========
Content: from typing import Optional\n
from pydantic import BaseModel, ConfigDict, Field\n
# TODO: Update the descriptions import
from genai.schemas.descriptions import TunesAPIDescriptions as tx\n\n
class TunesListParams(BaseModel):
    \"\"\"Class to hold the parameters for listing tunes.\"\"\"\n
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")\n
    limit: Optional[int] = Field(None, description=tx.LIMIT, le=100)
    offset: Optional[int] = Field(None, description=tx.OFFSET)
Source: tmp/ibm-generative-ai/src/genai/schemas/tunes_params.py
Content: from typing import Optional\n
from pydantic import BaseModel, ConfigDict, Field\n
from genai.schemas.descriptions import FilesAPIDescriptions as tx\n\n
class FileListParams(BaseModel):
    \"\"\"Class to hold the parameters for file listing.\"\"\"\n
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")\n
    limit: Optional[int] = Field(None, description=tx.LIMIT, le=100)
    offset: Optional[int] = Field(None, description=tx.OFFSET)
Source: tmp/ibm-generative-ai/src/genai/schemas/files_params.py
=========
FINAL ANSWER: Class used for holding the parameters for listing tunes is TunesListParams.
SOURCES: tmp/ibm-generative-ai/src/genai/schemas/tunes_params.py


CHAT HISTORY: {chat_history}
=========
QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""
REPO_COMBINE_PROMPT = PromptTemplate(
    template=repo_combine_prompt_template,
    input_variables=["assistant_identity", "summaries", "question", "chat_history"],
)


class RepoRetrievalQAWithSourcesChain(RetrievalQAWithSourcesChain):
    """Custom repository question answering chain summarizing the question-relevant parts of retrieved documents.

    Uses one LLM to summarize only the question-relevant parts of the retrieved code chunks and another to combine
    the question, chat history and the summaries into a final answer. The combine LLM prompt contains a single shot
    example. The combine LLM is also personalized with the assistant identity.
    """

    @classmethod
    def from_llms(
        cls,
        assistant_identity: str,
        question_llm: BaseLanguageModel,
        combine_llm: BaseLanguageModel,
        document_prompt: BasePromptTemplate = EXAMPLE_PROMPT,
        question_prompt: BasePromptTemplate = QUESTION_PROMPT,
        combine_prompt: BasePromptTemplate = REPO_COMBINE_PROMPT,
        **kwargs: Any,
    ) -> BaseQAWithSourcesChain:
        """Construct the chain from an LLM."""
        combine_prompt = combine_prompt.partial(assistant_identity=assistant_identity)
        llm_question_chain = LLMChain(llm=question_llm, prompt=question_prompt)
        llm_combine_chain = LLMChain(llm=combine_llm, prompt=combine_prompt)
        combine_results_chain = StuffDocumentsChain(
            llm_chain=llm_combine_chain,
            document_prompt=document_prompt,
            document_variable_name="summaries",
        )
        reduce_documents_chain = ReduceDocumentsChain(combine_documents_chain=combine_results_chain)
        combine_documents_chain = MapReduceDocumentsChain(
            llm_chain=llm_question_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="context",
        )
        return cls(
            combine_documents_chain=combine_documents_chain,
            **kwargs,
        )
