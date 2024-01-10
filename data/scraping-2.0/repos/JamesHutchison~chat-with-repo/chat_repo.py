import functools
import pickle
import os
from pathlib import Path
import textwrap
from typing import Any
from langchain.document_loaders import TextLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.llms.textgen import TextGen
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.messages import SystemMessage, HumanMessage
from ooba_api import LlamaInstructPrompt, OobaApiClient, Parameters
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

import langchain

from ooba_langchain import BlockingLangChainOobaLLM

CACHE_DIR = Path(".repo_chat")
CACHE_FILE = CACHE_DIR / "docs.pickle"

DIR_LIST = ['fill me in']
# TODO: add command line arguments for file types
# TODO: add way to filter by file type at query time
# FILE_EXTENSIONS = {"ts", "tsx", "py", "json", "js", "jsx", "html", "md", "css"}
FILE_EXTENSIONS = {"py", "ts", "tsx", "js", "jsx", "html", "css"}

# TODO: some better way
USE_LLAMA_INSTRUCT = True

langchain.debug = True

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


@functools.lru_cache(1)
def get_deeplake_db():
    print("using deeplake vectorstore")
    my_activeloop_org_id = "your-org-id"
    my_activeloop_dataset_name = "your-dataset-name"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

    return DeepLake(dataset_path=dataset_path, embedding_function=embeddings)


@st.cache_resource()
def in_memory_vectorstore() -> FAISS:
    import faiss
    from langchain.docstore import InMemoryDocstore

    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    embedding_size = 1536

    index = faiss.IndexFlatL2(embedding_size)
    return FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


def do_load(db) -> None:
    dir_list = DIR_LIST
    docs = []
    file_extensions = FILE_EXTENSIONS
    ignored_extensions = set()

    for root_dir in dir_list:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for file in filenames:
                file_path = os.path.join(dirpath, file)

                if (
                    file_extensions
                    and (extension := os.path.splitext(file)[1][1:])
                    not in file_extensions
                ):
                    if extension not in ignored_extensions:
                        ignored_extensions.add(extension)
                        print(f"Ignoring file extension {extension}")
                    continue

                splitter_extension = extension
                if extension in ("ts", "tsx", "jsx", "json"):
                    splitter_extension = "js"
                if extension in ("md", "css"):
                    splitter_extension = None

                CHUNK_SIZE = 1000
                CHUNK_OVERLAP = 100
                if splitter_extension:
                    text_splitter = RecursiveCharacterTextSplitter.from_language(
                        splitter_extension, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
                    )
                else:
                    text_splitter = CharacterTextSplitter(
                        separator="\n\n", chunk_size=CHUNK_SIZE
                    )
                docs.extend(
                    text_splitter.create_documents(
                        [Path(file_path).read_text()], metadatas=[{"source": file_path}]
                    )
                )
                print(f"Num docs: {len(docs)}")

    print("Loading documents...")
    cache_docs(docs)
    db.add_documents(docs)


def cache_docs(docs):
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir()

    pickle.dump(docs, CACHE_FILE.open("wb"))


def do_load_of_cached_embeddings(db):
    assert CACHE_FILE.exists()

    docs = pickle.load(CACHE_FILE.open("rb"))

    db.add_documents(docs)


@st.cache_resource()
def do_load_in_memory():
    print("Using in-memory vectorstore")
    db = in_memory_vectorstore()
    if Path(CACHE_DIR).exists():
        print("Loading cached embeddings. Run with '--clear-cache' to recompute.")
        do_load_of_cached_embeddings(db)
    else:
        print("Loading with fresh embeddings. This will use the OpenAI embeddings API.")
        do_load(db)


# TODO: this is from map reduce, which is no longer happening. Remove?
@functools.lru_cache(1)
def question_prompt() -> PromptTemplate:
    question_prompt_template = """Use the following portion of code from the source file to answer the question.
    Return all text verbatim. Include the path. Include portions around it necessary for context.
    Consider whether the user is looking for code, documentation, or sample files when deciding relevance. Do not provide advice.
    Do not speak to the user, only return maybe relevant code snippets. If it is completely irrelevant, return "no code snippets".

    EXAMPLE output, code is absolutely not relevant:
    no code snippets

    EXAMPLE output, code is maybe relevant:

    `./app/users/get.py`
    ```python
    def retrieve_user(request):
        return Users.get(request.post['id'])
    ```

    `./app/urls.py`
    ```python
    user_url = "http://localhost/users/<id:int>"
    ```
    ```python
    class Users:
        url = user_url
        def get(self, request):
            return retrieve_user(request).to_output()
    ```

    CONTEXT:
    {context}
    Question: {question}
    Relevant text or code, if any:"""
    if USE_LLAMA_INSTRUCT:
        question_prompt_template = LlamaInstructPrompt(
            system_prompt="Do not give the user advice. Only return relevant code snippets that may help with answering the question.",
            prompt=question_prompt_template,
        ).full_prompt()
    return PromptTemplate(
        template=textwrap.dedent(question_prompt_template),
        input_variables=["context", "question"],
    )


def combine_prompt() -> PromptTemplate:
    combine_prompt_template = """Given the following code files exerpts, answer the question the best you can.
    Provide many possible answers, but prioritize the best. Consider whether the file is for testing
    purposes or not. Return entire functions if you can. Always indicate the paths to the source files.
    Include relevant code snippets necessary for context. Do not give the user general advice, only
    specific answers. If you don't know the answer, don't try to make up an answer.
    Only consider responses from the given sources and context.

    QUESTION: {question}
    =========
    {summaries}
    =========
    SEARCH RESULTS:"""
    if USE_LLAMA_INSTRUCT:
        combine_prompt_template = LlamaInstructPrompt(
            system_prompt="Do not give the user advice. Only return relevant code snippets that may help with answering the question.",
            prompt=combine_prompt_template,
        ).full_prompt()
    return PromptTemplate(
        template=textwrap.dedent(combine_prompt_template),
        input_variables=["summaries", "question"],
    )


class RetrievalQAWithSourcesChainWithPathInDoc(RetrievalQAWithSourcesChain):
    _query_override: str | None = None

    @property
    def query_override(self) -> str:
        return self._query_override

    @query_override.setter
    def query_override(self, value: str | None) -> None:
        self._query_override = value

    def get_relevant_documents(
        self,
        query: str,
        *,
        callbacks=None,
        tags: list[str] = None,
        metadata: dict[str, Any] | None = None,
        run_name: str | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        query = self._query_override or query
        return super().get_relevant_documents(
            query,
            callbacks=callbacks,
            tags=tags,
            metadata=metadata,
            run_name=run_name,
            **kwargs,
        )

    def _get_docs(self, inputs: dict[str, Any], *, run_manager):
        """
        Add the source path to the page_content of each document.
        """
        result = super()._get_docs(inputs, run_manager=run_manager)
        for doc in result:
            doc.page_content = f"from {doc.metadata['source']}:\n" + doc.page_content
        return result


def do_streamlit(
    in_memory: bool, no_load: bool, llm_strategy: dict
) -> None:
    from streamlit_chat import message

    use_ooba = llm_strategy['primary_llm'] == 'ooba'
    if not use_ooba:
        use_open_ai = True
    else:
        use_open_ai = False
    if in_memory:
        db = in_memory_vectorstore()
        if not no_load:
            do_load_in_memory()
    else:
        db = get_deeplake_db()
    retriever = db.as_retriever()

    # Set the search parameters for the retriever
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 10

    # Create a ChatOpenAI model instance
    if use_ooba:
        ooba_client = OobaApiClient(ooba_url)
        llm = BlockingLangChainOobaLLM(
            base_prompt=LlamaInstructPrompt(
                system_prompt="Do not give the user advice. Only return relevant code snippets that may help with answering the question.",
                prompt="",
            ),
            api_client=ooba_client,
            parameters=Parameters(temperature=0.1, max_new_tokens=750),
        )
    else:
        if use_open_ai:
            llm = ChatOpenAI(model_name=llm_strategy['primary_llm'])
        else:
            raise Exception("No LLM specified!")

    llm_combine_chain = LLMChain(llm=llm, prompt=combine_prompt())
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_combine_chain,
        document_prompt=PromptTemplate(
            template="Content: {page_content}\nSource: {source}",
            input_variables=["page_content", "source"],
        ),
        document_variable_name="summaries",
    )
    qa_chain = RetrievalQAWithSourcesChainWithPathInDoc(
        combine_documents_chain=combine_documents_chain, retriever=retriever
    )

    # Set the title for the Streamlit app
    st.title(f"Chat with Code Repository")

    # Initialize the session state for placeholder messages.
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["ready"]

    if "past" not in st.session_state:
        st.session_state["past"] = ["hello"]

    chat_history_container = st.container()

    # A field input to receive user queries
    user_input = st.text_area("", key="input")

    user_input_modified = (
        user_input
        + " Be sure to include neighboring context that is required to know why it is relevant."
    )
    user_input_modified += (
        "\nProvide thorough explanations of how to accomplish anything and also relevant code snippets "
        "and file paths from relevant source files. Be sure to include neighboring context that "
        "is required to know why it is relevant."
    )
    send = st.button("Send")

    with chat_history_container:
        # Search the databse and add the responses to state
        if send and user_input:
            if use_ooba:
                vector_search_prompt = ooba_client.instruct(
                    LlamaInstructPrompt(
                        system_prompt="Convert the following user prompt into a query suitable for vector search",
                        prompt=user_input,
                    )
                )
            else:
                vector_search_prompt = llm(
                    [
                        SystemMessage(
                            content="Convert the following user prompt into a query suitable for vector search"
                        ),
                        HumanMessage(content=user_input),
                    ]
                ).content
            RetrievalQAWithSourcesChainWithPathInDoc.query_override = (
                vector_search_prompt
            )

            output = qa_chain(user_input_modified, return_only_outputs=True)
            answer = output["answer"]
            st.session_state.past.append(user_input)
            st.session_state.generated.append(answer)

        # Create the conversational UI using the previous states
        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
                message(st.session_state["generated"][i], key=str(i))


@st.cache_resource()
def clear_cache() -> None:
    if CACHE_DIR.exists():
        for file in CACHE_DIR.iterdir():
            file.unlink()
        CACHE_DIR.rmdir()


def get_llm_strategy(use_ooba: bool, ooba_url: str, use_gpt3: bool, use_gpt4: bool):
    more_kwargs = {}
    
    selection_count = sum([use_ooba, use_gpt3, use_gpt4]):
    if selection_count != 1:
        raise Exception("Specify one of --use-ooba, --gpt-3, --gpt-4")

    if use_gpt3:
        primary_llm = "gpt-3-turbo"
    elif use_gpt4:
        primary_llm = "gpt-4"
    elif use_ooba:
        primary_llm = "ooba"
        more_kwargs["ooba_url"] = ooba_url
    return {
        "primary_llm": primary_llm,
    } | more_kwargs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--load-deeplake", action="store_true")
    parser.add_argument("--in-memory", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--ooba-url", type=str, default="http://localhost:8000", help="URL for --use-ooba")
    parser.add_argument(
        "--no-load",
        action="store_true",
        help="Disable loading embeddings into the in-memory vector database.",
    )
    parser.add_argument("--gpt-3", action="store_true", help="Use GPT-3")
    parser.add_argument("--gpt-4", action="store_true", help="Use GPT-4")
    parser.add_argument("--use-ooba", action="store_true", help="Use Ooba-booga's Text Gen Web UI")
    args = parser.parse_args()

    if args.clear_cache:
        clear_cache()
    if args.load_deeplake:
        do_load(in_memory=False)
    else:
        do_streamlit(
            args.in_memory, args.no_load, get_llm_strategy(args.use_ooba, args.ooba_url, args.gpt_3, args.gpt_4)
        )
