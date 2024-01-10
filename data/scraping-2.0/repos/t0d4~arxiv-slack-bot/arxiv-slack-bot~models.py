import glob
import os
import pickle
import re
import string
import textwrap
from typing import List, Optional

import config
import deepl
import torch
from arxiv import Result, Search  # arxiv.Result represents each thesis
from exceptions import DocumentAlreadyVectorizedException
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from slack_sdk.models.blocks import (
    ActionsBlock,
    ButtonElement,
    MarkdownTextObject,
    PlainTextObject,
    SectionBlock,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline, pipeline


class Document:
    def __init__(self, arxiv_doc: Result) -> None:
        self.arxiv_doc: Result = arxiv_doc
        self.key_points: Optional[List[str]] = None

    def __str__(self) -> str:
        return str(self.arxiv_doc)

    def get_formatted_message(self) -> List[SectionBlock]:
        """
        create slack message that contains information of this document.

        Note: Document must be summarized using `DocumentHandler.summarize_documents` before executing this method.

        Parameters:
            None

        Returns:
            List[Block]: slack message (consists of blocks) that contains information of this document.
        """
        if not self.key_points:
            raise Exception("This document is not summarized yet.")

        key_points_formatted = ""
        for key_point in self.key_points:
            key_points_formatted += f"• {key_point}\n"

        blocks = [
            SectionBlock(text=MarkdownTextObject(text=f"*[ {self.arxiv_doc.title} ]*")),
            SectionBlock(
                text=MarkdownTextObject(
                    text=f"<{self.arxiv_doc.entry_id}|view on arxiv.org>"
                )
            ),
            SectionBlock(text=MarkdownTextObject(text=key_points_formatted)),
            ActionsBlock(
                block_id="talk_about_thesis-button-block",
                elements=[
                    ButtonElement(
                        text=PlainTextObject(text="Discuss it!"),
                        value=self.arxiv_doc.get_short_id(),
                        action_id="discuss-button-action",
                    )
                ],
            ),
        ]

        return blocks


class Searcher:
    def __init__(self, initial_query) -> None:
        self._query: str = initial_query

    def search(self, **kwargs) -> List[Document]:
        """
        search for theses on arXiv.

        Note: when both `query` and `id_list` is NOT specified, then search with `self._query`.

        Parameters:
            kwargs: keyword arguments passed to `arxiv.Seach.__init__()`.
        Returns:
            List[Documents]: list of `Document` instances which contains the search result
        """
        if "query" in kwargs or "id_list" in kwargs:
            search = Search(**kwargs)
        else:
            search = Search(query=self._query, **kwargs)
        return [Document(arxiv_doc=result) for result in search.results()]

    def update_query(self, new_query: str) -> None:
        """
        update default search query.
        after query is updated, `new_query` will be used if no query is specified for `search()`.

        Parameters:
            new_query: str - query string to replace current query.
        Returns:
            None
        """
        self._query = new_query


class DocumentHandler:
    def __init__(self, llm_model_name_or_path: str, deepl_token: str) -> None:
        """
        initialize `DocumentHandler` instance.

        Parameters:
            model_name_or_path: str - model name or path to the LLM, which will be passed to `from_pretrained` method in `AutoClass` by hugging face transformers.

        Returns:
            None
        """
        self._pipe = self._load_model(llm_model_name_or_path=llm_model_name_or_path)
        # self.hf_pipeline will store LangChain's wrapper around Hugging Face pipeline.
        self.hf_pipeline = HuggingFacePipeline(pipeline=self._pipe)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.translator = deepl.Translator(auth_key=deepl_token)
        self.embedding = SentenceTransformerEmbeddings(
            model_kwargs={"device": config.SENTENCE_TRANSFORMER_DEVICE}
        )

    def _load_model(self, llm_model_name_or_path) -> Pipeline:
        # TODO: add procedure to handle in case available VRAM is too small.

        """
        load LLM to RAM or VRAM.

        Parameters:
            None

        Returns:
            None
        """

        print(f"Loading model: {llm_model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=llm_model_name_or_path,
            device_map=config.HUGGING_FACE_DEVICE_MAP,
        )
        if config.LOAD_IN_16BIT:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=llm_model_name_or_path,
            device_map=config.HUGGING_FACE_DEVICE_MAP,
            torch_dtype=torch_dtype,
        )

        self._pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            top_k=50,  # TODO: there's room for optimization
            top_p=0.95,
        )
        print("Successfully loaded the model")
        return self._pipe

    def _get_response_from_llm(self, prompt: str) -> str:
        if not self._pipe:
            raise Exception(
                "Model is not loaded yet. Call `load_model` to load it in advance."
            )

        # TODO: make this more efficient by processing documents as a batch
        model_output = self._pipe(prompt)[0]["generated_text"]  # type:ignore
        return model_output  # type:ignore

    def _translate_key_points_of_documents(
        self, docs: List[Document]
    ) -> List[Document]:
        for doc in docs:
            if not doc.key_points:
                raise Exception(
                    f"Document named {doc.arxiv_doc.title} is not summarized yet."
                )
            doc.key_points = [
                translation_result.text
                for translation_result in self.translator.translate_text(
                    text=doc.key_points, target_lang="ja"
                )  # type:ignore because translate_text MUST return List[TextResult]
                # when List[str] is passed to the argument "text"
            ]
        return docs

    def summarize_documents(
        self, docs: List[Document], translate=True
    ) -> List[Document]:
        """
        summarize documents by extracting their key points, and then
        write them to `Document.key_points`.

        Note: this method DESTRUCTIVELY changes the docs passed as argument.

        Parameters:
            docs: List[Document] - list of documents to process.
            translate: bool (default: True) - whether to translate key points into Japanese.

        Returns:
            docs: List[Document] - list of documents written their key points in `Document.key_points`
        """
        # TODO: consider better prompt
        prompt_base = string.Template(
            textwrap.dedent(
                """\
            You're a professional summary writer. Read the abstract delimited by triple backquotes and summarize it, \
            then write exactly 3 key points in the output section indicated with [OUTPUT]

            ```${abstract}```

            [OUTPUT]
            -- key point 1:
            """
            )
        )

        for doc in docs:
            prompt = prompt_base.safe_substitute({"abstract": doc.arxiv_doc.summary})
            output = self._get_response_from_llm(prompt=prompt)
            key_points = re.findall(
                pattern=r"(?<=-- key point \d:)[\s\S]+?(?=--|$)", string=output
            )
            for idx, key_point in enumerate(key_points):
                key_points[idx] = key_point.strip()
            doc.key_points = key_points

        if translate:
            docs = self._translate_key_points_of_documents(docs=docs)

        return docs

    def convert_pdf_into_vector_db(self, thread_id: str, thesis_id: str) -> None:
        """
        retrieve PDF from arxiv.org and split it into smaller parts, then vectorize them,
        finally store the generated vector store on disk.

        Parameters:
            thread_id: str - ID of the thread that the message with the button is a parent of.
            thesis_id: str - ID of the thesis described in the message with the button.

        Returns:
            None

        Raises:
            DocumentAlreadyVectorizedException - raised when the specified thesis has already been vectorized.
        """
        vector_db_save_path = os.path.join(
            config.VECTOR_DB_SAVE_DIR, f"{thesis_id}-{thread_id}"
        )

        if os.path.exists(vector_db_save_path):
            raise DocumentAlreadyVectorizedException(
                "this document is already converted into vector db."
            )

        # create document loader to load pdf
        loader = PyPDFLoader(file_path=f"https://arxiv.org/pdf/{thesis_id}.pdf")
        # split it into chunks
        pdf_pages = loader.load()
        # prepare text splitter TODO: optimize the parameters
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=32)
        # split pdf pages into smaller chunks
        docs = splitter.split_documents(documents=pdf_pages)
        # vectorize the document using FAISS
        db = FAISS.from_documents(documents=docs, embedding=self.embedding)
        # save the embedding as a local folder ( folder name will be f"{thesis_id}-{thread_id}" )
        db.save_local(folder_path=vector_db_save_path)

    def _search_vector_db_by_thread_id(self, thread_id: str) -> FAISS:
        """
        search for existing vector store that stores information about the thesis
        described in the parent message of the thread `thread_id`, then returns
        the instance of the vector store.

        Parameters:
            thread_id: str - ID of the thread whose parent message describes the thesis you want to discuss.

        Returns:
            FAISS - instance of the found vector store

        Raises:
            FileNotFoundError - raised when vector store related to `thread_id` was not found.
        """
        db_paths = glob.glob(
            pathname=os.path.join(config.VECTOR_DB_SAVE_DIR, f"*-{thread_id}")
        )
        if not db_paths:
            raise FileNotFoundError(
                f"db related to thread_id: {thread_id} was not found."
            )

        db_path = db_paths[0]
        return FAISS.load_local(folder_path=db_path, embeddings=self.embedding)

    def answer_question_with_source_documents(
        self, thread_id: str, question: str, translate=True
    ):
        """
        load vector store and chat history from disk and generate answer to `question`.

        Parameters:
            thread_id: str - ID of the thread where the question message by the user is sent.
            question: str - question by the user, written in arbitrary language.
            translate: bool (default: True) - whether to translate the text when inputting question to LLM and outputting answer from LLM.

        Returns:
            (answer, source_documents): Tuple[str, List[Document]] - the answer from LLM and the source documents LLM referred to.
        """
        db = self._search_vector_db_by_thread_id(thread_id=thread_id)

        chat_history_filepath = os.path.join(
            config.CHAT_HISTORY_SAVE_DIR, f"{thread_id}.pkl"
        )
        if os.path.exists(path=chat_history_filepath):
            with open(chat_history_filepath, "rb") as file:
                chat_history = pickle.load(file)
        else:
            chat_history = []

        qa = ConversationalRetrievalChain.from_llm(
            llm=self.hf_pipeline,
            retriever=db.as_retriever(
                k=2
            ),  # retrieve 2 most related parts from the document
            chain_type="refine",
            return_source_documents=True,
            max_tokens_limit=1024,  # max tokens limit of Vicuna-7B
        )

        # translate question into English before passing it to LLM
        if translate:
            question = self.translator.translate_text(
                text=question, target_lang="en-US"
            ).text  # type:ignore

        result = qa({"question": question, "chat_history": chat_history})
        answer = result["answer"]

        chat_history.append((question, answer))
        with open(chat_history_filepath, "wb") as file:
            pickle.dump(obj=chat_history, file=file)

        # translate answer into Japanese before sending it to the user
        if translate:
            answer = self.translator.translate_text(
                text=answer, target_lang="ja"
            ).text  # type:ignore

        source_documents = result["source_documents"]

        return answer, source_documents
