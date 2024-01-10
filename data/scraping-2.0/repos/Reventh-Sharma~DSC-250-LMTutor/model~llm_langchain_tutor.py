import os

import torch
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ChatMessageHistory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from model.llm_encoder import LLMBasedEmbeddings
from query_engineering import add_query_instruction

# Pipeline type dictionary
PIPELINE_TYPE = {
    "lmsys/vicuna-7b-v1.3": "text-generation",
    "stas/tiny-random-llama-2": "text-generation",
}


class LLMLangChainTutor:
    def __init__(
        self,
        doc_loader="dir",
        embedding="openai",
        llm="openai",
        vector_store="faiss",
        openai_key=None,
        token="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",
        embed_device="cuda",
        llm_device="cuda",
        cache_dir=".cache",
        debug=False,
        aggregation="mean",
        hidden_state_id=-1,
    ) -> None:
        """
        Wrapper class for conversational retrieval chain.
        Args:
            doc_loader: Loader for documents. Default is 'dir'.
            embedding: Embedding model to embed document and queries. Default is 'openai'.
            llm: Language model for generating results for query output. Default is 'openai'.
            vector_store: Vector store to store embeddings and associated documents. Default is 'faiss'.
            openai_key: Key for openai, out of scope for now.
            embed_device: Device to use for embedding. Default is 'cuda'.
            llm_device: Device to use for llm. Default is 'cuda'.
            cache_dir: Directory to store cache files. Default is '~/.cache'.
        """
        self.openai_key = openai_key
        self.token = token
        self.llm_name = llm
        self.embedding_name = embedding
        self.embed_device = embed_device
        self.llm_device = llm_device
        self.cache_dir = cache_dir
        self.debug = debug
        self.aggregation = aggregation
        self.hidden_state_id = hidden_state_id

        self._document_loader(doc_loader=doc_loader)
        self._embedding_loader(embedding=embedding)
        self._vectorstore_loader(vector_store=vector_store)
        self._memory_loader()

    def _document_loader(self, doc_loader):
        """
        Args:
            doc_loader: Loader for documents, currently only supports 'dir'.
        """
        if doc_loader == "dir":
            self.doc_loader = DirectoryLoader
        else:
            raise NotImplementedError

    def _embedding_loader(self, embedding):
        """
        This function initializes the embedding model, and is the key part of our project.
        Args:
            embedding: Embedding model to embed document and queries

        Returns:

        """
        if embedding == "openai":
            os.environ["OPENAI_API_KEY"] = self.openai_key
            self.embedding_model = OpenAIEmbeddings()
        elif embedding == "instruct_embedding":
            self.embedding_model = HuggingFaceInstructEmbeddings(
                query_instruction="Represent the query for retrieval: ",
                model_kwargs={
                    "device": self.embed_device,
                },
                encode_kwargs={"batch_size": 32},
                cache_folder=self.cache_dir,
            )
        elif embedding.startswith("hf"):  # If an LLM is chosen from HuggingFace
            llm_name = embedding.split("_")[-1]
            self.base_embedding_model = AutoModelForCausalLM.from_pretrained(
                llm_name,
                torch_dtype=torch.float16,
                cache_dir=self.cache_dir,
                token=self.token,
                output_hidden_states=True,
            ).to(self.embed_device)
            self.base_embedding_tokenizer = AutoTokenizer.from_pretrained(
                llm_name, token=self.token, device=self.embed_device
            )
            self.embedding_model = LLMBasedEmbeddings(
                model=self.base_embedding_model,
                tokenizer=self.base_embedding_tokenizer,
                device=self.embed_device,
                aggregation=self.aggregation,
                hidden_state_id=self.hidden_state_id,
            )
        else:
            raise NotImplementedError
            # self.embedding_model = HuggingFaceEmbeddings(
            #     model_kwargs={
            #         "device": self.embed_device,
            #     },
            #     encode_kwargs={"batch_size": 32},
            #     cache_folder=self.cache_dir,
            # )

    def _vectorstore_loader(self, vector_store):
        """
        Args:
            vector_store: Vector store to store embeddings and associated documents. Default is 'faiss'.
        """
        if vector_store == "faiss":
            self.vector_store = FAISS

    def _memory_loader(self):
        """
        Buffer to store conversation chatbot's converstational history
        """
        self.memory = ChatMessageHistory()

    def _load_document(self, doc_path, glob="*.pdf", chunk_size=400, chunk_overlap=0):
        """
        Loads document from the given path and splits it into chunks of given size and overlap.
        Args:
            doc_path: Path to the document
            glob: Glob pattern to use to find files. Defaults to "**/[!.]*"
               (all files except hidden).
            chunk_size: Size of tokens in each chunk
            chunk_overlap: Number of overlapping chunks within consecutive documents.
        """
        docs = self.doc_loader(
            doc_path,
            glob=glob,
            show_progress=True,
            use_multithreading=True,
            max_concurrency=16,
        ).load()  ### many doc loaders

        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(docs)

    def generate_vector_store(
        self,
        doc_path,
        vec_path,
        glob="*.pdf",
        chunk_size=400,
        chunk_overlap=0,
        query_choice=None,
    ):
        """
        Generates vector store from the documents and embedding model
        """
        logger.info("Creating vector store...")

        # split documents
        splitted_documents = self._load_document(
            doc_path, glob, chunk_size, chunk_overlap
        )

        # add query prefix
        if query_choice == "1":
            query_prefix = "Summarize the following in 10 word: "
        elif query_choice == "2":
            query_prefix = "You are a teaching assistant. Explain this information to your students:"
        else:
            query_prefix = ""
        logger.info(f"Using query prefix: {query_prefix}")

        # generate vector store
        if query_prefix:
            splitted_documents = add_query_instruction(query_prefix, splitted_documents)
        self.gen_vectorstore = self.vector_store.from_documents(
            splitted_documents, self.embedding_model
        )

        # save vectorstore
        logger.info("Saving vector store...")
        self.gen_vectorstore.save_local(folder_path=vec_path)

    def load_vector_store(self, vec_path):
        """Load vectors from existing folder_path"""
        self.gen_vectorstore = self.vector_store.load_local(
            folder_path=vec_path, embeddings=self.embedding_model
        )

    def similarity_search_topk(self, query, k=4):
        """Top k-similarity search"""
        retrieved_docs = self.gen_vectorstore.similarity_search(query, k=k)
        return retrieved_docs

    def similarity_search_thres(self, query, thres=0.8, k=10):
        """Similarity search with which qualify threshold"""
        retrieval_result = self.gen_vectorstore.similarity_search_with_score(query, k)
        retrieval_result.sort(key=lambda x: x[1], reverse=True)
        retrieval_result = [d[0] for d in retrieval_result]

        return retrieval_result

    def conversational_qa_init(self):
        """
        Creates a 'qa' object of type ConversationalRetrievalChain, which creates response to given queries based on
        retreived documents from the vector store.
        """
        # mark first conversation as true and reset memory
        self.first_conversation = True
        self._memory_loader()

        # setup conversational qa chain
        if self.llm_name == self.embedding_name:
            llm_name = self.llm_name.split("_")[-1]
            llm = self.base_embedding_model
            tokenizer = self.base_embedding_tokenizer
        elif self.llm_name == "openai":
            llm = OpenAI(temperature=0)
        elif self.llm_name.startswith("hf"):
            llm_name = self.llm_name.split("_")[-1]
            llm = AutoModelForCausalLM.from_pretrained(
                llm_name,
                temperature=0.7,
                torch_dtype=torch.float16,
                cache_dir=self.cache_dir,
                token=self.token,
            ).to(self.llm_device)
            tokenizer = AutoTokenizer.from_pretrained(
                llm_name, token=self.token, device=self.llm_device
            )
        else:
            raise NotImplementedError

        # set generation pipeline
        self.gen_pipe = pipeline(
            PIPELINE_TYPE[llm_name],
            model=llm,
            tokenizer=tokenizer,
            device=self.llm_device,
            max_new_tokens=512,
            return_full_text=False,
        )

    def conversational_qa(self, user_input):
        """
        Return output of query given a user input.
        Args:
            user_input: User input query
        Returns:
            output: Output of the query using LLM and previous buffers
        """
        FIRST_PROMPT = "A chat between a student user and a teaching assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context.\n"
        PROMPT_TEMPLTATE = "CONTEXT: {context} \n USER: {user_input} \n ASSISTANT:"

        # retrieve relevant documents as context
        context = " \n ".join(
            [each.page_content for each in self.similarity_search_topk(user_input, k=5)]
        )

        # create prompt
        if self.first_conversation:
            prompt = FIRST_PROMPT + PROMPT_TEMPLTATE.format(
                context=context, user_input=user_input
            )
            self.first_conversation = False
        else:
            prompt = (
                self.memory.messages[-1]
                + "\n\n "
                + PROMPT_TEMPLTATE.format(context=context, user_input=user_input)
            )

        # query model and return output
        output = self.gen_pipe(prompt)[0]["generated_text"]
        self.memory.add_message(prompt + output)
        return output
