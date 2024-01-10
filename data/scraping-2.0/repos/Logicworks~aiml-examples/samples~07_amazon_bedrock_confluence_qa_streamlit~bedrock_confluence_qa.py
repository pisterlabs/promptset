import os
import sys 

module_path = ".."
sys.path.append(os.path.abspath(module_path))

import langchain
from langchain.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from utils import bedrock



boto3_bedrock = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None),
)


class BedrockConfluenceQA:
    def __init__(self, config: dict = {}):
        self.config = config
        self.embedding = None
        self.vectordb = None
        self.llm = None
        self.qa = None
        self.retriever = None
        self.model_id = None

    def init_embeddings(self) -> None:
        # AWS Bedrock Embeddings
        self.embedding = BedrockEmbeddings(client=boto3_bedrock)

    def init_models(self, parameters: dict = {}) -> None:
        self.parameters = parameters
        max_token_count = self.parameters.get("max_token_count", 512)
        temprature = self.parameters.get("temprature", 1)
        top_p = self.parameters.get("top_p", 1)
        top_k = self.parameters.get("top_k", 1)
        model_id = self.parameters.get("model_id", "amazon.titan-tg1-large")
        self.model_id = model_id
        # AWS Bedrock titan
        if "claude" in model_id:
            self.llm = Bedrock(
                model_id=model_id,
                client=boto3_bedrock,
                model_kwargs={
                    "max_tokens_to_sample":max_token_count,
                    "temperature": temprature,
                    "top_k": top_k,
                    "top_p": top_p,
                }
            )
        if "titan" in model_id:
            self.llm = Bedrock(model_id=model_id, client=boto3_bedrock, model_kwargs= {
                "maxTokenCount": max_token_count,
                "temperature": temprature,
                "topP": top_p,
            })
        if "ai21" in model_id:
            self.llm = Bedrock(model_id=model_id, client=boto3_bedrock, model_kwargs= {
                "maxTokens": max_token_count,
                "temperature": temprature,
                "topP": top_p,
            })


    def vector_db_confluence_docs(self, force_reload: bool = False) -> None:
        """
        creates vector db for the embeddings and persists them or loads a vector db from the persist directory
        """
        persist_directory = self.config.get("persist_directory", None)
        confluence_url = self.config.get("confluence_url", None)
        username = self.config.get("username", None)
        api_key = self.config.get("api_key", None)
        space_key = self.config.get("space_key", None)
        if persist_directory and os.path.exists(persist_directory) and not force_reload:
            ## Load from the persist db
            self.vectordb = FAISS.load_local("faiss_index", embeddings=self.embedding)
        else:
            loader = ConfluenceLoader(
                url=confluence_url, username=username, api_key=api_key
            )
            documents = loader.load(space_key=space_key, limit=50)
            ## 2. Split the texts
            text_splitter = RecursiveCharacterTextSplitter(
                # Set a really small chunk size, just to show.
                # Make sure the chunk size does not exceed titan text embeddings max tokens (512)
                chunk_size=1000,
                chunk_overlap=100,
                # separators=["\n", "\n\n"]

            )
            docs = text_splitter.split_documents(documents)
            print(len(docs))

            ## 3. Create Embeddings and add to chroma store
            ##TODO: Validate if self.embedding is not None
            vectorstore_faiss = FAISS.from_documents(
                docs,
                self.embedding,
            )
            VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)
            self.vectordb = vectorstore_faiss
            # vectorstore_faiss_aws.save_local("faiss_index")

    def retreival_qa_chain(self):
        """
        Creates retrieval qa chain using vectordb as retrivar and LLM to complete the prompt
        """
        ##TODO: Use custom prompt
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 10})
        # self.qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff",retriever=self.retriever)
#         prompt_template = """Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just  say that you don't know, don't try to make up an answer.
#         {context}

#         Question: {question}
#         Assistant:"""

#         prompt_template = """Human: Please use the context below to craft a succinct response to the following question. If you don't have the information, it's okay to state that you're unaware instead of inventing an answer.
#         {context}

#         Question: {question}
#         Assistant:"""

        prompt_template = """Human: Utilize the context provided to formulate a comprehensive response to the following question. If you're uncertain about the answer, it's perfectly fine to acknowledge that you're unsure rather than providing speculative information.
            {context}

            Question: {question}
            Assistant:"""
    
        ## used for the bulk answers generation
        prompt_template = """# INSTRUCTION
            Answer any question about onboarding or company-related topics at LogicWorks acting as a onboarding manager. If you don't have the information, it's okay to state that you're unaware instead of inventing an answer.
            Utilize the context provided to formulate a comprehensive response to the following question. If you don't have the information, it's okay to state that you're unaware instead of inventing an answer.

            # CONTEXT
            {context}

            # QUESTION
            {question}

            Assistant:
        """


        prompt_template = """User: Answer the question based only on the information provided between ##. If you don't know the answer, just  say that you don't know, don't try to make up an answer.
        #
        {context}
        #

        Question: {question}
        Assistant:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
    
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )

    def answer_confluence(self, question: str) -> str:
        """
        Answer the question
        """
        answer = self.qa({"query": question})
        return answer
