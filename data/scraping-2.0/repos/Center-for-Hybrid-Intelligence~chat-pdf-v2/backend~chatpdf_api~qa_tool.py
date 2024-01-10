############# IMPORTS #############

import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()

import pinecone  # For our vectorstore
import openai  # For our LLM

from uuid import uuid4

# LangChain imports. Everything we need is in langchain.
from langchain.vectorstores import Pinecone  # For our vectorstore
from langchain.embeddings.openai import \
    OpenAIEmbeddings  # For our word embeddings, through langchains OpenAIEmbeddings class
from langchain.chat_models import ChatOpenAI  # for LLM
from langchain.text_splitter import \
    RecursiveCharacterTextSplitter  # OBS: This is the function that splits the text into chunks!!!

#### MY ADDITIONS ####
from langchain.chains import ConversationalRetrievalChain  # for conversational QA
from langchain.memory import ConversationBufferMemory  # for conversational QA
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.question_answering.stuff_prompt import system_template

# For tokenization
import tiktoken  # For counting the number of tokens in a text, used for splitting the text into chunks.

from .database import remove_document, add_document, retrieve_documents  # for removing document from database

############# SET ENVIRONMENT VARIABLES #############

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

PINE_API_KEY = os.getenv('PINECONE_API_KEY')

YOUR_ENV = os.getenv('YOUR_ENV')

############# PINECONE #############

pinecone.init(api_key=PINE_API_KEY, environment=YOUR_ENV)


############# FUNCTIONS #############

def get_system_prompt(prompt):
    template = prompt or ''
    template = ''.join([
        'Use the following pieces of context to answer the users question.\n',
        template,
        """. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n
----------------\n
{context}
Question: {question}"""
    ])
    print('template')
    print(template)
    template = template or system_template
    return PromptTemplate(template=template, input_variables=["context", "question"])


class QaTool:  # class for the QA tool
    def __init__(self, chunk_size=400, chunk_overlap=20, chain_type="stuff") -> None:
        self.chunk_overlap = None  # overlap between chunks
        self.chunk_size = None  # size of chunks
        self.tokenizer_name = 'cl100k_base'  # model used for tokenization through tiktoken
        self.namespace = None  # namespace for pinecone
        self.chain_type = chain_type  # type of chain used for QA, is set to "stuff" by default
        self.embedding_model = 'text-embedding-ada-002'  # model used for embedding (OpenAI)
        self.llm_model = 'gpt-4'  # model used for LLM (OpenAI)
        self.model_temperature = 0.0  # temperature for LLM
        self.memory = ConversationBufferMemory( # memory for conversational QA
            memory_key="chat_history",
            output_key='answer',
            return_messages=True
        )
        self.index_name = 'chatpdf-langchain-retrieval-agent'  # name of the pinecone index, go to https://www.pinecone.io/ to see it
        pinecone.list_indexes()  # list all indexes in pinecone
        if self.index_name not in pinecone.list_indexes():
            # we create a new index
            pinecone.create_index(
                name=self.index_name,
                metric='dotproduct',
                dimension=1536
            )
        self.text_field = 'text'
        self.loaded_documents = []

    def tiktoken_len(self, text):
        tiktoken.encoding_for_model(self.llm_model)
        tokenizer = tiktoken.get_encoding(self.tokenizer_name)
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

    def set_namespace(self, namespace):
        self.namespace = namespace

    def set_chunks(self, chunk_size, chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def set_llm(self, llm='gpt-4', temperature=0.0):
        self.llm_model = llm
        self.model_temperature = temperature

    def loading_data_to_pinecone(self, data):
        # =============Warning to change if several indexes
        index = pinecone.GRPCIndex(self.index_name)  # we are connected to the pinecone index

        # INDEXING
        batch_limit = 100  # pinecone doesn't allow more than 100 vectors simultaneous upserting

        texts = []
        metadatas = []

        embed = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=OPENAI_API_KEY)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.tiktoken_len,
            separators=["\n\n", "\n", " ", ""]
            # This is the default list of separators for the text splitter (see langchain docs).
        )
        print("Loading embeddings to Pinecone")
        for i, record in (data.iterrows()):
            # first get metadata fields for this record
            metadata = {
                'id': record['Id'],
                'title': record['Title'],
                'author': record['Author']
            }

            # now we create chunks from the record text
            record_texts = text_splitter.split_text(record['Summary'])

            # create individual metadata dicts for each chunk
            record_metadatas = [{
                "chunk": j, "text": text, **metadata
            } for j, text in enumerate(record_texts)]
            # append these to current batches
            if self.namespace is None:
                raise ValueError("Namespace not set")

            for record_text, record_metadata in zip(record_texts, record_metadatas):
                texts.append(record_text)
                metadatas.append(record_metadata)
                # if we have reached the batch_limit we can add texts
                if len(texts) >= batch_limit:
                    ids = [str(uuid4()) for _ in range(len(texts))]
                    embeds = embed.embed_documents(texts)
                    index.upsert(vectors=zip(ids, embeds, metadatas), namespace=self.namespace)
                    texts = []
                    metadatas = []

        if len(texts) > 0:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embed.embed_documents(texts)
            index.upsert(vectors=zip(ids, embeds, metadatas), namespace=self.namespace)
        self.loaded_documents.append(data['Id'].tolist()[0])

    def delete_all(self):
        index = pinecone.GRPCIndex(self.index_name)  # we are connected to the pinecone index
        index.delete(delete_all=True, namespace=self.namespace)
        self.loaded_documents = []

    def erase_doc(self, document_id):
        remove_document(document_id=document_id)
        index = pinecone.GRPCIndex(self.index_name)  # we are connected to the pinecone index
        index.delete(ids=[document_id], namespace=self.namespace)

    def __call__(self, query, top_closest, system_prompt) -> Any:
        print("Loading embeddings")
        embed = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=OPENAI_API_KEY)

        print("Loading vectorstore")  # We are using Pinecone as a vectorstore through langchain.
        index = pinecone.Index(self.index_name)
        vectorstore = Pinecone(
            index, embed.embed_query, self.text_field, self.namespace
        )

        print("Loading LLM")
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name='gpt-4',
            temperature=0.0
        )

        # Multiple answer mode
        # If we want to ask a single question to whole data base of document see the following commented code
        docs = retrieve_documents(self.namespace)

        results = []
        # memory = ConversationBufferMemory(memory_key="chat_history",  output_key='answer', return_messages=True)
        for doc in docs:
            # filter = {'$and': [{"title":{"$eq": doc.document_title}}, {"author":{"$eq": doc.document_author}}]}
            # filter = {"title": {"$eq": doc.document_title}}
            filter = {
                "title": doc.document_title}  # only one working for now. The other should be working based on the source code and documentation but not in practice
            print("Loading QA for document: ", doc.document_title)

            ############### RETRIEVAL QA ################
            # qa = RetrievalQA.from_chain_type(
            #      llm=llm,
            #      chain_type=self.chain_type,
            #      retriever=vectorstore.as_retriever(search_kwargs={"k": top_closest, "filter": filter}), #for now we are not applying any filter
            #      return_source_documents=True,
            #      verbose=True

            ############### CONVERSATIONAL QA ################

            qa = ConversationalRetrievalChain.from_llm(
                llm=llm,
                chain_type=self.chain_type,
                # condense_question_prompt = PromptTemplate.from_template("Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language."), # default prompt. tweak for wins
                retriever=vectorstore.as_retriever(search_kwargs={"k": top_closest, "filter": filter}),
                # for now we are not applying any filter
                # return_source_documents=False, ### OBS: Changed to false for testing
                combine_docs_chain_kwargs={"prompt": get_system_prompt(system_prompt)},
                verbose=True,
                memory=self.memory
            )
            try:
                print("Querying...")
                # results.append(qa({"query": query})) #TODO : Erreur après cette ligne : "illegal condition for field Title, got {\"eq\":\"Cover Letter EN.pdf\"}","details":[]}
                result_from_query = qa({"question": query})  # FOR CONVERSATIONAL QA, QUERY IS CHANGED TO QUESTION
                results.append(result_from_query['answer'])  # FOR CONVERSATIONAL QA, QUERY IS CHANGED TO QUESTION
            except openai.error.InvalidRequestError as e:
                print((f"Invalid request error: {e}"))
                error_message = str(e)
                return error_message, 401  # Invalid request, might have reached maximum tokens

        # docs : the title and author of the document, responses : the result of the query and the source documents
        final_response = [(document.document_title, document.document_author, result) for document, result in
                          zip(docs, results)]
        return final_response

        # # Unique answer mode based on the entire set of pdf in the namespace
        # results = []
        # print("Loading QA for document: ", doc.document_title)
        # qa = RetrievalQA.from_chain_type(
        #     llm=llm,
        #     chain_type=self.chain_type,
        #     retriever=vectorstore.as_retriever(search_kwargs={"k": top_closest}), #for now we are not applying any filter
        #     return_source_documents=True,
        #     verbose=True
        # )
        # try:
        #     print("Querying...")
        #     results.append(qa({"query": query})) #TODO : Erreur après cette ligne : "illegal condition for field Title, got {\"eq\":\"Cover Letter EN.pdf\"}","details":[]}
        # except openai.error.InvalidRequestError as e:
        #     print((f"Invalid request error: {e}"))
        #     error_message = str(e)
        #     return error_message, 401 #Invalid request, might have reached maximum tokens
        #     
        #  #docs : the title and author of the document, responses : the result of the query and the source documents
        # final_response = [document.document_title, document.document_author, result]
        # return final_response

    def __repr__(self) -> str:
        return f"QaTool(chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}, chain_type={self.chain_type}), index_name={self.index_name}, namespace={self.namespace})"
