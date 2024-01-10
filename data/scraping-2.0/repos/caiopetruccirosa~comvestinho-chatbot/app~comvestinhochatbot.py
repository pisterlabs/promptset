from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Chroma

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

# Constants for  prompt messages templates
SYSTEM_MESSAGE_TEMPLATE = """Você é uma IA assistente cuja função é responder dúvidas sobre o Vestibular da Unicamp 2024 a partir da publicação da Resolução GR-031/2023, de 13/07/2023 que \"Dispõe sobre o Vestibular Unicamp 2024 para vagas no ensino de Graduação\".
Considere a conversa, o contexto e a pergunta dada para dar uma resposta. Caso você não saiba uma resposta, fale 'Me desculpe, mas não tenho uma resposta para esta pergunta' em vez de tentar inventar algo.
----
Conversa:
{chat_history}
----
Contexto:
{context}
----
"""
HUMAN_MESSAGE_TEMPLATE = """Pergunta:
{question}"""

# ComvestinhoChatBot Class
class ComvestinhoChatBot():
    # Inits ComvestinhoChatBot
    def __init__(self):
        # Sets model names
        self.chat_model_name = "gpt-3.5-turbo"
        self.embeddings_model_name = "text-embedding-ada-002"

        # Sets default values
        self.chunk_size = 1000
        self.chunk_overlap = 0
        self.temperature = 0

        # Creates embeddings and chat models
        self.chat_model = ChatOpenAI(
            model_name=self.chat_model_name, 
            temperature=self.temperature,
        )
        self.embeddings_model = OpenAIEmbeddings(
            model=self.embeddings_model_name, 
            chunk_size=self.chunk_size,
        )

        # Creates document vectostore and retriever
        doc_url = "https://www.pg.unicamp.br/norma/31594/0"
        docs = self.__load_webpage(doc_url)
        docs_splits = self.__split_documents(docs)
        self.vectordb = self.__create_doc_vectorstore(docs_splits)
        self.retriever = self.vectordb.as_retriever(search_type="similarity", search_kwargs={"k":4})
        
        # Defines prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE_TEMPLATE),
            HumanMessagePromptTemplate.from_template(HUMAN_MESSAGE_TEMPLATE),
        ])

    # Augments prompt adding chat history and context to question
    def __create_augmented_prompt(self, question, chat_history, related_docs):       
        # Join all documents to a string
        docs_str = '\n'.join([doc.page_content for doc in related_docs])
        
        # Format chat history and join all messages to a string
        role_map = { "user": "Usuário", "assistant": "Sistema"}
        chat_history_formatted = [ f"{role_map[message['role']]}: {message['content']}" for message in chat_history ]
        chat_history_str = '\n'.join(chat_history_formatted)
        
        # Create prompt template with input and output languages as Portuguese
        prompt = self.prompt_template.format_messages(
            input_language="Portuguese", 
            output_language="Portuguese", 
            question=question,
            context=docs_str,
            chat_history=chat_history_str
        )
        return prompt

    # Loads html page and convert it to text
    def __load_webpage(self, page_url):
        loader = AsyncHtmlLoader([page_url])
        html2text = Html2TextTransformer()
        docs = loader.load()
        docs_transformed = html2text.transform_documents(docs)
        return docs_transformed

    # Splits documents to smaller chunks
    def __split_documents(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        docs_splits = text_splitter.split_documents(docs)
        return docs_splits

    # Creates document vector store
    def __create_doc_vectorstore(self, docs):
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings_model,
            persist_directory='./vectorstore'
        )
        vectordb.persist()
        return vectordb

    # Asks the ComvestinhoChatBot a question, saves in chat history and returns the answer
    def ask(self, question, chat_history):
        related_docs = self.retriever.get_relevant_documents(query=question)
        prompt = self.__create_augmented_prompt(question, chat_history, related_docs)
        answer = self.chat_model(prompt)
        return answer.content