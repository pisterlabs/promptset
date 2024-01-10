import os
import time
import openai
import pinecone
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate

from utils.gcp_secret_manager import get_secret

class queryClass:
    def __init__(self):
        load_dotenv()
        
        self.api_key_pinecone =  get_secret("pinecone_api_key")
        self.openai_api_key =  get_secret("openai_api_key")

        pinecone.init(api_key=self.api_key_pinecone, environment="asia-southeast1-gcp")

        index_name = "texts"
        self.index = pinecone.Index(index_name=index_name)
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.vectorstore = Pinecone.from_existing_index(index_name, self.embeddings)

        self.llm = ChatOpenAI(
                temperature=0,
                model="gpt-3.5-turbo-16k",
                openai_api_key=self.openai_api_key,
                )

        self.metadata_field_info = [
            AttributeInfo(
                name="formato",
                description="formato en el que se ofrece el producto",
                type="string",
            ),

            AttributeInfo(
                name='precio',
                description="costo del curso",
                type="integer",
            ),
        ]
        self.search_kwargs = {
            "k":5
        }
        self.document_content_description = "Informacion sobre cursos programacion, precios y direcciones"
    
        self.retriever = SelfQueryRetriever.from_llm(
                self.llm,
                self.vectorstore,
                self.document_content_description,
                self.metadata_field_info,
                verbose=True,
                search_kwargs=self.search_kwargs
            )

    def get_answer(self, question, history):

        context  = self.retriever.get_relevant_documents(question)

        context_to_gtp = self.context_to_gpt(context)
        source_context = [[i.metadata.get('formato', 'N/A'), i.metadata.get('precio', 'N/A'), i.page_content] for i in context]
        history_sliced_str = "\n".join([f"{entry['type']}: {entry['message']}" for entry in history[-6:]])

        prompt_template = PromptTemplate(
                template = """
                Eres un vendedor experto sobre cursos de programación, que responde consultas acerca de los tipos de cursos, beneficios, URLs,y facilita la inscripción además de persuadir al usuario para que se inscriban a los cursos, limitate al contexto.
                Devolver adicionalmente la dirección de la página web donde se encuentra la información.
                Vas responder la siguiente pregunta "{question}" de forma concisa en formato markdown.
                Este es el historial de la conversación:	
                {history_sliced_str}

                """,

                input_variables = ["question", "history_sliced_str"],
            )

        query_enriched = prompt_template.format(question=question,history_sliced_str=history_sliced_str)

        messages = [
                    {"role": "system", "content": str(query_enriched)},
                    {"role": "user", "content": str(context_to_gtp)}
                ]
        
        start_time = time.time()    
        res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=messages,
                api_key=self.openai_api_key,
            )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"get answer from gpt  113: {elapsed_time} segundos")

        data = {
                  "query" : str(query_enriched),
                  "result" : res["choices"][0]["message"]["content"],
                  "source_documents" : source_context,
            }

        return data


    def context_to_gpt(self, context):
        if context == []:
            return "no hay informacion disponible"
        contex_formatted = ""

        for index, i in enumerate(context):
            formato = i.metadata.get('formato', 'N/A')
            precio = i.metadata.get('precio', 'N/A')
            page_content = i.page_content
            contex_formatted = contex_formatted + f"\n---\n formato: {formato} | precio: {precio} \n contexto  {index+1}: {page_content}\n---\n"
        return contex_formatted
