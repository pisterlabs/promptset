from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field
import streamlit as st

from custom_retrievers.custom_retrievers import get_multi_query_retriever_deep_lake, get_multi_query_retriever_deep_lake_cloud, get_multi_query_retriever

# from dotenv import load_dotenv
#
# load_dotenv()

openai_key = st.secrets["OPENAI_API_KEY"]
embeddings = OpenAIEmbeddings()


def get_retrieval_qa_tools(files: list, language: str = "en", create_db=False) -> list[Tool]:

    if language == "en":
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Helpful Answer:"""
        tool_description = "Useful for when you need to extract information from the following document:"

        input_must_be = "| Input must be:"

    elif language == "fr":
        prompt_template = """Utilisez les éléments de contexte suivants pour extraire toutes les informations 
        nécessaires pour répondre à la question à la fin. Faites une liste détaillée de tous les points discutés dans le 
        contexte. Si aucune information pertinente n'est trouvée pour répondre à la question, indiquez-le clairement et 
        n'essayez pas d'inventer des informations.
        
        Contexte :
        {context}

        Question : {question}
        
        Liste des points d'information extraits : :"""
        tool_description = "Utile lorsque vous souhaitez extraire des informations du document suivant :"
        input_must_be = "| Le paramètre doit être:"

    else:
        raise ValueError("The language must be either 'en' or 'fr'")

    class DocumentInput(BaseModel):
        question: str = Field()

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k-0613", streaming=True)

    tools = []
    for file in files:

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # retriever = get_multi_query_retriever_deep_lake_cloud(file, language=language, create_db=create_db)
        retriever = get_multi_query_retriever(file, language=language)

        qa_tool = RetrievalQA.from_chain_type(llm=llm,
                                              chain_type="stuff",
                                              retriever=retriever,
                                              chain_type_kwargs={"prompt": prompt},
                                              return_source_documents=file['return_source_documents'],
                                              )

        tools.append(
            Tool(
                args_schema=DocumentInput,
                name=file["name"],
                description=f"{tool_description} {file['name']} | {input_must_be} {file['input_format']}",
                func=qa_tool,
                return_direct=file['return_direct'],
            )
        )

    return tools
