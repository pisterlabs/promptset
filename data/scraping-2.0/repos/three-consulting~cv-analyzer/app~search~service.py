import pandas as pd
from db.core import client, embedding_function
from config import Settings
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

def semantic_search(query: str, filter: str, k: int = 10):
    try:
        collection = client.get_collection(
            name=Settings().openai_collection_name,
            embedding_function=embedding_function,
        )

        if filter:
            res = collection.query(
                query_texts=[query],
                where_document={"$contains": filter},
                n_results=k,
            )
        else:
            res = collection.query(
                query_texts=[query],
                n_results=k,
            )

        return pd.DataFrame(
            {
                "distances": list(res["distances"][0]),
                "documents": list(res["documents"][0]),
            }
        )
    except Exception as e:
        return pd.DataFrame({
            "Error": [str(e)],
        })

def hyde_search(query: str, filter: str, k: int = 10):
    try:
        system_template = """
        You are a helpful assistant who creates resumes for various jobs.

        Good resume for a applicant contains:

        - skills of the applicant
        - experience of the applicant

        We only need the required skills for the job in the resume. All personal identity details should be left out.
        """

        human_template = """
        Create a resume for {query} with {filter}
        """

        system_message = SystemMessagePromptTemplate.from_template(system_template)
        human_message = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

        chat = ChatOpenAI(temperature=0.9, openai_api_key=Settings().openai_api_key)
        chain = LLMChain(llm=chat, prompt=chat_prompt)

        hyde = chain.run(query=query, filter=filter)

        collection = client.get_collection(
            name=Settings().openai_collection_name,
            embedding_function=embedding_function,
        )

        res = collection.query(
            query_texts=[hyde],
            n_results=k,
        )

        return pd.DataFrame({
            "distances": list(res["distances"][0]),
            "documents": list(res["documents"][0]),
        })

    except Exception as e:
        return pd.DataFrame({
            "Error": [str(e)]
        })        

