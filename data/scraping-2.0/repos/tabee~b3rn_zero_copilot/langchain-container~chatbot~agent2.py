from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import asyncio
import requests
from urllib.parse import quote
import os

# dirty hack to switch between local and docker container, depending on the environment sys_path
sys_path = os.getenv('DATA_PATH', default=os.path.join(os.path.dirname(__file__)))
server_name = "fastapi"
if str(sys_path).startswith('/workspaces'):
    server_name = "127.0.0.1"
    #print(f"workspaces ... set server_name to {server_name}")
# end dirty hack


def get_answer(question):
    """ Wrapper-Funktion für get_suggestions, die die erforderlichen Parameter übergibt. """
    if question:
        encoded_question = quote(question) # Kodieren des Strings für die URL
        response = requests.get(f'http://{server_name}:80/vectorstore/answers-questions/{question}')

        if response.status_code == 200:
            suggestions = response.json()
            return suggestions
        else:
            return "Sorry, I don't know the answer to your question."



def agent_for(topic):

    model = ChatOpenAI(verbose=False)
    template = """Answer the question based only on the following context:
    {context}

    context: {question}\n
    ==========================\n
    answer always in german. cite the source (context) always in german.
    if the context not suitable, please answer with "no suitable context".
    """
    prompt = ChatPromptTemplate.from_template(template)

    vectorstore_resp = get_answer(topic)
    vectorstore = FAISS.from_texts(
        vectorstore_resp, embedding=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever()

    retrieval_chain = (
        {
            "context": retriever.with_config(run_name="Docs"),
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    for s in retrieval_chain.stream(topic):
        yield s


if __name__ == "__main__":
    for chunk in agent_for(topic="was ist ahv?"):
        print(chunk, end="", flush=True)
