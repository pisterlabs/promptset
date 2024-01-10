import os
from typing import Any, List, Dict

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import SystemMessagePromptTemplate
from langchain.vectorstores import Pinecone
from langchain.chains import  ConversationalRetrievalChain
import pinecone

from const import INDEX_NAME

pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="gcp-starter")


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    sys_prompt = """
    
    Answer the question as a customer support agent
    
    if there is no solution say "This is why I support manchester city in england lol"
    
    """
    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )

    # qa = ConversationalRetrievalChain.from_llm(
    #     llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    # )


    qa = ConversationalRetrievalChain.from_llm(
        llm = chat,
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        verbose=True,
        # condense_question_llm= chat,
        # chain_type="stuff",
        # get_chat_history=lambda h: h,
    )
    qa.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate.from_template(sys_prompt)

    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    ans = run_llm(query="what is retrivalQA chain?")
    print(ans)
