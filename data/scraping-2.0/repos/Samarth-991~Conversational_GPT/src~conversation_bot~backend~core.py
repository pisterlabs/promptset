import os
from langchain.chat_models import ChatOpenAI

from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from typing import Any
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from utils.embedder_model import HuggingFaceEmbeddings
from typing import List, Dict


def create_prompt():
    prompt_template = """
    Analyze conversations between customer and sales executive from context.
    If customer shows interest in service or Property , conversation is a potential lead.  
    Always answer point wise with person names. Don't make up answers
   
    {context}
   
    {chat_history}
   
    Question: {question}
    Answer stepwise: 
    """
    prompt = PromptTemplate(input_variables=["context", "question", "chat_history"], template=prompt_template)
    return prompt


def run_llm(query: str, embedding_model='openai', vector_store='', chat_history: List[Dict[str, Any]] = []) -> Any:
    if embedding_model == 'open_ai':
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceEmbeddings()
    docsearch = FAISS.load_local(vector_store, embeddings=embeddings)
    prompt = create_prompt()
    chat = ChatOpenAI(verbose=True, temperature=0)

    qa = ConversationalRetrievalChain.from_llm(llm=chat,
                                               retriever=docsearch.as_retriever(),
                                               combine_docs_chain_kwargs={"prompt": prompt},
                                               max_tokens_limit=4097
                                               )
    return qa({"question": query, "chat_history": chat_history})
