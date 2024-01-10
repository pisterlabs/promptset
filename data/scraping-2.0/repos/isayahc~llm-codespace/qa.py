from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub

# from langhchain.llms import openai
from langchain.llms import OpenAI

from langchain.vectorstores import Chroma

from langchain.chains import RetrievalQA
# from langchain.chains import RetrievalQAChain

from langchain.document_loaders import PyPDFLoader
from langchain.memory import VectorStoreRetrieverMemory

from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import CohereEmbeddings


from langchain.embeddings import HuggingFaceHubEmbeddings, OpenAIEmbeddings

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp
from langchain.retrievers import WikipediaRetriever

from langchain.chains import ConversationalRetrievalChain

from src.llms.merged_dpo_llm import dpo_llm


template = """
You are the friendly AI assistant, who helps the user discover insights on their medical insurance policy, \

You must answer the user's question and never tell the user to "Review the document" as that is the antithesis of your role \

    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question :
------

------

------
{question}
Answer:
"""

retriever = WikipediaRetriever()


prompt = PromptTemplate(
input_variables=[
    "history", 
    "question",
    ],
template=template,
)
memory = ConversationBufferMemory(
    memory_key="history", 
    input_key="question"
    )


qa = ConversationalRetrievalChain.from_llm(
    dpo_llm, 
    retriever=retriever,
    memory=memory,
)

questions = [
    # "What is Apify?",
    # "When the Monument to the Martyrs of the 1830 Revolution was created?",
    # "What is the Abhayagiri Vihāra?",
    # "How big is Wikipédia en français?",
    "what is Cellular automaton?",
    "who created it and for what reason?",
    "List a real lifr use case for it.",
]
chat_history = []

for question in questions:
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")


# qa = RetrievalQAWithSourcesChain.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     return_source_documents=True,
#     verbose=True,
#     chain_type_kwargs={
#         "verbose": True,
#         "memory": memory,
#         "prompt": prompt,
#         "document_variable_name": "context"
#         }
#     )
# https://python.langchain.com/docs/integrations/providers/vectara/vectara_chat#conversationalretrievalchain-with-question-answering-with-sources
x = 0
