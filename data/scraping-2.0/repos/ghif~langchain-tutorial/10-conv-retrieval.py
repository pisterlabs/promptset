from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import json
from langchain.chains import LLMChain, ConversationChain, ConversationalRetrievalChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import (
    PromptTemplate,
    FewShotPromptTemplate,
    SemanticSimilarityExampleSelector
)
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import CharacterTextSplitter

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 0,
        length_function = len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstore(text_chunks, existing_vectorstore=None):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    if existing_vectorstore is not None:
        vectorstore = existing_vectorstore
        # vectorstore.add_texts(texts=text_chunks, embedding=embeddings)
        new_vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        vectorstore.merge_from(new_vectorstore)
    else:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


template = """You are a knowledgeable customer service agent from Pusat Bantuan Merdeka Belajar Kampus Merdeka (MBKM).
Use the historical conversation below to answer various questions from users.
If you don't know the answer, just say I don't know. Don't make up an answer.
The answer given must always be in Indonesian with a friendly tone.

Current conversation:
{chat_history}
Human: {question}
AI Assistant:"""

# Load context from external sources
markdown_path = "chat_samples.md"
loader = UnstructuredMarkdownLoader(markdown_path)
data = loader.load()

raw_text = data[0].page_content

chunks = get_text_chunks(raw_text)
db = get_vectorstore(chunks)
retriever = db.as_retriever()


# LLM
chat_llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    # verbose=True,
    temperature=0.0,
    # streaming=True
)

memory = ConversationSummaryMemory(
    llm=chat_llm,
    memory_key="chat_history",
    return_messages=True
)

# Chain
prompt = PromptTemplate.from_template(template)
# chain = LLMChain(
#     prompt=prompt,
#     llm=chat_llm,
#     memory=memory,
#     verbose=True
# )

# combine_docs_chain = StuffDocumentsChain(...)

# conv_chain = ConversationalRetrievalChain(
#     question_generator=chain,
#     retriever=retriever
# )

conv_chain = ConversationalRetrievalChain.from_llm(
    llm=chat_llm,
    # condense_question_prompt=prompt,
    retriever=retriever,
    memory=memory,
    verbose=True
)


query = "Halo, kamu dengan siapa? nama saya Ghifary"
print(f"Query: {query}")
chat_history = []
response = conv_chain.run(question=query)
print(response)

query = "Tolong jelaskan mengenai program MSIB (Magang dan Studi Independen Bersertifikat)."
print(f"Query: {query}")
response = conv_chain.run(question=query)
print(response)

query = "Apa nama program yang saya tanyakan sebelumnya?"
print(f"Query: {query}")
response = conv_chain.run(question=query)
print(response)


