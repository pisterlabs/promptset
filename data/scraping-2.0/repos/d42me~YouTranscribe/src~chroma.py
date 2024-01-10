from enum import Enum

from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from constants import *

llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.9,
)

class EmbeddingsType(Enum):
    OPENSOURCE = "OPENSOURCE"
    OPENAI = "OPENAI"

EMBEDDDING_FUNCTION = EmbeddingsType.OPENAI

if EMBEDDDING_FUNCTION == EmbeddingsType.OPENSOURCE:
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
elif EMBEDDDING_FUNCTION == EmbeddingsType.OPENAI:
    embedding_function = OpenAIEmbeddings()

DEFAULT_CONTEXT_PROMPT = """This is a conversation transcript between George Hotz and Eliezer Yudkowsky. Use the following pieces of context to answer the question at the end.

{context}

Question: {question}
"""

"""
Vectorize the paper in Chroma vector store using the given embedding function
"""

def vectorize_transcript_in_chroma(input: str):
    # split it into sentences
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap  = 0
    )
    chunks = text_splitter.split_text(input)

    docs = []
    for speaker_section in chunks:
        print(speaker_section)
        doc = Document(
            page_content=speaker_section,
        )

        docs.append(doc)

    # load it into Chroma
    db = Chroma.from_documents(
        docs, embedding_function, persist_directory="../chroma_db"
    )
    db.persist()


def query_chroma(input: str, number_of_results=10):
    # Load chroma
    db = Chroma(persist_directory="../chroma_db", embedding_function=embedding_function)

    docs = db.similarity_search(input, k=number_of_results)

    return docs


def query_chroma_by_prompt(question: str):
    # Load chroma
    docs = query_chroma(question)

    # Query your database here
    chain = load_qa_chain(llm, chain_type="stuff")

    return chain.run(input_documents=docs, question=question)


def query_chroma_by_prompt_with_template(
    question: str, prompt_template: str = DEFAULT_CONTEXT_PROMPT
):
    # Load chroma
    docs = query_chroma(question, 2)

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT, verbose=True)

    result = chain(
        {"input_documents": docs, "question": question}, return_only_outputs=True
    )
    return result["output_text"]


if __name__ == "__main__":
    with open("transcript-filtered.txt", "r") as f:
        input = f.read()
        vectorize_transcript_in_chroma(input)

    print(query_chroma_by_prompt("Tell me 10 facts I need to take away from this debate."))
