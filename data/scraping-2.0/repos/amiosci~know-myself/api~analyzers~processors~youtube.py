import os

from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate

# only relevant while running script for testing purposes
if __name__ == "__main__":
    import sys

    api_root = os.path.abspath(os.path.join(__file__, "../../.."))
    sys.path.append(api_root)

from analyzers import utils
from doc_store import doc_loader


def interrogate_youtube_video(url: str, questions: list[str]):
    docs = doc_loader.get_url_documents(url)

    combined_docs = [doc.page_content for doc in docs]
    text = " ".join(combined_docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_text(text)

    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(
        splits,
        embedding_function,
    )
    prompt = PromptTemplate(
        template="""Given the context about a video. Answer the user in a friendly and precise manner.
    
    Context: {context}
    
    Human: {question}
    
    AI:""",
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=utils.chat_llm(),
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    responses = []
    for question in questions:
        response = qa_chain({"query": question})
        responses.append((question, response.get("result", None)))

    return responses


if __name__ == "__main__":
    for question, response in interrogate_youtube_video(
        "https://youtube.com/shorts/IicbiwTAslE?si=H1qA7---M4ZiuHTc",
        ["What is this video about?"],
    ):
        print(f"{question}")
        print(f"\t{response}")
