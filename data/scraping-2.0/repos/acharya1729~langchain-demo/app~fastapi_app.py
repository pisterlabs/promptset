# app/fastapi_app.py
from fastapi import FastAPI, UploadFile, HTTPException
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores.chroma import Chroma

from app.utils import save_uploaded_files, load_questions, load_document
from app.langchain_init import text_splitter, openai_embeddings, llm, rag_prompt, format_docs

app = FastAPI()


def process_question(question, retriever):
    q = "".join(question.values())
    rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | rag_prompt | llm | StrOutputParser())
    answer = rag_chain.invoke(q)

    return {"question": q, "answer": answer}


@app.post("/qa/")
async def question_answering(document_file: UploadFile, questions_file: UploadFile):
    try:
        # Save uploaded files
        document_file_path, questions_file_path = save_uploaded_files(document_file, questions_file)

        # Load the document
        document = load_document(document_file_path)

        # Load questions from JSON file
        questions_data = load_questions(questions_file_path)

        # Split the document
        splits = text_splitter.split_documents(document)

        # Embed and store splits
        vectorstore = Chroma(embedding_function=openai_embeddings)
        vectorstore.add_documents(splits, embedding=openai_embeddings)
        retriever = vectorstore.as_retriever()
        answers = []

        # Process each question
        for question in questions_data['questions']:
            answer = process_question(question, retriever)
            answers.append(answer)

        return answers

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
