# https://www.manceps.com/articles/tutorial/how-to-extract-knowledge-from-documents-with-google-palm-2-llm
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
# from langchain.embeddings import GooglePalmEmbeddings # If no GPU/time.
from InstructorEmbedding import INSTRUCTOR
embeddings_model_name = INSTRUCTOR("hkunlp/instructor-xl")
embeddings = HuggingFaceInstructEmbeddings(model_name=embeddings_model_name,model_kwargs={"device": "cuda"})
# Cloud based embedding alternative:
# embeddings = GooglePalmEmbeddings()
# ---chroma---
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader
# ------------

pdf = "pdf/Development_of_an_Embedded_Device_for_Stroke_Prediction_via_Artificial_Intelligence-Based_Algorithms.pdf"
if pdf:
    # Extract text from each page
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
        )
    chunks = text_splitter.split_text(text=text)

    # Embed chunks and store them in a Vector Store
    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

    query = "name the author(s)"

    if query:
        docs = VectorStore.similarity_search(query=query, k=3)

        llm = GooglePalm()
        llm.temperature = 0.1 # Increase for more creativity [0.0-1.0]
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)

        print(response)

