import os 
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub, Replicate
from PyPDF2 import PdfReader
from tqdm import tqdm

from langchain.chains import LLMChain
import docx
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from dotenv import load_dotenv
from pathlib import Path
from random import sample
import pandas as pd

load_dotenv()


def get_pdf_text_chunks(file_path):
    """Extract text from a PDF file and return it in chunks of 500 characters."""
    chunks = []
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:
                    for i in range(0, len(text), chunk_size):
                        chunk = text[i:i+chunk_size] + f" - Page {page_num + 1}, {os.path.basename(file_path)}"
                        chunks.append(chunk)
            print(f"{os.path.basename(file_path)}: {len(chunks)} chunks")
    except Exception as e:
        print(f"{os.path.basename(file_path)}: can't chunk this file - corrupt or unsupported format")

    return chunks


def load_files_and_chunk_text(directory):
    print("Chunking..")
    """Load all files from a directory, extract text, and return it in chunks of 500 characters."""
    all_chunks = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if file_name.lower().endswith('.pdf'):
            all_chunks.extend(get_pdf_text_chunks(file_path))
        if all_chunks.__len__() > 500:
            break
    return all_chunks
chunk_size = 1000
splits = load_files_and_chunk_text("document")


prompt_template = """
            Create exactly {num_questions} questions using the context and make 
            sure each question doesn’t reference
            terms like "this study", "this research", or anything that’s 
            not available to the reader.
            End each question with a ‘?’ character and then in a newline
            write the answer to that question using only 
            the context provided.

            Separate each question/answer pair by "XXX"

            Each question must start with "question:".

            Each answer must start with "answer:".

            CONTEXT = {context}
        """
prompt = PromptTemplate(template=prompt_template, input_variables=['num_questions', 'context'])  # Creating a prompt template object


embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

# Load the documents

llm = Replicate(
                model="meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d",
                model_kwargs={"temperature": 0.1, "max_length": 1000, "top_p": 0.95},
            )

chain = LLMChain(llm=llm, prompt=prompt)

SYNTHETIC_DATASET_SIZE = 70
NUM_QUESTIONS_TO_GENERATE = 3
sampled_chunks = sample(splits, k=SYNTHETIC_DATASET_SIZE)

synthetic_dataset = []
for sampled_chunk in tqdm(sampled_chunks):
    prediction = chain.invoke(
        {
            "num_questions": NUM_QUESTIONS_TO_GENERATE,
            "context": sampled_chunk,
        }
    )
    output = prediction["text"]
    print(output)

