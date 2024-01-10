from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv


def get_pdf_text(pdf_path):
    pdf = PdfReader(pdf_path)
    text = ''
    pages_dict = {}
    for i in range(len(pdf.pages)):
        page_text = pdf.pages[i].extract_text()
        text += page_text
        pages_dict[i + 1] = page_text
    return text, pages_dict


def get_text_chunks(text, chunk_size):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


class PDF_reader:
    load_dotenv()

    def __init__(self, pdf_path):
        self.raw_text, self.pages_dict = get_pdf_text(pdf_path)
        self.text_chunks = get_text_chunks(self.raw_text, chunk_size=3000)
        self.large_chunks = get_text_chunks(self.raw_text, chunk_size=14000)
        self.vectorstore = get_vectorstore(self.text_chunks)

    def get_relevant_chunks(self, query):
        text = ''
        page_no = 'none'
        chunks = self.vectorstore.similarity_search_with_score(query)
        for chunk in chunks:
            text += chunk[0].page_content

        first_chunk = chunks[0][0].page_content
        first_chunk_score = chunks[0][1]

        if first_chunk_score < 0.4:
            for key, value in self.pages_dict.items():
                if first_chunk in value:
                    page_no = key

        return first_chunk, page_no


