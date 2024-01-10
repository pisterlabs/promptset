import math
from typing import List, Tuple
from PIL import Image

from tqdm import tqdm
import numpy as np
import pypdfium2 as pdfium
import os
import pytesseract
import openai
from embedding import BaseEmbedding
from vectordb import IndexDocument, TopChunks


# os.environ['OPENAI_API_KEY'] = "" #   Your API Key
# openai.api_key = os.environ['OPENAI_API_KEY']

class DocInput:
    """
    Takes PDF/Image input; extracts text info from it, does chunking and sends forward
    """

    def __init__(self, documents: str, chunk_size: int = 128, chunk_overlap: int = 10):
        extension = file.split(".")[-1]
        self.text = ""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if extension in ["jpg", "jpeg", "png", "tiff"]:
            self.text = pytesseract.image_to_string(Image.open(file))
        elif extension == "pdf":
            pdf = pdfium.PdfDocument(file)
            for i in tqdm(range(len(pdf))):
                page = pdf[i]
                pil_image = page.render(scale=1).to_pil()
                self.text += pytesseract.image_to_string(pil_image)
        elif extension in ["doc", "docx"]:
            import textract

            self.text = textract.process(file).decode()
        elif extension == "txt":
            self.text = open(file, "r").read()
        elif extension == "rtf":
            from striprtf.striprtf import rtf_to_text

            self.text = rtf_to_text(open(file, "r").read())
        elif extension == "odt":
            from odf import text, teletype
            from odf.opendocument import load

            textdoc = load(file)
            allparas = textdoc.getElementsByType(text.P)
            self.text = " ".join(
                [teletype.extractText(allparas[i]) for i in range(len(allparas))]
            )
        else:
            raise Exception("Doesnot support this format", extension)
        self.text = self.text.replace("\n", " ")

    # def get_ocr(self,ocr_engine: str="tesseract") -> str:
    #     return ocr_output

    def get_chunks(self):
        self.doc_chunks = []
        words = self.text.split(" ")
        words = [word for word in words if word not in [".", "?", "!", ",", ":", ";"]]
        self.doc_chunks = [
            " ".join(
                words[
                    max(0, i * self.chunk_size - self.chunk_overlap) : min(
                        len(words), (i + 1) * self.chunk_size
                    )
                ]
            )
            for i in range(math.ceil(len(words) / self.chunk_size))
        ]

    def preprocess_chunks(self):
        # Preprocessing; nothing added for now; operate on self.doc_chunks
        pass

    def get_doc_input(self) -> List[str]:
        self.get_chunks()
        self.preprocess_chunks()

        return self.doc_chunks


class DocQA:
    """
    Creates a DocQA instance with all document indexed and processed. Receives the query and answers it
    """

    def __init__(self, documents: str):
        self.documents = documents
        self.embedding_obj = BaseEmbedding()
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.post_init()

    def post_init(self):
        "Document indexing"
        doc_input = DocInput(self.documents)
        self.chunks = doc_input.get_doc_input()

        index_document = IndexDocument(self.chunks, self.embedding_obj)
        self.index, self.index_matrix = index_document.indexed_document()

        self.top_chunks = TopChunks(
            indexes=self.index,
            index_matrix=self.index_matrix,
            embedding_obj=self.embedding_obj,
        )
        self.context = [
            {"role": "system", "content": "You are a question answering bot"},
        ]

    def answer_query(self, query: str) -> str:
        answer_chunks = self.top_chunks.top_k(query)

        pointer_prompt = "\n ".join(
            [str(i + 1) + ". " + chunk for i, chunk in enumerate(answer_chunks)]
        )
        # Base prompt based on vector retrieval
        base_prompt = (
            "These are the relevant chunks from the document:\n" + pointer_prompt
        )
        # Adding the initial query to the base prompt
        init_prompt = (
            base_prompt
            + "\nAnswer this query based on the above information:\n"
            + query
        )
        # Adding preventive prompt
        final_prompt = (
            init_prompt
            + "\nGive your answer based only on the above information, do not use any other information"
        )

        if query == "reset":
            self.context = [
                {"role": "system", "content": "You are a question answering bot"},
            ]
            return {
                "choices": [
                    {
                        "message": {
                            "content": "The chat session is reset now, proceed with a fresh slate now\n-----------------x------------------x-------------------"
                        }
                    }
                ]
            }
        self.context.append({"role": "user", "content": final_prompt})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.context,
            temperature=0.0,
        )
        self.context.append(
            {
                "role": "assistant",
                "content": response["choices"][0]["message"]["content"],
            }
        )

        return response


if __name__ == "__main__":
    # test for images
    doc_qa = DocQA("tests/test.jpg")
    response = doc_qa.answer_query(
        "What are the things talked about luxury in this document?"
    )
    print(response)

    # test for pdf
    doc_qa = DocQA("tests/test.pdf")
    response = doc_qa.answer_query(
        "What are the things talked about luxury in this document?"
    )
    print(response)

    # test for doc
    doc_qa = DocQA("tests/test.docx")
    response = doc_qa.answer_query(
        "What are the things talked about luxury in this document?"
    )
    print(response)

    # test for txt
    doc_qa = DocQA("tests/test.txt")
    response = doc_qa.answer_query(
        "What are the things talked about luxury in this document?"
    )
    print(response)

    # test for rtf
    doc_qa = DocQA("tests/test.rtf")
    response = doc_qa.answer_query(
        "What are the things talked about luxury in this document?"
    )
    print(response)

    # test for odt
    doc_qa = DocQA("tests/test.odt")
    response = doc_qa.answer_query(
        "What are the things talked about luxury in this document?"
    )
    print(response)
