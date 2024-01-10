import os
import json
import numpy as np
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain import LangChain, TextPreprocessor, TextPostprocessor
from pathlib import Path
from typing import List, Dict, Union, Generator
from ..config import Config
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import KDTree
import pinecone

class OpenAIModel:
    def __init__(self):
        self.OPENAI_API_KEY = Config.OPENAI_API_KEY
        self.PINECONE_API_KEY = Config.PINECONE_API_KEY
        self.client: OpenAI = OpenAI(api_key=self.OPENAI_API_KEY)
        self.CURRENT_DIR = Path(__file__).resolve().parent
        self.RESOURCE_PATH = self.CURRENT_DIR / ".." / "resources"
        self.preprocessor = TextPreprocessor()
        self.postprocessor = TextPostprocessor()
        self.langchain = LangChain(preprocessor=self.preprocessor, postprocessor=self.postprocessor)
        self.vectorizer = TfidfVectorizer()
        self.index_name = "openai-model-index"
        self.messages: List[Dict[str, str]] = []
        self.init_pinecone()

    def init_pinecone(self):
        pinecone.deinit()
        pinecone.init(api_key=PINECONE_API_KEY)
        pinecone.create_index(index_name=self.index_name, metric="cosine", shards=1)
        self.index = pinecone.Index(index_name=self.index_name)

    def read_pdf(self) -> str:
        with open(f"{self.RESOURCE_PATH}/cs4349_chida.pdf", "rb") as f:
            reader: PdfReader = PdfReader(f)
            text: str = "".join(page.extract_text() for page in reader.pages)
        return text

    def initialize(self):
        self.messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": """You are a useful teacher's assistant for a class at the University of Texas at Dallas.
            You should not say you're an AI, instead your name is Alice.
            A student can either ask a question about the coursework, or the syllabus. If it's the syllabus, directly cite the part(s) of the syllabus that pertains to the question.
            Wrap all LaTeX equations or output in $, for example $y = mx + b$
            Below is the syllabus. Make sure every syllabus question is answered completely accurately, and if you cannot do so say you do not know the answer.
            """,
            },
            {"role": "system", "content": self.read_pdf()},
        ]

    def ask_question_single(self, prompt: str) -> str:
        try:
            self.messages.append({"role": "user", "content": prompt})
            response: dict = self.client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=self.messages,
            )
            response_content: str = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": response_content})
            return response_content
        except Exception as e:
            return str(e)

    def ask_question_stream(self, prompt: str):
        try:
            self.messages.append({"role": "user", "content": prompt})
            response: dict = self.client.chat.completions.create(
                model="gpt-4-1106-preview", messages=self.messages, stream=True
            )
            response_text: str = ""
            yield json.dumps({"type": "start", "content": True})
            for chunk in response:
                content: str = chunk.choices[0].delta.content or ""
                if content is None:
                    continue
                response_text += content

                yield json.dumps({"type": "partial", "content": response_text})

            self.messages.append({"role": "assistant", "content": response_text})
            yield json.dumps({"type": "full", "content": response_text})
            print("Answer: ", response_text)
        except Exception as e:
            print(e)

    def vectorize_text(self, text: str) -> np.ndarray:
        return self.vectorizer.transform([text]).toarray()

    def find_similar(self, vector: np.ndarray) -> List[int]:
        return self.index.query(queries=[vector], top_k=5)
