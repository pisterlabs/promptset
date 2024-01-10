from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

import numpy as np
import os
import pandas as pd
import re

embeddings=HuggingFaceEmbeddings(model_name='hiiamsid/sentence_similarity_spanish_es')

class PromptGenerator:
    def __init__(self, knowledge_dir = 'documents', k = 3, chunk_length = 600):
        self.knowledge_dir = knowledge_dir  # Path to the knowledge base
        self.k = k  # The number of most related paragraphs to be included in the prompt
        self.chunk_length = chunk_length  # Length of each chunk of text
        self._read_paragraphs()
        print('Loading CrossEncoder...')
        self.cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

    # Prompt generation
    def _generate_prmopt(self, docs, question):
        return str.join('\n', docs)

    def _read_paragraphs(self):
        print('Loading embeddings...')
        embeddings=HuggingFaceEmbeddings(model_name='hiiamsid/sentence_similarity_spanish_es')
        print(f'Loading txt documents from {self.knowledge_dir}')
        loader =DirectoryLoader(self.knowledge_dir, show_progress=True, use_multithreading=True, glob='*.txt')
        data = loader.load()
        print('Splitting documents...')
        documents = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_length, chunk_overlap=self.chunk_length//5, separators=["\n\n", "\n", " ", ""])
        for d in data:
            print(f'Processing {d.metadata["source"]}...')
            d = d.page_content
            paragraphs = re.split('\n\n', d)
            documents += paragraphs
            #for p in paragraphs:
                #documents += splitter.split_text(p)
        self.n_documents = len(documents)
        print(f'Found {self.n_documents} txt documents')
        print('Creating vector store...')
        self.docsearch = Chroma.from_texts(documents, embeddings)

    def get_prompt(self, question):
        docs=[doc.page_content for doc in self.docsearch.similarity_search(question, k=np.min([10, self.n_documents]))]
        scores = self.cross_encoder.predict([[question, doc] for doc in docs])
        filter=scores>0.3
        scores = np.array(scores)[filter]
        docs = np.array(docs)[filter]
        top_k_indices = scores.argsort()[::-1][:self.k]
        context_docs = [docs[i] for i in top_k_indices]
        if len(context_docs) == 0:
            prompt = ''
        else:
            prompt = self._generate_prmopt(context_docs, question)
        return prompt


if __name__ == '__main__':
    promptGenerator = PromptGenerator()
    prompt = promptGenerator.get_prompt("¿Qué clase tiene la especialidad MARMT de 1º EMIES el jueves a las 8?")
    print(prompt)
