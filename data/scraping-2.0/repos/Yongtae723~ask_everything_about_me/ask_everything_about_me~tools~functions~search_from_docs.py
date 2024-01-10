import os
import pickle
from glob import glob

from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import (HuggingFaceEmbeddings,
                                  HypotheticalDocumentEmbedder,
                                  OpenAIEmbeddings)
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from ask_everything_about_me.config import settings


class SearchFromDocs:
    def __init__(self):
        if not os.path.exists(settings.docsearch_path) or settings.reset_docsearch:
            if settings.base_embeddings == "openai":
                base_embeddings = OpenAIEmbeddings()
            else:
                base_embeddings = HuggingFaceEmbeddings(
                    model_name=settings.base_embeddings
                )

            # # HypotheticalDocumentEmbedderの準備
            embeddings = HypotheticalDocumentEmbedder.from_llm(
                llm=OpenAI(n=settings.HyDE_n, best_of=settings.HyDE_best_of),
                base_embeddings=base_embeddings,
                prompt_key="web_search",
            )

            def read_text(path):
                try:
                    with open(path) as f:
                        text = f.read()
                    return text
                except:
                    return None

            text_splitter = CharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                separator=".",
            )

            text_path_candidates = glob("ask_everything_about_me/data/*/*")
            texts = [
                read_text(text_path_candidate)
                for text_path_candidate in text_path_candidates
                if read_text(text_path_candidate)
            ]
            splitted_text = []
            for text in texts:
                splitted_text.extend(text_splitter.split_text(text))

            self.docsearch = FAISS.from_texts(splitted_text, embeddings)
            with open(settings.docsearch_path, "wb") as f:
                pickle.dump(self.docsearch, f)
        else:
            with open(settings.docsearch_path, "rb") as f:
                self.docsearch = pickle.load(f)
        self.chain = load_qa_chain(OpenAI(), chain_type="stuff")

    def __call__(self, question: str) -> str:
        docs = self.docsearch.similarity_search(question, k=3)
        res = self.chain(
            {
                "input_documents": docs,
                "question": question + ". Please answer in longer sentences.",
            },
            return_only_outputs=True,
        )

        return res["output_text"]

    def run(self, question: str) -> str:
        return self(question)
