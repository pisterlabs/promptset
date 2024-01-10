import os

from time import sleep
from math import ceil
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import CharacterTextSplitter


class RelevancyTagger:
    def __init__(self, llm, model_name, threshold, chunk_size):
        self.llm = llm
        self.model_name = model_name
        self.total_tokens = 0
        self.threshold = threshold
        self.chunk_size = chunk_size


    def split_text(self, text):
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, chunk_overlap=0, separator='\n'
        )

        texts = text_splitter.split_text(text)
        print(f'Document was split into {len(texts)} chunks')
        return texts


    def prompt_llm(self, texts, document_title=None):
        sys_message = SystemMessage(content=(
            "You are a knowledgeable assistant that takes in a chunk of a document and outputs a score from 0-100. "
            "You should only output the numerical score and nothing else. "
            f"For context, the document's title is {document_title}" if document_title else ""
        ))

        results = []
        for i, chunk in enumerate(texts):
            with get_openai_callback() as cb:
                print(f'Processing chunk {i}')
                results.append(self.llm([sys_message, HumanMessage(content=chunk)]))
                self.total_tokens += cb.total_tokens
                # sleep(0.05)  # Rate limits

        print(results)
        return results


    def extract_relevant_chunks(self, texts, results):
        scores = [(int(msg.content) if msg.content.isnumeric() else -1, idx) for idx, msg in enumerate(results)] # If response isn't a number, set score to -1
        scores.sort(reverse=True, key=lambda x: (x[0], -x[1]))  # If scores are tied, prefer earlier chunk
        num_chunks = ceil(len(scores) * self.threshold)
        print(f"Using a threshold of {self.threshold}, {num_chunks} chunks were selected out of {len(scores)}")

        scores = scores[:num_chunks]
        scores.sort(key=lambda x: x[1]) # Resort relevant chunks back into order
        relevant_chunks = [texts[idx] for _, idx in scores]
        print(f"Top {num_chunks} highest scores: {' '.join(str(score) for score, idx in scores)}")
        print(f"Tokens used: {self.total_tokens}")
        return relevant_chunks


    def tag(self, text, document_title=None):
        texts = self.split_text(text)
        results = self.prompt_llm(texts, document_title=document_title)
        relevant_chunks = self.extract_relevant_chunks(texts, results)
        return '\n'.join(relevant_chunks)


if __name__ == '__main__':
    CHUNK_SIZE = 1000
    THRESHOLD = 0.5
    text = open("../Gatsby.txt", encoding='utf-8').read()
    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(temperature=0, openai_api_key="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",
                    model_name=model_name)
    tagger = RelevancyTagger(llm, model_name, THRESHOLD, CHUNK_SIZE)
    print(tagger.tag(text))
