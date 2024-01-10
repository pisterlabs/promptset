from typing import List
from langchain.text_splitter import TextSplitter
from langchain.schema import Document
import spacy



from langchain.text_splitter import TextSplitter
from langchain.schema import Document
import spacy


class MyCustomTextSplitter(TextSplitter):
    def __init__(self, chunk_size, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.nlp = spacy.load('en_core_web_lg')

    def find_eos_spacy(self, text):
        doc = self.nlp(text)
        return [sent.end_char for sent in doc.sents]

    def split_text(self, text: str) -> List[Document]:
        chunks_and_metadata = self.text_to_chunks(text, self.chunk_size)
        documents = [Document(page_content=chunk, metadata=metadata) for chunk, metadata in chunks_and_metadata]
        return documents

    def text_to_chunks(self, text, size):
        if size and len(text) > size:
            out = []
            pos = 0
            eos = self.find_eos_spacy(text)
            if len(text) not in eos:
                eos += [len(text)]
            for i in range(len(eos)):
                if eos[i] - pos > size:
                    text_chunk = text[pos:eos[i]]
                    metadata = {'length': len(text_chunk), 'start_pos': pos, 'end_pos': eos[i]}
                    out.append((text_chunk, metadata))
                    pos = eos[i]
            # ugly: last chunk
            text_chunk = text[pos:eos[i]]
            metadata = {'length': len(text_chunk), 'start_pos': pos, 'end_pos': eos[i]}
            out.append((text_chunk, metadata))
            out = [(chunk, metadata) for chunk, metadata in out if chunk]
            return out
        else:
            return [(text, {'length': len(text), 'start_pos': 0, 'end_pos': len(text)})]



















class CustomTextSplitter(TextSplitter):
    def __init__(self, chunk_size, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.nlp = spacy.load('en_core_web_lg')

    def find_eos_spacy(self, text):
        doc = self.nlp(text)
        return [sent.end_char for sent in doc.sents]

    def split_text(self, text: str) -> List[Document]:
        chunks = self.text_to_chunks(text, self.chunk_size)
        documents = [Document(page_content=chunk, metadata={'length': len(chunk)}) for chunk in chunks]
        return documents

    def text_to_chunks(self, text, size):
        if size and len(text) > size:
            out = []
            pos = 0
            eos = self.find_eos_spacy(text)
            if len(text) not in eos:
                eos += [len(text)]
            for i in range(len(eos)):
                if eos[i] - pos > size:
                    text_chunk = text[pos:eos[i]]
                    out += [text_chunk]
                    pos = eos[i]
            # ugly: last chunk
            text_chunk = text[pos:eos[i]]
            out += [text_chunk]
            out = [x for x in out if x]
            return out
        else:
            return [text]
