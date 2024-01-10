from langchain.document_loaders import PDFPlumberLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import tempfile
import os
import re


class FileProcessor:

    def __init__(self, file):

        self.file = file
        self.preview_text, self.cleaned_docs = self.get_preview_text_and_cleaned_docs(file)
        self.text_chunks = self.get_chunks_for_embedding(self.preview_text, self.cleaned_docs)
        self.text_length = 0
        self.nr_tokens, self.price = self.get_nr_of_tokens_and_price(self.text_chunks)


    # PDFPlumberLoader.load() returns list with page content and metadata for each page in PDF
    def load_data(self, file_path):
        loader = PDFPlumberLoader(file_path)
        data = loader.load()
        return data


    def save_as_temp_file_and_get_data(self, file_obj):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_path = temp_file.name

            # Write the contents of the file-type object to the temporary file
            temp_file.write(file_obj.read())
            data = self.load_data(temp_path)

            temp_file.close()
            os.remove(temp_path)

        return data


    # consider removing some of the cleaning code (may interfere with recursive splitting)
    def clean_docs(self, data):
        cleaned_docs = []
        for doc in data:
            text = doc.page_content
            clean_content = re.sub(r"\n{2,}", '###!!!###', text).replace('-\n', '').replace('\n', ' ').replace('###!!!###', '\n\n')
            cleaned_doc = Document(page_content=clean_content, metadata=doc.metadata)
            cleaned_docs.append(cleaned_doc)
        return cleaned_docs


    def get_full_clean_text(self, clean_docs):
        full_clean_text = ''
        for doc in clean_docs:
            full_clean_text += doc.page_content
        return full_clean_text


    def get_preview_text_and_cleaned_docs(self, file_obj):
        data = self.save_as_temp_file_and_get_data(file_obj)
        cleaned_docs = self.clean_docs(data)
        full_clean_text = self.get_full_clean_text(cleaned_docs)
        self.text_length = len(full_clean_text)
        return full_clean_text, cleaned_docs


    def get_text_chunks(self, full_text, chunk_size=1000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "(?<=\. )", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        text_chunks = text_splitter.create_documents([full_text])
        return text_chunks


    def combine_chunks_with_metadata(self, clean_docs, chunks):
        final_chunks = []
        for chunk in chunks:
            final_chunk_content = chunk.page_content

            # get first line of chunk content to be matched with clean docs
            match_content = final_chunk_content[:120]

            # loop through clean docs and find match
            for doc in clean_docs:
                if match_content in doc.page_content:
                    final_chunk_metadata = doc.metadata
                    final_chunk = Document(page_content=final_chunk_content, metadata=final_chunk_metadata)
                    final_chunks.append(final_chunk)
                else:
                    pass

        return final_chunks


    def get_nr_of_tokens_and_price(self, chunks, token_price=0.0001):
        '''takes as arguments chunks created via previous function as well as price which can be researched on OpenAI website
        (https://openai.com/pricing)'''

        nr_tokens = 0

        for chunk in chunks:
            enc = tiktoken.get_encoding("p50k_base")
            chunk_tokens = enc.encode(chunk.page_content)
            nr_tokens += len(chunk_tokens)

        price = round((nr_tokens / 1000) * token_price, 4)
        return nr_tokens, price


    def get_chunks_for_embedding(self, clean_text, clean_docs):
        text_chunks = self.get_text_chunks(clean_text)
        text_chunks_with_metadata = self.combine_chunks_with_metadata(clean_docs, text_chunks)
        return text_chunks_with_metadata


# fp = FileProcessor()
# doc1 = Document(page_content="this is a test string. Hallelujah!", metadata={"test_info": "a"})
# doc2 = Document(page_content="this is another, somewhat longer test string", metadata={"test_info": "b"})
# text_chunk1 = Document(page_content="this is a test string.")
# text_chunk2 = Document(page_content="somewhat longer test string")
# chunks_with_metadata = fp.combine_chunks_with_metadata([doc1, doc2], [text_chunk1, text_chunk2])
# print(chunks_with_metadata[0])
# print(chunks_with_metadata[1])