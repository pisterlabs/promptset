from langchain.text_splitter import MarkdownTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.document_loaders import DirectoryLoader
import os
import glob
import openai

class Knowledge_base():
    def __init__(self, knowledge_base, threshold=0.8):
        self.threshold=threshold
        self.knowledge_base = knowledge_base
    
    def search_from_knowledge_base(self, question):
        # Return the closet doc to question
        docs = self.knowledge_base.similarity_search_with_score(question)
        closest_doc = self._get_closet_doc_from_docs(docs)
        if closest_doc:
            return [closest_doc]
        else:
            return None
    
    # Only return closest doc
    def _get_closet_doc_from_docs(self, docs):
        # Return the doc having the min score below threshold
        min_score = self.threshold
        min_id = -1
        for id, item in enumerate(docs):
            doc, score = item
            if min_score > score:
                min_id = id
                min_score = score
        if min_score < self.threshold:
            return docs[min_id][0]  # return doc
        else:
            return None
        
class Text_processor():
    def __init__(self, folder_path, chunk_size=256, chunk_overlap=20):
        self.folder_path = folder_path
        self.folder_list = self._get_rescursive_folder()
        self.doc_list = self._get_doc_path()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings=OpenAIEmbeddings()
        self.data = None
        self.knowledge_base=None
    
    def _get_rescursive_folder(self):
        folder_list = []
        for folder in glob.iglob(f"{self.folder_path}/**"):
            folder_list.append(folder)
        return folder_list
    
    def _get_doc_path(self):
        doc_list = []
        for filename in glob.iglob(f'{self.folder_path}/**/*.md', recursive=True):
            doc_list.append(filename)
        return doc_list
    
    def _load_docs(self):
        loader = DirectoryLoader(self.folder_path)
        data = loader.load()
        return data
        
    def _split_docs(self):
        # Split the text into chunks using Langchain's CharacterTextSplitter
        text_splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        upf_splits = text_splitter.split_documents(self.data)
        return upf_splits
    
    def embed_docs(self):
        # Load docs
        self.data = self._load_docs()
        
        # Split docs
        self.upf_splits = self._split_docs()
        
        # Filter chunks shorter than 1 sentence or 10 words
        self.upf_splits = self._filter_chunk()
        
        # Embed chunks
        self.knowledge_base = Knowledge_base(FAISS.from_documents(documents=self.upf_splits, embedding=self.embeddings))
        return self.knowledge_base

    def _filter_chunk(self):
        filtered_upf_splits = []
        for chunk in self.upf_splits:
            if len(chunk.page_content.split()) > 5:
                filtered_upf_splits.append(chunk)
        return filtered_upf_splits
    
    def _get_doc_len(self, text, threshold_char=10):
        # Check if the split is longer than 1 sentences
        return len(text.split()) >= threshold_char

    def get_n_doc(self):
        return len(self.doc_list)

    def save_knowledge_base(self, output_path):
        self.knowledge_base.knowledge_base.save_local(output_path)
    
    def load_knowledge_base(self, input_path):
        self.knowledge_base = Knowledge_base(FAISS.load_local(input_path, self.embeddings))
        return self.knowledge_base