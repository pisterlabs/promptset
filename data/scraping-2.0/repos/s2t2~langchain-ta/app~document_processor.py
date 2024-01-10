




import os
from functools import cached_property

from dotenv import load_dotenv
from pandas import DataFrame
import plotly.express as px

from langchain.document_loaders import NotebookLoader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter, PythonCodeTextSplitter, MarkdownTextSplitter #, MarkdownHeaderTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

from app.colors import CELL_COLORS_MAP
from app.text_splitter import split_text_by_substrings #parse_cell_type, TEXT_CELL_PREFIX, CODE_CELL_PREFIX
from app.cell import Cell
from app.document_formatting import documents_to_df


load_dotenv()

CHUNK_SIZE =  int(os.getenv("CHUNK_SIZE", default="1_000")) # 500
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", default="0")) # 40

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", default="0.75"))

FIG_SHOW = bool(os.getenv("FIG_SHOW", default="true") == "true")




class DocumentProcessor:
    """Processes .IPYNB notebook documents."""

    def __init__(self, filepath, chunk_overlap=CHUNK_OVERLAP, chunk_size=CHUNK_SIZE, verbose=True, similarity_threshold=SIMILARITY_THRESHOLD, file_id=None):
        """Param : filepath to the notebook document"""

        self.filepath = filepath
        self.filename = self.filepath.split("/")[-1] # might not work on windows?
        self.file_id = file_id or self.filename.split("_")[1] # assumes files are named like "Homework 4_USERNAME_more_stuff.ipynb" ... todo: regex instead, to be more precise

        self.chunk_overlap = int(chunk_overlap)
        self.chunk_size = int(chunk_size)

        self.embeddings_model_name = "text-embedding-ada-002"
        #self.faiss_index = self.filepath.upper().replace(".IPYNB", "") + "_FAISS_INDEX"
        self.similarity_threshold = float(similarity_threshold)

        self.verbose = bool(verbose)
        if self.verbose:
            print("---------------------")
            print("FILEPATH:", self.filepath)

    @cached_property
    def docs(self):
        loader = NotebookLoader(self.filepath, include_outputs=True, remove_newline=True)
        if self.verbose:
            print("LOADING...")

        docs = loader.load()
        #if self.verbose:
        #    print("DOCS:", len(docs))
        assert len(docs) == 1 # right? let's see if this is never the case
        return docs

    @cached_property
    def doc(self):
        return self.docs[0]

    # CELL SPLITTING (TEXT VS CODE):

    @cached_property
    def cells(self):
        cell_docs = []
        cell_texts = split_text_by_substrings(str(self.doc.page_content))
        for i, cell_text in enumerate(cell_texts):
            cell_metadata = { #"filepath": self.filepath,
                "file_id": self.file_id,
                "filename": self.filename,
                "cell_id": i+1,
                #"cell_type": parse_cell_type(cell_text),
                "cell_length": len(cell_text)
            }
            #chunk.metadata = {**chunk.metadata, **cell_metadata} # dict merge
            #doc = Document(page_content=cell_text, metadata=cell_metadata)
            doc = Cell(page_content=cell_text, metadata=cell_metadata)
            cell_docs.append(doc)
        return cell_docs

    @cached_property
    def text_cells(self):
        #return [cell for cell in self.cells if cell.metadata["cell_type"] == "TEXT"]
        return [cell for cell in self.cells if cell.is_text]

    @cached_property
    def code_cells(self):
        #return [cell for cell in self.cells if cell.metadata["cell_type"] == "CODE"]
        return [cell for cell in self.cells if cell.is_code]

    @cached_property
    def cells_df(self):
        return documents_to_df(self.cells)

    def plot_cell_lengths(self, fig_show=FIG_SHOW, height=500):
        title = f"Cell Lengths"
        #subtitle = f"Text Cells: {len(self.text_cells)} | Code Cells: {len(self.code_cells)}"
        subtitle = f"Document: {self.filename} | Text Cells: {len(self.text_cells)} | Code Cells: {len(self.code_cells)}"
        title += f"<br><sup>{subtitle}</sup>"

        fig = px.violin(self.cells_df, x="cell_length", facet_row="cell_type",
                        color="cell_type", color_discrete_map=CELL_COLORS_MAP,
                        title=title, height=height, points="all", box=True,
        )
        #fig.add_annotation(text= (f"Document: {self.filepath}"),
        #    font=dict(size=10, color="grey"), align="left", showarrow=False,
        #    x= 0,      xref='paper', xanchor='left',    xshift=-1,
        #    y= -0.15,  yref='paper', yanchor='bottom',  yshift=-5,
        #)

        if fig_show:
            fig.show()

    # CHUNKS (TEXT VS CODE):

    @cached_property
    def chunks(self):
        """preserves metadata about absolute chunk order within notebook"""
        chunks = []

        #chunk_idx = 0
        for cell_i, cell in enumerate(self.cells):
            cell_type = cell.metadata["cell_type"]
            if cell_type == "TEXT":
                splitter = MarkdownTextSplitter(chunk_size=self.chunk_size)
                #splitter = MarkdownTextSplitter(chunk_size=self.text_chunk_size)
            elif cell_type == "CODE":
                splitter = PythonCodeTextSplitter(chunk_size=self.chunk_size)
                #splitter = PythonCodeTextSplitter(chunk_size=self.code_chunk_size)
            else:
                splitter = CharacterTextSplitter(chunk_size=self.chunk_size)

            cell_chunks = splitter.split_documents([cell])
            for cell_chunk_i, cell_chunk in enumerate(cell_chunks):
                cell_chunk.metadata["chunk_id"] = cell_chunk_i+1
                cell_chunk.metadata["chunk_length"] = len(cell_chunk.page_content)
                chunks.append(cell_chunk)
                #chunk_idx+=1

        return chunks

    @cached_property
    def text_chunks(self):
        return [chunk for chunk in self.chunks if chunk.metadata["cell_type"] == "TEXT"]

    @cached_property
    def code_chunks(self):
        return [chunk for chunk in self.chunks if chunk.metadata["cell_type"] == "CODE"]

    @cached_property
    def chunks_df(self):
        return documents_to_df(self.chunks)

    def plot_chunk_lengths(self, fig_show=FIG_SHOW, height=500):
        title = f"Chunk Lengths ({self.chunk_size} chars max, {self.chunk_overlap} chars overlap)"
        #subtitle = f"Text Chunks: {len(self.text_chunks)} | Code Chunks: {len(self.code_chunks)}" + f" | Document: {self.filename}"
        subtitle=f"Document: {self.filename} | Text Chunks: {len(self.text_chunks)} | Code Chunks: {len(self.code_chunks)}"
        title += f"<br><sup>{subtitle}</sup>"
        #subtitle = f"Document: {self.filename}"
        #title += f"<br><sup>{subtitle}</sup>"

        fig = px.violin(self.chunks_df, x="chunk_length", facet_row="cell_type",
                        color="cell_type", color_discrete_map=CELL_COLORS_MAP,
                        title=title, height=height, points="all", box=True,
        )
        if fig_show:
            fig.show()

    # EMBEDDINGS / RELEVANCE FILTERING:

    @property
    def embeddings_model(self):
        model = OpenAIEmbeddings(model=self.embeddings_model_name)
        #if self.verbose:
        #    print(type(model))
        return model

    def make_retriever(self, cell_type="TEXT", storage_strategy="chunks"):
        docs_map = {
            "ALL": {
                "cells": self.cells,
                "chunks": self.chunks,
            },
            "TEXT": {
                "cells": self.text_cells,
                "chunks": self.text_chunks,
            },
            "CODE": {
                "cells": self.code_cells,
                "chunks": self.code_chunks,
            }
        }
        docs = docs_map[cell_type][storage_strategy]
        db = FAISS.from_documents(docs, self.embeddings_model)
        #faiss_index = self.filepath.upper().replace(".IPYNB", "") + "_FAISS_INDEX"
        #db.save_local(faiss_index)
        #db = FAISS.load_local(faiss_index, self.embeddings)
        retriever = db.as_retriever()
        print(cell_type, storage_strategy.upper(), "RETRIEVER:", type(retriever))
        return retriever

    @cached_property
    def retriever(self):
        return self.make_retriever(cell_type="ALL")

    @cached_property
    def code_retriever(self):
        return self.make_retriever(cell_type="CODE")

    @cached_property
    def text_retriever(self):
        return self.make_retriever(cell_type="TEXT")

    # COMPRESSION:

    def make_compression_retriever(self, base_retriever, similarity_threshold=0.75):
        compressor = EmbeddingsFilter(embeddings=self.embeddings_model, similarity_threshold=similarity_threshold)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
        #relevant_docs = compression_retriever.get_relevant_documents(STUDENT_QUERY)
        return compression_retriever

    @cached_property
    def text_compression_retriever(self):
        return self.make_compression_retriever(self.text_retriever, self.similarity_threshold)

    @cached_property
    def code_compression_retriever(self):
        return self.make_compression_retriever(self.code_retriever, self.similarity_threshold)
