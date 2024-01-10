# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creation Date: July 10, 2023

@author: Aaron Wilkowitz
"""

################
### import 
################

# GCP   
from google.api_core.client_options import ClientOptions
from google.auth import default
from google.auth import impersonated_credentials
from google.cloud import documentai, storage
import vertexai

import utils_config

# Langchain 
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.embeddings.base import Embeddings
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import BaseLLM, VertexAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import MatchingEngine
from pypdf import PdfReader, PdfWriter
from tqdm import tqdm
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type
from urllib.parse import urlparse

# Others
import streamlit as st
import streamlit.components.v1 as components
from streamlit_player import st_player
import concurrent.futures
import datetime
import IPython
import numpy as np
import os
import pandas as pd
import PIL, PIL.ImageDraw
import requests
import scann
import shapely
import time
import tempfile
import threading
from io import BytesIO

################
### page intro
################

# Make page wide
st.set_page_config(
    page_title="GCP GenAI",
    layout="wide",
  )

# Title
st.title('GCP HCLS GenAI Demo: Q&A Chatbot on a Large Document')

# Author & Date
st.write('**Author**: Aaron Wilkowitz, aaronwilkowitz@google.com')
st.write('**Original code from**: Ayo Adedeji, ayoad@google.com')
# Original file is located at https://colab.research.google.com/drive/1YUae5R6GwzQbFG__LJRXfrR7vUUTCIxk?resourcekey=0-MZCdofdacgZJKfuNHTimug

st.write('**Date**: 2023-08-16')
st.write('**Purpose**: Use DocAI + Generative AI + ScaNN to answer a question against a large document')

# Gitlink
st.write('**Github repo**: https://github.com/aaronlutkowitz2/genai_app_hcls_general')

# Video
st.divider()
st.header('30 Second Video')

# video_url = 'https://youtu.be/AtVCwywl_q8'
# st_player(video_url)

# Architecture

st.divider()
st.header('Architecture')

# components.iframe("https://docs.google.com/presentation/d/e/2PACX-1vSmJtHbzCDJsxgrrrcWHkFOtC5PkqKGBwaDmygKiinn0ljyXQ0Xaxzg4mBp2mhLzYaXuSzs_2UowVwe/embed?start=false&loop=false&delayms=3000000",height=800) # width=960,height=569

################
### setup / initialize 
################

# Note: During Q & A, signed URLs will be served that reference the relevant matches used for summarization by the LLM. 
# As a result, a service account will be used that signs the URLs (this step cannot be performed using user credentials).

USER_EMAIL = "aaronwilkowitz@google.com"  # @param {type:"string"}
PROJECT_ID = utils_config.get_env_project_id() # @param {type:"string"}
LOCATION = utils_config.LOCATION # @param {type:"string"}
SIGNING_SERVICE_ACCOUNT = utils_config.SIGNING_SERVICE_ACCOUNT  # @param {type:"string"}


vertexai.init(project=PROJECT_ID, location=LOCATION)

################
### set classes
################

class SourceDocument:
    """Document Metadata for DocAILoader"""

    def __init__(self, content: Optional[bytes], mime_type: str, source: str, page: Optional[int] = None, **kwargs):
        """Initialize with content, mime type, source, and optional page"""
        super().__init__(**kwargs)
        self.content = content
        self.mime_type = mime_type
        self.source = source
        self.page = page
        self.document = None

    def set_document(self, document: documentai.Document) -> None:
        self.content = None
        self.document = document

class DocAILoader(BaseLoader):
    """Loading logic for loading documents from GCS."""

    def __init__(
        self,
        sources: List[str],
        project_id: str,
        location: str = "us",
        processor_id: str = None,
        default_processor_display_name: str = "doc-search-form-parser",
        default_processor_type: str = "FORM_PARSER_PROCESSOR",
        create_processor_if_not_exists: bool = True,
        max_doc_ai_requests_per_min: int = 96,
        max_parallel_doc_ai_requests: int = 8,
        verbose: bool = True
    ) -> None:
      """
      Initialize

      Args:
        sources (List[str]): List of source documents
        project_id (str): Project ID
        location (str): Location of the project
        processor_id (str): Processor ID
        default_processor_display_name (str): Default processor display name
        default_processor_type (str): Default processor type
        create_processor_if_not_exists (bool): Create processor if it does not exist
        max_doc_ai_requests_per_min (int): Max parallel document AI requests
        max_parallel_doc_ai_requests (int): Max parallel document AI requests
        verbose (bool): Verbose
      Returns:
        None
      """
      self.sources = sources
      self.project_id = project_id
      self.location = location
      self.processor_id = processor_id
      self.default_processor_display_name = default_processor_display_name
      self.default_processor_type = default_processor_type
      self.create_processor_if_not_exists = create_processor_if_not_exists
      self.max_doc_ai_requests_per_min = max_doc_ai_requests_per_min
      self.max_parallel_doc_ai_requests = max_parallel_doc_ai_requests
      self.verbose = verbose
      self.processor_name = None
      self.doc_ai_api_calls = np.array([])
      self.active_calls = 0
      self.lock = threading.Lock()

    def set_processor(self) -> None:
      """
      Set processor

      Args:
        processor_id (str): Processor ID

      Returns:
        None
      """

      if self.processor_id is None:
        opts = ClientOptions(api_endpoint=f"{self.location}-documentai.googleapis.com")
        client = documentai.DocumentProcessorServiceClient(client_options=opts)
        parent = client.common_location_path(self.project_id, self.location)

        self.processor_name = None
        # check if processor already exists
        processor_list = client.list_processors(parent=parent)
        for processor in processor_list:
          if processor.display_name == self.default_processor_display_name:
            self.processor_name = processor.name
            if self.verbose is True:
              print(f'Set processor "{self.processor_name}" ✓')
            break

        if self.processor_name is None and self.create_processor_if_not_exists:
          if self.verbose is True:
            print(f'Creating new processor of type "{self.default_processor_type}"' +
                   'with display name "{self.default_processor_display_name}"...')
          # create a processor
          processor = client.create_processor(
            parent=parent,
            processor=documentai.Processor(
                display_name=self.default_processor_display_name,
                type_=self.default_processor_type
            ),
          )
          self.processor_name = processor.name
          if self.verbose is True:
            print(f'Created and set processor "{self.processor_name}" ✓')
      else:
        self.processor_name = (
            f"projects/{self.project_id}/locations/{self.location}/processors/{self.processor_id}"
        )

    def _process_gcs_uri(self, uri: str) -> Sequence[str]:
      """
      Deconstruct GCS URI into scheme, bucket, path and file

      Args:
          uri (str): GCS URI

      Returns:
          scheme (str): URI scheme
          bucket (str): URI bucket
          path (str): URI path
          filename (str): URI file
      """
      url_arr = uri.split("/")
      if "." not in url_arr[-1]:
          filename = ""
      else:
          filename = url_arr.pop()
      scheme = url_arr[0]
      bucket = url_arr[2]
      path = "/".join(url_arr[3:])
      path = path[:-1] if path.endswith("/") else path
      return scheme, bucket, path, filename

    def _is_url(self, string):
      parsed = urlparse(string)
      return parsed.scheme and parsed.netloc

    def _get_mime_type(self, filename: str):
      """
      Get MIME type

      Args:
        filename (str): URI or file name.

      Returns:
        mime_type (str)
      """
      mime_type = 'application/pdf'
      if filename.lower().endswith("pdf"):
          mime_type = "application/pdf"
      elif filename.lower().endswith("tiff") or filename.endswith("tif"):
          mime_type = "image/tiff"
      elif filename.lower().endswith("jpeg") or filename.lower().endswith("jpg"):
          mime_type = "image/jpeg"
      elif filename.lower().endswith("png"):
          mime_type = "image/png"
      elif filename.lower().endswith("bmp"):
          mime_type = "image/bmp"
      elif filename.lower().endswith("webp"):
          mime_type = "image/webp"

      return mime_type

    def _get_docs_from_sources(self, sources: Sequence[str]):
      """
      Get docs from sources

      Args:
        sources (Sequence[str]): List of GCS URIs and website URLs

      Returns:
        docs (List[Document]): SourceDocuments
      """

      # initialize storage client
      storage_client = storage.Client()

      docs = []
      for source in sources:
          if source.startswith('gs://'):
              scheme, bucket, path, filename = self._process_gcs_uri(source)
              # create a bucket object for our bucket
              bucket = storage_client.get_bucket(bucket)
              # create a blob object from the filepath
              blob = bucket.blob(os.path.join(path, filename))

              if blob.exists():
                # get content bytes
                content = blob.download_as_bytes()
                if blob.content_type == "application/pdf":
                  pdf_reader = PdfReader(BytesIO(content))
                  for index, page in enumerate(pdf_reader.pages):
                    pdf_writer = PdfWriter()
                    pdf_writer.add_page(page)
                    response_bytes_stream = BytesIO()
                    pdf_writer.write(response_bytes_stream)
                    content = response_bytes_stream.getvalue()

                    # define source document
                    doc = SourceDocument(content=content, mime_type=blob.content_type, source=source, page=index + 1)
                    docs.append(doc)
                else:
                  # define source document
                  doc = SourceDocument(content=content, mime_type=blob.content_type, source=source)
                  docs.append(doc)

              else:
                raise ValueError(f'Source "{source}" does not exist.')

          elif self._is_url(source):
            # fetch content
            content = requests.get(source).content

            # get mime type
            mime_type = self._get_mime_type(source)

            if mime_type == "application/pdf":
              pdf_reader = PdfReader(BytesIO(content))
              for index, page in enumerate(pdf_reader.pages):
                pdf_writer = PdfWriter()
                pdf_writer.add_page(page)
                response_bytes_stream = BytesIO()
                pdf_writer.write(response_bytes_stream)
                content = response_bytes_stream.getvalue()

                # define source document
                doc = SourceDocument(content=content, mime_type=mime_type, source=source, page=index + 1)
                docs.append(doc)
            else:
              # define source document
              doc = SourceDocument(content=content, mime_type=mime_type, source=source)
              docs.append(doc)
          else:
              raise ValueError(f'Source "{source}" is not valid.')

      return docs

    def _process_doc_rate_limiter(self) -> None:
      """Process doc rate limiter"""
      import math

      # ensure thread safety
      with self.lock:
          current_time = time.time()
          last_minute_calls = self.doc_ai_api_calls[self.doc_ai_api_calls >= current_time - 60]

          # check the number of API calls within the sliding window
          while len(last_minute_calls) >= self.max_doc_ai_requests_per_min:
              # wait_time = last_minute_calls[-1] + 60 - current_time
              wait_time = last_minute_calls[math.ceil((len(last_minute_calls) - 1) / 2)] + 60 - current_time
              time.sleep(wait_time)

          while self.active_calls >= self.max_parallel_doc_ai_requests:
              time.sleep(.5)

          # add the current timestamp to the list of API calls
          self.active_calls += 1
          current_time = time.time()
          self.doc_ai_api_calls = np.append(self.doc_ai_api_calls, current_time)

    def process_doc(self, doc: SourceDocument) -> documentai.Document:
      """
      Process doc

      Args:
        doc (SourceDocument): SourceDocument

      Returns:
        result.document (documentai.Document)
      """
      # Rate limit the API calls
      self._process_doc_rate_limiter()

      # initialize Document AI client
      opts = ClientOptions(api_endpoint=f"{self.location}-documentai.googleapis.com")
      client = documentai.DocumentProcessorServiceClient(client_options=opts)

      # configure the process request
      raw_document = documentai.RawDocument(content=doc.content, mime_type=doc.mime_type)
      request = documentai.ProcessRequest(name=self.processor_name, raw_document=raw_document)

      # process the document
      # For a full list of `Document` object attributes, reference this page:
      # https://cloud.google.com/document-ai/docs/reference/rest/v1/Document
      result = client.process_document(request=request)

      return result.document

    def _process_sources(self, max_workers: Optional[int] = None) -> List[SourceDocument]:
      """
      Process sources

      Args:
        max_workers (Optional[int], optional): Maximum number of workers. Defaults to None.

      Returns:
        results (List[documentai.Document]): Documents
      """
      # get docs
      docs = self._get_docs_from_sources(self.sources)

      # initialize empty results
      results = [None] * len(docs)

      if not max_workers:
          max_workers = len(docs)

      # create thread pool with a max number of workers
      start_time = time.time()
      with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
          # submit each doc processing task to the thread pool
          future_to_doc = {executor.submit(self.process_doc, doc): (index, doc) for index, doc in enumerate(docs)}

          # Use tqdm to create a progress bar
          with tqdm(total=len(future_to_doc), position=0, leave=True) as progress_bar:

            # process the completed tasks as they finish
            for future in concurrent.futures.as_completed(future_to_doc):
                index, doc = future_to_doc[future]
                doc.set_document(future.result())
                results[index] = doc
                self.active_calls -= 1
                # update the progress bar
                progress_bar.update(1)

      end_time = time.time()
      time_lapse = end_time - start_time
      time_lapse_mins = round(time_lapse / 60, 3)
      if self.verbose is True:
        print(f'Processed {len(sources)} source(s) and got {len(results)} result(s) in {time_lapse_mins} mins.  ✓')

      return results

    def load(self, max_workers: Optional[int] = None) -> Tuple[List[Document], dict[Tuple[str, str]: bytes]]:
      """
      Load documents

      Args:
        max_workers (Optional[int], optional): Maximum number of workers. Defaults to None.

      Returns:
        documents (List[Document]): Documents
        images (dict[Tuple[str, str]]: bytes]): Images
      """
      self.set_processor()

      # get results
      results = self._process_sources(max_workers=max_workers)

      documents = []
      images = {}
      for result in results:

        images.setdefault((result.source, result.page), result.document.pages[0].image.content)

        # create documents from detected tables on page
        table_documents = []
        for table_index, detected_table in enumerate(result.document.pages[0].tables):
            text_segments = []
            for text_segment in detected_table.layout.text_anchor.text_segments:
              text_segments.append({"start_index": text_segment.start_index, "end_index": text_segment.end_index})
            text_segments_df = pd.DataFrame(text_segments)
            text_segments_df.sort_values(["start_index"], inplace=True)
            text_segments_df["text"] = text_segments_df.apply(lambda x: result.document.text[x.start_index: x.end_index], axis=1)

            vertices = []
            for vertex in detected_table.layout.bounding_poly.vertices:
              vertices.append({"x": vertex.x, "y": vertex.y})

            document = Document(
                page_content='\n'.join(text_segments_df["text"]),
                metadata={
                    "page": result.page,
                    "table": table_index + 1,
                    "mime_type": result.mime_type,
                    "source": result.source,
                    "vertices": vertices
                }
            )
            table_documents.append(document)

        documents.extend(table_documents)

        # create documents from detected blocks on page
        for block_index, detected_block in enumerate(result.document.pages[0].blocks):
            text_segments = []
            for text_segment in detected_block.layout.text_anchor.text_segments:
              text_segments.append({"start_index": text_segment.start_index, "end_index": text_segment.end_index})
            text_segments_df = pd.DataFrame(text_segments)
            text_segments_df.sort_values(["start_index"], inplace=True)
            text_segments_df["text"] = text_segments_df.apply(lambda x: result.document.text[x.start_index: x.end_index], axis=1)

            vertices = []
            for vertex in detected_block.layout.bounding_poly.vertices:
              vertices.append({"x": vertex.x, "y": vertex.y})
            block_shape = shapely.geometry.Polygon([(vertex['x'], vertex['y']) for vertex in vertices])

            # only use blocks that are not within table boundaries
            add_block = True
            for table_document in table_documents:
              table_shape = shapely.geometry.Polygon([(vertex['x'], vertex['y']) for vertex in table_document.metadata["vertices"]])
              if block_shape.intersects(table_shape):
                add_block = False

            if add_block:
              document = Document(
                  page_content=''.join(text_segments_df["text"]),
                  metadata={
                      "page": result.page,
                      "block": block_index + 1,
                      "mime_type": result.mime_type,
                      "source": result.source,
                      "vertices": vertices
                  }
              )
              documents.append(document)

      if self.verbose is True:
        print(f'Loaded {len(documents)} document(s) and {len(images)} image(s)  ✓')

      return documents, images

class ScaNN(VectorStore):
    """
    This class is a wrapper around the ScaNN Vector Similarity Search library.

    To use, you should have the ``scann`` python package installed.

    References:

    https://github.com/google-research/google-research/tree/master/scann
    """

    def __init__(
        self,
        embedding_function: Optional[Embeddings] = None,
        max_embedding_requests_per_min: int = 300,
        verbose: bool = True
    ) -> None:
        """Initialize the ScaNN vector store"""
        if embedding_function is None:
          embedding_function = VertexAIEmbeddings()
        self._embedding_function = embedding_function
        self._searcher = None
        self.max_embedding_requests_per_min = max_embedding_requests_per_min
        self.verbose = verbose
        self.embedding_api_calls = np.array([])
        self.lock = threading.Lock()

    def _embedding_rate_limiter(self) -> None:
      """Embedding rate limiter"""

      # ensure thread safety
      with self.lock:
          current_time = time.time()

          self.embedding_api_calls = self.embedding_api_calls[self.embedding_api_calls >= current_time - 60]

          # check the number of API calls within the sliding window
          if len(self.embedding_api_calls) >= self.max_embedding_requests_per_min:
              # if the limit is reached, calculate the remaining time until the next API call is allowed
              next_call_time = self.embedding_api_calls[0] + 60
              wait_time = next_call_time - current_time
              if wait_time > 0:
                  time.sleep(wait_time)

          # add the current timestamp to the list of API calls
          self.embedding_api_calls = np.append(self.embedding_api_calls, current_time)

    def _embed_query(self, query: str, index: Optional[int] = None) -> List:
      """
      Embed query

      Args:
        query (str): Query to embed
        index (Optional[int]): Index of the query

      Returns:
        index (Optional[int]): Index of the query
        embeddings (List): Embeddings
      """
      self._embedding_rate_limiter()
      embedding = self._embedding_function.embed_query(query)
      return index, embedding

    def embed_texts(self, texts: List[str], max_workers: Optional[int] = None) -> List[list]:
      """
      Embed texts

      Args:
        texts (List[str]): Texts to embed
        max_workers (Optional[int]): Maximum number of workers to use

      Returns:
        results (List[list]): Embeddings
      """
      # initialize empty results
      results = [None] * len(texts)

      if not max_workers:
          max_workers = len(texts)

      # create thread pool with a max number of workers
      start_time = time.time()
      with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
          # submit each doc processing task to the thread pool
          futures = [executor.submit(self._embed_query, text, index) for index, text in enumerate(texts)]

          # Use tqdm to create a progress bar
          with tqdm(total=len(futures), position=0, leave=True) as progress_bar:

            # process the completed tasks as they finish
            for future in concurrent.futures.as_completed(futures):
                index, embedding = future.result()
                results[index] = embedding

                # update the progress bar
                progress_bar.update(1)

      end_time = time.time()
      time_lapse = end_time - start_time
      time_lapse_mins = round(time_lapse / 60, 3)

      if self.verbose is True:
        print(f'Embedded {len(documents)} documents(s) in {time_lapse_mins} mins. ✓')

      return results

    def add_texts(
        self,
        texts: Iterable[str],
        num_neighbors: int = 10,
        distance_function: str = "dot_product",
        num_leaves: Optional[int] = None,
        num_leaves_to_search: Optional[int] = None,
        training_sample_size: Optional[int] = None,
        dimensions_per_block: int = 2,
        anisotropic_quantization_th: Optional[float] = 0.2,
        reorder_count: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Generate embeddings and add to the vectorstore

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            num_neighbors (int, optional): Number of neighbors to consider during partitioning. Defaults to 10.
            distance_function (str): Distance function to use. Defaults to "dot_product".
            num_leaves (Optional[int], optional): Number of leaves to search. Defaults to None.
            num_leaves_to_search (Optional[int], optional): Number of leaves to search. Defaults to None.
            training_sample_size (Optional[int], optional): Number of documents to use for training. Defaults to None.
            dimensions_per_block (int, optional): Number of dimensions per block. Defaults to 2.
            anisotropic_quantization_th (Optional[float], optional): Anisotropic quantization threshold. Defaults to 0.2.
            reorder_count (Optional[int], optional): Number of reorders to perform. Defaults to None.

        Returns:
            None
        """
        num_texts = len(texts)

        if num_leaves is None:
          num_leaves = num_texts
        if num_leaves_to_search is None:
          num_leaves_to_search = num_texts
        if training_sample_size is None:
          training_sample_size = num_texts
        if reorder_count is None:
          reorder_count = num_texts

        embeddings = self._embedding_function.embed_documents(list(texts))
        embeddings = self.embed_texts(texts)
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]

        # use scann.scann_ops.build() to instead create a TensorFlow-compatible searcher
        self._searcher = (
            scann.scann_ops_pybind.builder(normalized_embeddings, num_neighbors, distance_function)
            .tree(
                num_leaves=num_leaves,
                num_leaves_to_search=num_leaves_to_search,
                training_sample_size=training_sample_size,
            )
            .score_ah(
                dimensions_per_block,
                anisotropic_quantization_threshold=anisotropic_quantization_th)
            .reorder(reorder_count)
            .build()
        )

        try:
          INDEX_DIR = './index'
          os.makedirs(INDEX_DIR, exist_ok=True)
          self._searcher.serialize(INDEX_DIR) # store the scann_module
        except:
          print("Scanning failed")

        if self.verbose is True:
          print("ScaNN vector store is indexed and loaded. ✓")

    @classmethod
    def from_texts(
        cls: Type[VectorStore],
        texts: Iterable[str],
        embedding: Optional[Embeddings] = None,
        max_embedding_requests_per_min: int = 300,
        num_neighbors: int = 10,
        distance_function: str = "dot_product",
        num_leaves: Optional[int] = None,
        num_leaves_to_search: Optional[int] = None,
        training_sample_size: Optional[int] = None,
        dimensions_per_block: int = 2,
        anisotropic_quantization_th: Optional[float] = 0.2,
        reorder_count: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> VectorStore:
        """
        Create a ScaNN vector store from a list of texts.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            embedding (Optional[Embeddings], optional): Embeddings to use for the vectorstore. Defaults to None.
            max_embedding_requests_per_min (int, optional): Maximum number of embedding requests per minute. Defaults to 500.
            num_neighbors (int, optional): Number of neighbors to consider during partitioning. Defaults to 10.
            distance_function (str): Distance function to use. Defaults to "dot_product".
            num_leaves (Optional[int], optional): Number of leaves to search. Defaults to None.
            num_leaves_to_search (Optional[int], optional): Number of leaves to search. Defaults to None.
            training_sample_size (Optional[int], optional): Number of documents to use for training. Defaults to None.
            dimensions_per_block (int, optional): Number of dimensions per block. Defaults to 2.
            anisotropic_quantization_th (Optional[float], optional): Anisotropic quantization threshold. Defaults to 0.2.
            reorder_count (Optional[int], optional): Number of reorders to perform. Defaults to None.
            verbose (bool, optional): Verbosity. Defaults to True.
        Returns:
            scann_store: ScaNN vectorstore.
        """
        # create scann store
        scann_store = cls(
            embedding_function=embedding,
            max_embedding_requests_per_min=max_embedding_requests_per_min,
            verbose=verbose)

        # add texts to scann score
        scann_store.add_texts(
            texts=texts,
            num_neighbors=num_neighbors,
            distance_function=distance_function,
            num_leaves=num_leaves,
            num_leaves_to_search=num_leaves_to_search,
            training_sample_size=training_sample_size,
            dimensions_per_block=dimensions_per_block,
            anisotropic_quantization_th=anisotropic_quantization_th,
            reorder=reorder_count,
        )
        return scann_store


    @classmethod
    def from_documents(
        cls: Type[VectorStore],
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        max_embedding_requests_per_min: int = 300,
        num_neighbors: int = 10,
        distance_function: str = "dot_product",
        num_leaves: Optional[int] = None,
        num_leaves_to_search: Optional[int] = None,
        training_sample_size: Optional[int] = None,
        dimensions_per_block: int = 2,
        anisotropic_quantization_th: Optional[float] = 0.2,
        reorder_count: Optional[int] = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> VectorStore:
        """
        Create a ScaNN vector store from a list of documents.

        Args:
            documents (List[Document]): Documents to add to the vectorstore.
            embedding (Optional[Embeddings], optional): Embeddings to use for the vectorstore. Defaults to None.
            max_embedding_requests_per_min (int, optional): Maximum number of embedding requests per minute. Defaults to 500.
            num_neighbors (int, optional): Number of neighbors to consider during partitioning. Defaults to 10.
            distance_function (str): Distance function to use. Defaults to "dot_product".
            num_leaves (Optional[int], optional): Number of leaves to search. Defaults to None.
            num_leaves_to_search (Optional[int], optional): Number of leaves to search. Defaults to None.
            training_sample_size (Optional[int], optional): Number of documents to use for training. Defaults to None.
            dimensions_per_block (int, optional): Number of dimensions per block. Defaults to 2.
            anisotropic_quantization_th (Optional[float], optional): Anisotropic quantization threshold. Defaults to 0.2.
            reorder_count (Optional[int], optional): Number of reorders to perform. Defaults to None.
            verbose (bool, optional): Verbosity. Defaults to True.
        Returns:
            scann_store: ScaNN vectorstore.
        """
        texts = [doc.page_content for doc in documents]

        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            max_embedding_requests_per_min=max_embedding_requests_per_min,
            num_neighbors=num_neighbors,
            distance_function=distance_function,
            num_leaves=num_leaves,
            num_leaves_to_search=num_leaves_to_search,
            training_sample_size=training_sample_size,
            dimensions_per_block=dimensions_per_block,
            anisotropic_quantization_th=anisotropic_quantization_th,
            reorder_count=reorder_count,
            verbose=verbose
        )

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Run similarity search with ScaNN

        Args:
            query (str): Query text to search for.
            k (Optional[int]): Number of results to return. Defaults to 4.

        Returns:
            List[Document]: List of documents most similar to the query text.
            distances (List[float]): List of distances to the query text.
        """
        query = self._embedding_function.embed_query(query)
        neighbors, distances = self._searcher.search(query, final_num_neighbors=k)
        return neighbors, distances


    # @classmethod
    # def save_env(self, GCS_BUCKET, DEST_FOLDER):
    #     """
    #     Create a ScaNN vector store from a list of texts.

    #     Args:
    #         verbose (bool, optional): Verbosity. Defaults to True.
    #     Returns:
    #         scann_store: ScaNN vectorstore.
    #     """
    #     INDEX_DIR = './index'
    #     DEST_FOLDER = "gs://informed_consents/embeddings/"
    #     # create scann store
    #     !gsutil cp -r {INDEX_DIR} {DEST_FOLDER}
    #     import pickle
    #     with open("documents", "wb") as fp:   #Pickling
    #       pickle.dump(documents, fp)
    #     !gsutil cp -r './documents' {DEST_FOLDER}

    #     with open("images", "wb") as fp:   #Pickling
    #       pickle.dump(documents, fp)
    #     !gsutil cp -r './images' {DEST_FOLDER}

    #     print("ENV Saved.")

    # @classmethod
    # def load_env(self, GCS_BUCKET, DEST_FOLDER):
    #     """
    #     Create a ScaNN vector store from a list of texts.

    #     Args:
    #         verbose (bool, optional): Verbosity. Defaults to True.
    #     Returns:
    #         scann_store: ScaNN vectorstore.
    #     """
    #     # create scann store
    #     INDEX_DIR = './'
    #     DEST_FOLDER = "gs://informed_consents/embeddings/"
    #     # create scann store
    #     !gsutil cp -r {DEST_FOLDER} .
    #     print("Downloaded!")
    #     !ls .


        #!gsutil cp -r {GCS_BUCKET} './documents'
        import pickle

        with open("documents", "rb") as fp:   # Unpickling
           b = pickle.load(fp)

        try:
          with open("images", "rb") as fp:   # Unpickling
            c = pickle.load(fp)
        except:
          c = {}

        another_searcher = scann.scann_ops_pybind.load_searcher("/content/index/")

        print("ENV Loaded.")
        return c, b, another_searcher

class DocumentBot:
  """"
  A bot that can answer questions about documents.
  """

  def __init__(
      self,
      documents: List[Document],
      images: dict[Tuple[str, str]: bytes],
      vector_store: VectorStore,
      llm: BaseLLM,
      answer_template: str,
      prompt: Optional[PromptTemplate] = None,
      signed_url_target_scopes: Optional[List[str]] = None,
      signing_service_account: Optional[str] = None,
      signing_service_account_credentials_lifetime: int = 300,
      signed_url_mins_to_expiration: int = 15,
      verbose: bool = True
    ) -> None:
    self.documents = documents
    self.images = images
    self.vector_store = vector_store
    self.llm = llm
    self.answer_template = answer_template
    self.prompt = prompt

    if signed_url_target_scopes is None:
      signed_url_target_scopes = ['https://www.googleapis.com/auth/devstorage.read_only']
    self.signed_url_target_scopes = signed_url_target_scopes
    self.signing_service_account = signing_service_account
    self.signing_service_account_credentials_lifetime = signing_service_account_credentials_lifetime
    self.signed_url_mins_to_expiration = signed_url_mins_to_expiration

    self.verbose = verbose

    self.chain = load_qa_chain(self.llm, chain_type="stuff", prompt=self.prompt)

  def _get_qa_chain_output(self, query) -> Tuple[dict[str: Any], List[float]]:
    """
    Get the output of the QA chain for a given query.

    Args:
        query (str): Query text to search for.

    Returns:
        dict[str: Any]: Output of the QA chain.
        distances (List[float]): List of distances to the query text.
    """
    # get nearest neighbors and distances
    neighbors, distances = vector_store.similarity_search(query, k=5)
    matching_documents = [self.documents[i] for i, distance in zip(neighbors, distances)]
    output = self.chain({"input_documents": matching_documents, "question": query})
    return output, distances

  def _process_gcs_uri(self, uri: str) -> Sequence[str]:
      """
      Deconstruct GCS URI into scheme, bucket, path and file

      Args:
          uri (str): GCS URI

      Returns:
          scheme (str): URI scheme
          bucket (str): URI bucket
          path (str): URI path
          filename (str): URI file
      """
      url_arr = uri.split("/")
      if "." not in url_arr[-1]:
          filename = ""
      else:
          filename = url_arr.pop()
      scheme = url_arr[0]
      bucket = url_arr[2]
      path = "/".join(url_arr[3:])
      path = path[:-1] if path.endswith("/") else path
      return scheme, bucket, path, filename

  def _generate_download_signed_url_v4(
      self,
      bucket_name: str,
      blob_name: str,
      target_credentials: impersonated_credentials.Credentials,
  ) -> str:
    """
    Generates a v4 signed URL for downloading a blob.

    Args:
        bucket_name (str): Bucket name.
        blob_name (str): Blob name.
        target_credentials (impersonated_credentials.Credentials): Target credentials.

    Returns:
        str: Signed URL.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=self.signed_url_mins_to_expiration),
        method="GET",
        credentials=target_credentials
    )

    return url

  def _get_formatted_sources(
      self,
      output,
      distances,
      gradio_format: bool = False,
      serve_signed_urls: bool = True
  ) -> List[str]:
    """
    Format the sources.

    Args:
        output (dict[str: Any]): Output of the QA chain.

    Returns:
        List[str]
    """
    # initialize empty formatted sources
    formatted_sources = []

    # get credentials
    credentials, _ = default()

    # get target credentials
    target_credentials = impersonated_credentials.Credentials(
        source_credentials=credentials,
        target_principal=self.signing_service_account,
        target_scopes=self.signed_url_target_scopes,
        lifetime=self.signing_service_account_credentials_lifetime
    )

    for index, input_document in enumerate(output["input_documents"]):
      page_number = input_document.metadata["page"]
      source = input_document.metadata["source"]
      source = f'{source}#page={page_number}' if page_number else source
      signed_url = source

      # if GCS URI, get signed url
      if source.startswith("gs://") and serve_signed_urls:
        scheme, bucket_name, path, filename = self._process_gcs_uri(input_document.metadata["source"])

        signed_url = self._generate_download_signed_url_v4(
            bucket_name=bucket_name,
            blob_name=os.path.join(path, filename),
            target_credentials=target_credentials,
        )
        signed_url += f"#page={page_number}" if page_number else signed_url


      similarity = distances[index]
      if index == 0 and not gradio_format:
        top_reference = "* page: {page}, relevance to answer: {similarity:.2f}\n"
        top_reference += "* [{source}]({signed_url})"
        top_reference = top_reference.format(
          page=page_number,
          source=source,
          signed_url=signed_url,
          similarity=similarity
        )
        formatted_sources.append(top_reference)

      if gradio_format:
        reference = '<p><b>{index}</b>. <a href="{signed_url}">{source}</a>, relevance to question: {similarity:.2f}</p>'
        reference = reference.format(
          index=index + 1,
          source=source,
          signed_url=signed_url,
          similarity=similarity
        )
      else:
        reference = "* [{source}]({signed_url})\n\t* page: {page}, relevance to question: {similarity:.2f}"
        reference = reference.format(
            page=page_number,
            source=source,
            signed_url=signed_url,
            similarity=similarity
        )
      formatted_sources.append(reference)

    return formatted_sources


  def answer(self, query: str, serve_signed_urls: bool = False) -> None:
    """
    Get Answer from QA chain.

    Args:
        query (str): Query text to search for.
        serve_signed_urls (bool): Whether to serve signed urls.

    Returns:
      output (dict[str: Any]): Output of the QA chain.
    """
    output, distances = self._get_qa_chain_output(query)
    answer = output["output_text"]
    output, distances = self._get_qa_chain_output(answer)

    if self.verbose is True:
      # get formatted sources
      formatted_sources = self._get_formatted_sources(
          output, distances,
          serve_signed_urls=serve_signed_urls
      )

      # format answer
      answer_str = self.answer_template.format(
          question=query,
          output_text=answer,
          top_reference=formatted_sources[0],
          sources='\n'.join(formatted_sources[1:])
      )

      # display answer
      IPython.display.display(IPython.display.Markdown(answer_str))

      # display image
      top_source = output["input_documents"][0].metadata["source"]
      top_page = output["input_documents"][0].metadata["page"]
      top_vertices = output["input_documents"][0].metadata["vertices"]
      top_vertices = pd.DataFrame(top_vertices).stack().values.tolist()
      image = PIL.Image.open(BytesIO(self.images[(top_source, top_page)]))
      PIL.ImageDraw.Draw(image).polygon(top_vertices, outline = 'green', width=5)
      IPython.display.display(image.resize((800, 1000)))
    return output


  def gradio_answer(self, query: str, serve_signed_urls: bool = False) -> None:
    """
    Get answer formatted for gradio

    Args:
        query (str): Query text to search for.
        serve_signed_urls (bool): Whether to serve signed urls.

    Returns:
      output (dict[str: Any]): Output of the QA chain.
    """
    output, distances = self._get_qa_chain_output(query)
    answer = output["output_text"]
    output, distances = self._get_qa_chain_output(answer)

    # get formatted sources
    formatted_sources = self._get_formatted_sources(
        output, distances,
        gradio_format=True,
        serve_signed_urls=serve_signed_urls
    )
    formatted_sources = '</br>'.join(formatted_sources)
    top_source = output["input_documents"][0].metadata["source"]
    top_page = output["input_documents"][0].metadata["page"]
    top_vertices = output["input_documents"][0].metadata["vertices"]
    top_vertices = pd.DataFrame(top_vertices).stack().values.tolist()
    image = PIL.Image.open(BytesIO(self.images[(top_source, top_page)]))
    PIL.ImageDraw.Draw(image).polygon(top_vertices, outline = 'green', width=5)
    return answer, image, formatted_sources, round(distances[0], 3)

################
### Load document
################

# GCS URIs and public website URLs to files in the following formats are supported: PDF; Images (e.g. tiff, jpeg, png, bmp, webp)

sources = ["gs://bdoohan-icf-aspirin/Prot_000_Aspirin.pdf"]

docai_loader = DocAILoader(sources=sources, project_id=PROJECT_ID, max_doc_ai_requests_per_min=110)

documents, images = docai_loader.load()

# """## Create vector store

# ### Option #1: ScaNN
# """

# vector_store = ScaNN.from_documents(documents)

# """## Perform Retrieval QA

# Define the following:
# * Prompt template
# * LLM Model
# * Answer template
# """

# prompt_template = """\
# Use the context to answer the prompt enclosed within "<" and ">" replaced with the correct text. Use the context and prompt to replace the content between "<" and ">".

# {context}

# Question:
# {question}

# Helpful Answer and Explanation:
# """
# prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# # LLM model
# model_name = "text-bison@001" #@param {type: "string"}
# max_output_tokens = 256 #@param {type: "integer"}
# temperature = 0.2 #@param {type: "number"}
# top_p = 0.8 #@param {type: "number"}
# top_k = 40 #@param {type: "number"}
# verbose = True #@param {type: "boolean"}
# llm = VertexAI(
#   model_name=model_name,
#   max_output_tokens=max_output_tokens,
#   temperature=temperature,
#   top_p=top_p,
#   top_k=top_k,
#   verbose=verbose
# )

# answer_template = """\
# ## Response
# ### Question
# {question}
# ### Answer
# {output_text}
# ### Why
# {top_reference}

# ### Sources
# {sources}
# """

# document_bot = DocumentBot(
#   documents=documents,
#   images=images,
#   vector_store=vector_store,
#   #vector_store=another_searcher,
#   llm=llm,
#   prompt=prompt,
#   answer_template=answer_template,
#   signing_service_account=SIGNING_SERVICE_ACCOUNT
# )

# """### Ask a Question

# Note: it takes a few seconds more (e.g. 2-8 secs) ⌛ to serve signed urls for the GCS quick links. If solely trying to evaluate speed,  set _serve_signed_urls_ to False. This parameter does not have an effect either way on public documents.
# """

# query = "How many participants were included in this study?" #@param {type:"string"}
# _ = document_bot.answer(query, serve_signed_urls=True)

# query = "The mechanism of action for <insert name of drug being studied> is: ." #@param {type:"string"}
# _ = document_bot.answer(query, serve_signed_urls=True)

# query = "By doing this study, researchers hope to learn more about <insert>" #@param {type:"string"}
# _ = document_bot.answer(query, serve_signed_urls=True)

# query = "Which disease does the drug treat? How does it do that?" #@param {type:"string"}
# _ = document_bot.answer(query, serve_signed_urls=True)

# """### Gradio

# Lightweight UI for document search and summarization.
# """

# with gr.Blocks() as demo:
#     gr.Markdown(
#     """
#     ## Document Search

#     This demo showcases document search of input documents using Document AI 📑👀 + LangChain 🦜️🔗 + ScaNN / Matching Engine 🧩🔍.

#     """)
#     with gr.Row():
#       with gr.Column():
#         query = gr.Textbox(label="Query", placeholder="Enter a question")

#     with gr.Row():
#       generate = gr.Button("Answer")

#     gr.Markdown(
#     """
#     ## Summary
#     """)
#     with gr.Row():
#       answer_label = gr.Textbox(label="Response")

#     image = gr.Image(type="pil")

#     with gr.Row():
#       confidence_score = gr.Textbox(label="Confidence")

#     gr.Markdown(
#     """
#     ## Sources
#     """)
#     with gr.Row():
#       sources = gr.HTML(label="Sources")

#     generate.click(document_bot.gradio_answer, query, [answer_label, image, sources, confidence_score])

# demo.launch(share=False, debug=False)































# ################
# ### model inputs
# ################

# # Model Inputs
# st.divider()
# st.header('1. Model Inputs')

# model_id = st.selectbox(
#     'Which model do you want to use?'
#     , (
#           'code-bison@001'
#         , 'code-bison@latest'
#       )
#     , index = 0
#   )
# model_temperature = st.number_input(
#       'Model Temperature'
#     , min_value = 0.0
#     , max_value = 1.0
#     , value = 0.2
#   )
# model_token_limit = st.number_input(
#       'Model Token Limit'
#     , min_value = 1
#     , max_value = 1024
#     , value = 200
#   )
# model_top_k = st.number_input(
#       'Top-K'
#     , min_value = 1
#     , max_value = 40
#     , value = 40
#   )
# model_top_p = st.number_input(
#       'Top-P'
#     , min_value = 0.0
#     , max_value = 1.0
#     , value = 0.8
#   )

# ################
# ### Query information 
# ################

# # Query Information
# st.divider()
# st.header('2. Query information')

# # List datasets 
# # project_id = "cloudadopt-public-data"
# project_id = "cloudadopt"
# project_id_datasets = "cloudadopt-public-data"
# location_id = "us-central1"

# client = bigquery.Client(project=project_id_datasets)
# datasets_all = list(client.list_datasets())  # Make an API request.
# dataset_string = ''
# for dataset in datasets_all: 
#     dataset_id2 = dataset.dataset_id
#     dataset_id2 = '\'' + dataset_id2 + '\''
#     dataset_string = dataset_string + dataset_id2 + ','
# dataset_string = dataset_string[:-1]

# string1 = 'dataset_id = st.selectbox('
# string2 = '\'What dataset do you want to query?\''
# string3 = f', ({dataset_string}), index = 1)'
# string_full = string1 + string2 + string3
# exec(string_full)

# # List tables
# table_names = [table.table_id for table in client.list_tables(f"{project_id_datasets}.{dataset_id}")]
# table_names_str = '\n'.join(table_names)
# st.write(':blue[**Tables:**] ')
# st.text(table_names_str)

# table_uri = f"bigquery://{project_id_datasets}/{dataset_id}"
# engine = create_engine(f"bigquery://{project_id_datasets}/{dataset_id}")

# # Vertex 
# vertexai.init(
#     project=project_id
#   , location=location_id
# )

# # LLM model
# model_name = "text-bison@001" #@param {type: "string"}
# max_output_tokens = 1024 #@param {type: "integer"}
# temperature = 0.2 #@param {type: "number"}
# top_p = 0.8 #@param {type: "number"}
# top_k = 40 #@param {type: "number"}
# verbose = True #@param {type: "boolean"}

# llm = VertexAI(
#   model_name=model_name,
#   max_output_tokens=max_output_tokens,
#   temperature=temperature,
#   top_p=top_p,
#   top_k=top_k,
#   verbose=verbose
# )

# ################
# ### Provide SQL Rules
# ################

# # Provide SQL Rules
# st.divider()
# st.header('3. Provide SQL Rules')

# include_sql_rules = st.selectbox(
#     'Do you want to include general SQL rules?'
#     , (
#           'yes'
#         , 'no'
#       )
#     , index = 0
#   )

# include_dataset_rules = st.selectbox(
#     'Do you want to include SQL rules on this particular dataset?'
#     , (
#           'yes'
#         , 'no'
#       )
#     , index = 0
#   )

# sql_quesiton = st.text_input(
#    'What is your question?'
#    , value = "How many flights were delayed in California in 2002?"
#   )

# custom_rules = st.text_input(
#    'If you need to include any additional SQL rules, provide them here'
#    , value = "Define a delay as 30 minutes or greater; use 2-letter abbreviations for states"
#   )

# if include_sql_rules == 'yes':
#    text_sql_rules = """
# You are a GoogleSQL expert. Given an input question, first create a syntactically correct GoogleSQL query to run, then look at the results of the query and return the answer to the input question.
# Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per GoogleSQL. You can order the results to return the most informative data in the database.
# Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
# Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
# Use the following format:
# Question: "Question here"
# SQLQuery: "SQL Query to run"
# SQLResult: "Result of the SQLQuery"
# Answer: "Final answer here"
# Only use the following tables:
# {table_info}

# Rule 1:
# Do not filter and query for columns that do not exist in the table being queried.

# Rule 2:
# If someone asks for the column names in all the tables, use the following format:
# SELECT column_name
# FROM `{project_id}.{dataset_id}`.INFORMATION_SCHEMA.COLUMNS
# WHERE table_name in {table_info}

# Rule 3:
# If someone asks for the column names in a particular table (let's call that provided_table_name for sake of example), use the following format:
# SELECT column_name
# FROM `{project_id}.{dataset_id}`.INFORMATION_SCHEMA.COLUMNS
# WHERE table_name = "provided_table_name"

# # Rule 4:
# # Double check that columns used in query are correctly named and exist in the corresponding table being queried. Use actual names that exist in table, not synonyms.

# # Rule 5:
# # If someone mentions a specific table name (e.g. flights data, carriers data), assume the person is referring to the table name(s) that most closely correspond to the name.

# # Rule 6:
# # Follow each rule every time.

# """
# else:
#    text_sql_rules = ""

# if include_dataset_rules == 'yes' and 'faa' in dataset_id: 
#    text_sql_rules_specific = """
# Rule:
# When filtering on specific dates, convert the date(s) to DATE format before executing the query.

# Example:

#   User Query: "How many flights were there between January 10th and 17th in 2022?"
#   Desired Query Output:
#   SELECT count(*)
#   FROM `{project_id}.{dataset_id}.flights` 
#   AND partition_date BETWEEN DATE('2022-01-10') AND DATE('2022-01-17')

# Rule:
# Flights and airports join on flights.origin = airports.code

# Example: 

#   User Query: "How many flights were there in Florida between January 10th and 17th in 2022?"
#   Desired Query Output:
#   SELECT count(*)
#   FROM `{project_id}.{dataset_id}.flights` AS flights 
#   JOIN `{project_id}.{dataset_id}.airports` AS airports
#     ON flights.origin = airports.code
#   WHERE airports.state = 'Florida'
#   AND flights.partition_date BETWEEN DATE('2022-01-10') AND DATE('2022-01-17')

# Rule:
# Flights and carriers join on flights.carrier = carriers.code

# Example: 

#   User Query: "Which carriers had the highest percent of flights delayed by more than 15 minutes?"
#   Desired Query Output:
#   SELECT carriers.name, sum(case when dep_delay > 15 then 1 else 0 end) / count(*) as percent_delayed_flights
#   FROM `{project_id}.{dataset_id}.flights` AS flights 
#   JOIN `{project_id}.{dataset_id}.carriers` AS carriers
#     ON flights.carrier = carriers.code
#   GROUP BY 1 
#   ORDER BY 2 desc 
#   LIMIT 10

# Rule:
# If you are doing a join and referring to a table by short hand name, don't forget to name the table being joined with an "AS"

# Example:

#   SELECT carriers.name, flights.origin, flights.destination, count(*)
#   FROM `{project_id}.{dataset_id}.flights` AS flights 
#   JOIN `{project_id}.{dataset_id}.carriers` AS carriers
#     ON flights.carrier = carriers.code
#   GROUP BY 1,2,3
#   ORDER BY 4 desc 
#   LIMIT 200

# """
# else: 
#     text_sql_rules_specific = ""

# if custom_rules == "":
#    text_custom_rules = custom_rules 
# else: 
#    text_custom_rules = f"""

# Additional Context: {custom_rules}

# """

# text_final = "Question: {input}"

# sql_prompt = text_sql_rules + text_sql_rules_specific + text_custom_rules + text_final



# ################
# ### SQL Answer
# ################

# # Provide SQL Rules
# st.divider()
# st.header('4. SQL Answer')

# def bq_qna(question):
#   #create SQLDatabase instance from BQ engine
#   db = SQLDatabase(
#       engine=engine
#      ,metadata=MetaData(bind=engine)
#      ,include_tables=table_names # [x for x in table_names]
#   )

#   #create SQL DB Chain with the initialized LLM and above SQLDB instance
#   db_chain = SQLDatabaseChain.from_llm(
#       llm
#      , db
#      , verbose=True
#      , return_intermediate_steps=True)

#   #Define prompt for BigQuery SQL
#   _googlesql_prompt = sql_prompt

#   GOOGLESQL_PROMPT = PromptTemplate(
#       input_variables=[
#          "input"
#          , "table_info"
#          , "top_k"
#          , "project_id"
#          , "dataset_id"
#       ],
#       template=_googlesql_prompt,
#   )

#   #passing question to the prompt template
#   final_prompt = GOOGLESQL_PROMPT.format(
#        input=question
#      , project_id =project_id_datasets
#      , dataset_id=dataset_id
#      , table_info=table_names
#      , top_k=10000
#     )

#   # pass final prompt to SQL Chain
#   output = db_chain(final_prompt)

#   # outputs
#   st.write(':blue[**SQL:**] ')
#   sql_answer = output['intermediate_steps'][1]
#   st.code(sql_answer, language="sql", line_numbers=False)

#   st.write(':green[**Answer:**] ')
#   st.write(output['result'])


#   st.write(':blue[**Full Work:**] ')
#   st.write(output)  

# bq_qna(sql_quesiton)

















