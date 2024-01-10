
import logging
import os
from utils import get_env_var

from llama_index import OpenAIEmbedding, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.text_splitter import SentenceSplitter
from SecFilingReader import SecFilingReader
from llama_index.ingestion import IngestionPipeline
from llama_index.extractors import TitleExtractor


def get_edgar_index(cik: str):
	"""
	Create or load an existing index using filing data from the SEC Edgar API for the specified company.

	Args:
		cik (str): The CIK of the company to index.
	"""
	user_email = get_env_var('USER_EMAIL')

	storage_path = f"./storage/{cik}"  # Store index by cik to allow multiple companies to be indexed at some point
	if not os.path.exists(storage_path):
		# load the documents and create the index
		print("Creating index")

		logging.debug("Loading Documents")
		documents = SecFilingReader(user_email=user_email).load_data(cik=cik, filing_date_min="2020-01-01", forms=["10-K"])
		logging.debug(f"  Documents: {len(documents)}")

		transformations = [
			SentenceSplitter(),
			TitleExtractor(),
			OpenAIEmbedding()
			]

		pipeline = IngestionPipeline(
			transformations=transformations,	
		)
		logging.debug("Documents -> Nodes")
		nodes = pipeline.run(documents=documents)
		logging.debug(f"  Nodes: {len(nodes)}")

		logging.debug("Nodes -> Index")
		index = VectorStoreIndex(nodes)
		# index = VectorStoreIndex.from_documents(documents, transformations=transformations)

		# store it for later
		print("Storing index")
		index.storage_context.persist(persist_dir=storage_path)
	else:
		# load the existing index
		storage_context = StorageContext.from_defaults(persist_dir=storage_path)
		index = load_index_from_storage(storage_context)
		print("Loaded index from storage")
	return index