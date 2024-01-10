from logging import error as log_error
from os import getenv
from typing import List, Optional

from dotenv import load_dotenv
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores import VectorStore, Weaviate
from weaviate import AuthApiKey, Client

from .base import BaseVectorDB
from ..utils import COLLECTION_NAME, value_of

load_dotenv()

# WEAVIATE_API_KEY is automatically used if set
if value_of(getenv('WEAVIATE_URL')) is None:
	raise Exception('Error: environment variable WEAVIATE_URL is not set')


class_schema = {
	'properties': [
		{
			'dataType': ['text'],
			'description': 'The actual text',
			'name': 'text',
		},
		{
			'dataType': ['text'],
			'description': 'The type of source/mimetype of file',
			'name': 'type',
		},
		{
			'dataType': ['text'],
			'description': 'The name or subject of the source',
			'name': 'title',
		},
		{
			'dataType': ['text'],
			'description': 'The source of the text (for files: `file: fileId`)',
			'name': 'source',
		},
		{
			'dataType': ['int'],
			'description': 'Start index of chunk',
			'name': 'start_index',
		},
		{
			# https://weaviate.io/developers/weaviate/config-refs/datatypes#datatype-date
			'dataType': ['text'],
			'description': 'Last modified time of the file',
			'name': 'modified',
		},
	],
	# TODO: optimisation for large number of objects
	'vectorIndexType': 'hnsw',
	'vectorIndexConfig': {
		'skip': False,
		# 'ef': 99,
		# 'efConstruction': 127,  # minimise this for faster indexing
		# 'maxConnections': 63,
	}
}


class VectorDB(BaseVectorDB):
	def __init__(self, embedding: Optional[Embeddings] = None, **kwargs):
		try:
			client = Client(
				**({
					'auth_client_secret': AuthApiKey(getenv('WEAVIATE_APIKEY')),
				} if value_of(getenv('WEAVIATE_APIKEY')) is not None else {}),
				**{**{
					'url': getenv('WEAVIATE_URL'),
					'timeout_config': (1, 20),
					**kwargs,
				}},
			)
		except Exception as e:
			raise Exception(f'Error: Weaviate connection error: {e}')

		if not client.is_ready():
			raise Exception('Error: Weaviate connection error')

		self.client = client
		self.embedding = embedding

	def setup_schema(self, user_id: str) -> None:
		if not self.client:
			raise Exception('Error: Weaviate client not initialised')

		if self.client.schema.exists(COLLECTION_NAME(user_id)):
			return

		self.client.schema.create_class({
			'class': COLLECTION_NAME(user_id),
			**class_schema,
		})

	def get_user_client(
			self,
			user_id: str,
			embedding: Optional[Embeddings] = None  # Use this embedding if not None or use global embedding
		) -> Optional[VectorStore]:
		self.setup_schema(user_id)

		em = None
		if self.embedding is not None:
			em = self.embedding
		elif embedding is not None:
			em = embedding

		weaviate_obj = Weaviate(
			client=self.client,
			index_name=COLLECTION_NAME(user_id),
			text_key='text',
			embedding=em,
			by_text=False,
		)
		weaviate_obj._query_attrs = ['text', 'start_index', 'source', 'title', 'type', 'modified']

		return weaviate_obj

	def get_objects_from_sources(self, user_id: str, source_names: List[str]) -> dict:
		# NOTE: the limit of objects returned is not known, maybe it would be better to set one manually

		if not self.client:
			raise Exception('Error: Weaviate client not initialised')

		if not self.client.schema.exists(COLLECTION_NAME(user_id)):
			self.setup_schema(user_id)

		file_filter = {
			'path': ['source'],
			'operator': 'ContainsAny',
			'valueTextList': source_names,
		}

		results = self.client.query \
			.get(COLLECTION_NAME(user_id), ['source', 'modified']) \
			.with_additional('id') \
			.with_where(file_filter) \
			.do()

		if results.get('errors') is not None:
			log_error(f'Error: Weaviate query error: {results.get("errors")}')
			return {}

		dsources = {}
		for source in source_names:
			dsources[source] = True

		try:
			results = results['data']['Get'][COLLECTION_NAME(user_id)]
			output = {}
			for result in results:
				# case sensitive matching
				if dsources.get(result['source']) is None:
					continue

				output[result['source']] = {
					'id': result['_additional']['id'],
					'modified': result['modified'],
				}

			return output
		except Exception as e:
			log_error(f'Error: Weaviate query error: {e}')
			return {}
