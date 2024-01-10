import json

from src.clients.openai import langchain_processor
from src.repositories.file_indexes_repository import *
from src.repositories.file_indexer_status_repository import *
from src.configs.redis_config import redis_instance
from src.utils.indexing_states import INDEXING_FINISHED, INDEXING_STARTED
from src.repositories.api_key_repository import ApiKeyRepository


class IndexerStartedHandler():
  
    @staticmethod
    def listen():
        redisClient = redis_instance()
        pubsub = redisClient.pubsub()
        pubsub.subscribe('gpt_indexer')
        handler = IndexerStartedHandler()
        for message in pubsub.listen():
            try:
                item = json.loads(message.get('data'))
                handler.run(
                  item['organization'], 
                  item['process_uuid'], 
                  item['google_drive_id'],
                  item['google_token'],
                )
            except Exception as e:
                print("An exception occurred, payload: ", message)
                print(e)
  
  
    def run(self, organization, process_uuid, google_drive_id, google_token):
      # initialize mongodb repository
      repository = FileIndexesRepository()
      openai_api_key = ApiKeyRepository().get_by_organization_id(organization)['api_key']
      
      # get the initial state of the record (from request) so it can be processed by the indexer
      content = repository.get_by_organization(organization, INDEXING_STARTED)
      
      # generate content index using openai
      langchain_processor.generate_string_index(content, organization, google_drive_id, google_token, openai_api_key)

      # delete any existing records from organization
      repository.delete({
        "organization": organization,
        "state": INDEXING_FINISHED,
      })

      # add indexed content to the organization
      repository.insert({
        "organization": organization,
        "process_uuid": process_uuid,
        "state": INDEXING_FINISHED,
        "google_drive_id": google_drive_id,
        "content": content
      })

      FileIndexerStatusRepository().insert({
        "organization": organization,
        "process_uuid": process_uuid,
      })

