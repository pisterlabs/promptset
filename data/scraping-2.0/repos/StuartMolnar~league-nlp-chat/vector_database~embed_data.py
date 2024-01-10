import os
import openai
import pinecone
import yaml
import logging
import logging.config
from dotenv import load_dotenv
from prepare_data import PrepareData
import datetime
from typing import List, Tuple, Dict, Union, Optional, Callable

try:
    with open('log_conf.yml', 'r') as f:
        log_config = yaml.safe_load(f.read())
        logging.config.dictConfig(log_config)
except (FileNotFoundError, yaml.YAMLError) as e:
    print(f"Error loading logging configuration: {e}")

logger = logging.getLogger('basicLogger')

try:
    with open('app_conf.yml', 'r') as f:
        app_config = yaml.safe_load(f.read())
except (FileNotFoundError, yaml.YAMLError) as e:
    logger.error(f"Error loading application configuration: {e}")
    raise

load_dotenv()
openai.api_key = os.getenv("GPT_KEY")
if openai.api_key is None:
    logger.error("Missing environment variable: GPT_KEY")
    raise ValueError("Missing environment variable: GPT_KEY")

pinecone_key = os.getenv("PINECONE_KEY")
if pinecone_key is None:
    logger.error("Missing environment variable: PINECONE_KEY")
    raise ValueError("Missing environment variable: PINECONE_KEY")

pinecone.init(api_key=pinecone_key, environment="asia-northeast1-gcp")

index = pinecone.Index(pinecone.list_indexes()[0])

class StoreData:
    """
    A class to handle the preparation and uploading of different types of data to a pinecone database.

    Attributes:
        None

    Methods:
        store_service(service): Prepares, embeds, and uploads data for a specified service.
    """

    def __init__(self):
        """
        Initializes the StoreData instance.
        
        Initializes __data_prep and structure_funcs attributes.
        """
        self.__data_prep = PrepareData()
        self.__structure_funcs: Dict[str, Callable[[List[float], str], Tuple[str, List[float]]]] = {
            'matchups': self.__structure_matchup,
        }

    @staticmethod
    def __embed_data(data: List[Tuple[str, str]]) -> List[Tuple[List[float], str]]:        
        """
        Embeds the given data using OpenAI.

        Args:
            data (list): A list of data to be embedded.

        Returns:
            list: A list of tuples where the first element is the embedding vector and the second element is the ID of the data.
        """
        def embed_item(item):
            embed_model = app_config['models']['embed_model']
            text, id = item
            res = openai.Embedding.create(
                input=text, engine=embed_model
            )
            return res["data"][0]["embedding"], id
        embeddings = [embed_item(item) for item in data]
        return embeddings

    @staticmethod
    def __prepare_embeddings(embeddings: List[Tuple[List[float], str]], structure_id_func: Callable[[List[float], str], Tuple[str, List[float]]]) -> List[Tuple[str, List[float]]]:
        """    
        Prepares the embeddings for upload by structuring them and their IDs.

        Args:
            embeddings (list): A list of embeddings.
            structure_id_func (func): A function to structure the IDs for each embedding.

        Returns:
            list: A list of structured embeddings.
        """
        logger.info(f'Preparing embeddings')
        return [structure_id_func(embedding, id) for embedding, id in embeddings]
    
    @staticmethod
    def __upload_embeddings(embeddings: List[Tuple[str, List[float]]], service: str, batch_size: int = 100) -> None:
        """
        Uploads the embeddings to the specified service.

        Args:
            embeddings (list): A list of embeddings to upload.
            service (str): The name of the service to upload to.
            batch_size (int, optional): The number of embeddings to upload at once. Defaults to 100.
        """
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i+batch_size]
            try:
                index.upsert(vectors=batch, namespace=service)
            except Exception as e:
                logger.error(f"An error occurred while uploading embeddings: {e}")
                raise

    
    def __structure_service(self, embedding: List[float], id: str, service: str) -> Tuple[str, List[float]]:
        """
        Structures the ID and embedding for a given service.

        Args:
            embedding (list): The embedding to be structured.
            id (str): The ID of the embedding.
            service (str): The name of the service.

        Returns:
            tuple: A tuple where the first element is the structured ID and the second element is the embedding.
        """
        structured_id = f"{service}_{id}"
        return structured_id, embedding

    def __structure_matchup(self, embedding: List[float], id: str) -> Tuple[str, List[float]]:
        """
        Structures the ID and embedding specifically for the 'matchups' service.

        Args:
            embedding (list): The embedding to be structured.
            id (str): The ID of the embedding.

        Returns:
            tuple: A tuple where the first element is the structured ID and the second element is the embedding.
        """
        date_string = datetime.datetime.now().strftime('%B %d')
        return (f"matchup_{date_string}_{id}", embedding)
    
    def store_service(self, service: str) -> None:
        """
        Prepares, embeds, and uploads data for a specified service.

        Args:
            service (str): The name of the service to be processed and stored.
        """
        if not isinstance(service, str):
            raise ValueError(f"Expected string for 'service', got {type(service).__name__}")
        
        try:
            logger.info(f"preparing {service} data")
            data = self.__data_prep.prepare_service(service)

            if data is None:
                logger.info(f"No data to process for {service}")
                raise ValueError(f"No data to process for {service}")
            
            logger.info(f'embedding {service} data')
            data_embeddings = self.__embed_data(data)
            
            if data_embeddings is None:
                logger.error(f"No embeddings to store for {service}")
                return

            logger.info(f'preparing {service} embeddings')
            if service in self.__structure_funcs:
                data_embeddings = self.__prepare_embeddings(data_embeddings, self.__structure_funcs[service])
            else:
                data_embeddings = self.__prepare_embeddings(data_embeddings, lambda embedding, id: self.__structure_service(embedding, id, service))
            
            logger.info(f'uploading {service} embeddings')
            self.__upload_embeddings(data_embeddings, service)
            
            logger.info(f"{service} entries uploaded")
        except Exception as e:
            logger.error(f"An error occurred while storing {service} service: {e}")
            raise



