import yaml
from pathlib import Path
from contextlib import ExitStack
from realtime_ai_character.utils import Singleton, Character
from realtime_ai_character.database.chroma import get_chroma
from realtime_ai_character.logger import get_logger

from llama_index import SimpleDirectoryReader
from langchain.text_splitter import CharacterTextSplitter


logger = get_logger(__name__)


class CatalogManager(Singleton):
    def __init__(self, overwrite=True):
        super().__init__()

        self.db = get_chroma()

        if overwrite:
            logger.info('Overwriting existing data in the chroma.')
            self.db.delete_collection()
            self.db = get_chroma()

        self.characters = {}
        self.author_name_cache = {}
        self.load_characters(overwrite=overwrite)

        if overwrite:
            logger.info('Persisting data in the chroma.')
            self.db.persist()
        logger.info(
            f"Total document load: {self.db._client.get_collection('llm').count()}")
        
        # query = "Elon Musk unique human ID "
        # docs = self.db.similarity_search(query)
        # print(docs[0].page_content)


    def get_character(self, name) -> Character:
        return self.characters[name]

    def load_characters(self, overwrite):
        path = Path(__file__).parent
        # path = path / 'character'
        excluded_dirs = {"__pycache__"}
        directories = [d for d in path.iterdir() if d.is_dir()
                        and d.name not in excluded_dirs]
        # print(f'{directories}')
        for directory in directories:
                character_name = self.load_character(directory)
                # if overwrite:
                #     self.load_data(character_name, directory / 'data')
                #     logger.info('Loaded data for character: ' + character_name)
                logger.info('Loaded data for character: ' + character_name)


    def load_character(self, directory):
            with ExitStack() as stack:
                f_yaml = stack.enter_context(open(directory / 'config.yaml', encoding= 'utf-8'))
                yaml_content = yaml.safe_load(f_yaml)

            character_id = yaml_content['character_id']
            character_name = yaml_content['character_name']
            character_notification =  yaml_content['notification']

            self.characters[character_id] = Character(
                character_id=character_id,
                name=character_name,
                llm_system_prompt=yaml_content["system"],
                llm_user_prompt=yaml_content["user"],
                notification= character_notification
            )
            return character_name
    def load_data(self, character_name: str, data_path: str):
        # load all documents in /data
        # Reference: https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader.html
        loader = SimpleDirectoryReader(Path(data_path))
        documents = loader.load_data()

        text_splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=500,
            chunk_overlap=100)

        docs = text_splitter.create_documents(
            texts=[d.text for d in documents],
            metadatas=[{
                'character_name': character_name,
                'id': d.id_,
            } for d in documents])
        print(docs)
        self.db.add_documents(docs)
        

def get_catalog_manager():
    return CatalogManager.get_instance()