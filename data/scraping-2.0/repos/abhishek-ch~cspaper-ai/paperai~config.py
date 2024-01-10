import configparser
import os
import openai

def load_config(file_path: str) -> None:
    """
     read the configuration file
    """
    config = configparser.ConfigParser()
    config.read(file_path)
    config.sections()
    if config["azure"]["openai_key"]:
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_BASE"] = config['azure']['endpoint_url']
        os.environ["OPENAI_API_KEY"] = config['azure']['openai_key']
        os.environ["OPENAI_API_VERSION"] = config['azure']['api_version']


        openai_key = config['azure']['openai_key']
        endpoint_url = config['azure']['endpoint_url']
        api_version = config['azure']['api_version']

        openai.api_type = "azure"
        openai.api_key = openai_key
        openai.api_base = endpoint_url
        openai.api_version = api_version

    elif config["openai"]["openai_key"]:
        openai.api_key = config["openai"]["openai_key"]

    if config["snowflake"]:
        SNOWFLAKE_USER = config['snowflake']['SNOWFLAKE_USER']
        SNOWFLAKE_PASSWORD = config['snowflake']['SNOWFLAKE_PASSWORD']
        SNOWFLAKE_ACCOUNT_IDENTIFIER = config['snowflake']['SNOWFLAKE_ACCOUNT_IDENTIFIER']

        SNOWFLAKE_DATABASE = config['snowflake']['SNOWFLAKE_DATABASE']
        SNOWFLAKE_SCHEMA = config['snowflake']['SNOWFLAKE_SCHEMA']
        SNOWFLAKE_WAREHOUSE = config['snowflake']['SNOWFLAKE_WAREHOUSE']
        os.environ["SNOWFLAKE_URL"] = f"snowflake://{SNOWFLAKE_USER}:{SNOWFLAKE_PASSWORD}@{SNOWFLAKE_ACCOUNT_IDENTIFIER}/{SNOWFLAKE_DATABASE}/{SNOWFLAKE_SCHEMA}?role=ASCENT_RW&warehouse={SNOWFLAKE_WAREHOUSE}"


def set_api_key() -> None:
    openai.api_key = os.getenv("OPENAI_API_KEY")

db_persistent_path = "/Users/gmcjy/Documents/learning/vectordb/qdrant/data"
collection_name = "test_papers"
pdf_uploadpath = "/Users/gmcjy/Documents/Backup/test/"