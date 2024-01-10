import os

from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
from sql_analyzer.log_init import logger

load_dotenv()


SNOWFLAKE = "snowflake"
MYSQL = "mysql"
SELECTED_DBS = [SNOWFLAKE, MYSQL]


class SnowflakeConfig:
    snowflake_account = os.getenv("SNOWFLAKE_ACCOUNT")
    snowflake_user = os.getenv("SNOWFLAKE_USER")
    snowflake_password = os.getenv("SNOWFLAKE_PASSWORD")
    snowflake_database = os.getenv("SNOWFLAKE_DATABASE")
    snowflake_schema = os.getenv("SNOWFLAKE_SCHEMA")
    snowflake_warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
    snowflake_host = os.getenv("SNOWFLAKE_HOST")


class Config:
    model = "gpt-3.5-turbo-16k-0613"
    # model = 'gpt-4-0613'
    llm = ChatOpenAI(model=model, temperature=0)
    db_uri = os.getenv("DB_CONNECTION_STRING")
    snow_flake_config = SnowflakeConfig()
    selected_db = os.getenv("SELECTED_DB")
    if selected_db not in SELECTED_DBS:
        raise Exception(
            f"Selected DB {selected_db} not recognized. The possible values are: {SELECTED_DBS}."
        )


cfg = Config()

if __name__ == "__main__":
    logger.info("LLM %s", cfg.llm)
    logger.info("db_uri %s", cfg.db_uri)
    logger.info("selected_db %s", cfg.selected_db)
