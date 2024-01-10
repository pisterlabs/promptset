from langchain.sql_database import SQLDatabase
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine

from sql_analyzer.config import SNOWFLAKE, MYSQL, cfg

from log_init import logger


def sql_db_factory() -> SQLDatabase:
    if cfg.selected_db == SNOWFLAKE:
        snowflake_config = cfg.snow_flake_config
        schema = snowflake_config.snowflake_schema
        engine = create_engine(
            URL(
                account=snowflake_config.snowflake_account,
                user=snowflake_config.snowflake_user,
                password=snowflake_config.snowflake_password,
                database=snowflake_config.snowflake_database,
                schema=schema,
                warehouse=snowflake_config.snowflake_warehouse,
                host=snowflake_config.snowflake_host,
            )
        )
        return SQLDatabase(engine=engine, schema=schema)
    elif cfg.selected_db == MYSQL:
        return SQLDatabase.from_uri(cfg.db_uri, view_support=True)
    else:
        raise Exception(f"Could not create sql database factory: {cfg.selected_db}")


if __name__ == "__main__":
    logger.info("sql_db_factory")
    sql_database = sql_db_factory()
    logger.info("sql_database %s", sql_database)
