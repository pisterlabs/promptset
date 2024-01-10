
from dotenv import load_dotenv

from db import db_client




from projects.project_seeder import seed_projects
from projects.project_repository import initialize_projects_table, clean_project_data

from aiclients import openai_client
from  shared import config
from shared import logger

log = logger.get_logger(__name__)
def setup_database():
    db_client.initdb()
    #TOOD: Remove this  after everything is set
    #log.info("Dropping projects table")
    #clean_project_data()
    log.info("init projects table")
    initialize_projects_table()
    log.info("init projects table completed")
    seed_projects()
    log.info("seed projects table completed")




def init_app():
    load_dotenv()
    log.info("Loaded environment variables from .env file")
    log.info(f"DB_HOST: {config.get('PG_HOST')}")
    setup_database()
    log.info("Completed Db setup")
    openai_client.init_openai()
    log.info("App initialized successfully!")


def recreate_database():
    load_dotenv()
    log.info("Loaded environment variables from .env file")
    log.info(f"DB_HOST: {config.get('PG_HOST')}")
    db_client.initdb()
    # TOOD: Remove this  after everything is set
    log.info("Dropping projects table")
    clean_project_data()
    log.info("init projects table")
    initialize_projects_table()
    log.info("init projects table completed")
    seed_projects()
    log.info("seed projects table completed")
    log.info("Completed Db setup")
    openai_client.init_openai()
    log.info("App initialized successfully!")

