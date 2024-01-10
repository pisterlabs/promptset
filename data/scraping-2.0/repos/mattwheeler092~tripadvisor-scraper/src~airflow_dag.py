import logging
import traceback
import asyncio

from airflow import DAG
from airflow.operators.python import PythonOperator

from cache.city_cache import CityCache
from scraper.scraper import TripAdvisorScraper
from gcp_client import GCP_Client
from openai_embedding import get_text_embeddings
from utils import kill_airflow_job

from pinecone_db import (
    create_pinecone_data,
    pinecone_upsert_data
)
from config import (
    AIRFLOW_CITIES_PER_JOB,
    AIRFLOW_DAG_ID,
    AIRFLOW_SCHEDULE,
    AIRFLOW_START_DATE
)
logging.basicConfig(level=logging.INFO)


def process_cities():
    """ Airflow DAG function to process cities """
    
    cache = CityCache(AIRFLOW_CITIES_PER_JOB)
    if cache.cache_complete():
        kill_airflow_job(AIRFLOW_DAG_ID)
        return
    
    client = GCP_Client()
     
    for city_info in cache:

        try:
            logging.info(
                f"PROCESSING: {city_info['city']} ({city_info['country']})"
            )
            scraper = TripAdvisorScraper(city_info)
            loop = asyncio.get_event_loop()
            attr_details = loop.run_until_complete(scraper.get_attr_details())

            logging.info(
                f"SCRAPED {len(attr_details)} / {scraper.max_attr} ATTRACTIONS"
            )
            embeddings = get_text_embeddings(attr_details)
            pinecone_data = create_pinecone_data(attr_details, embeddings)

            logging.info(f"GENERATED EMBEDDING DATA")

            filename = f"backups/{city_info['namespace']}.json"
            client.upload_file(pinecone_data, filename)

            logging.info(f"UPLOADED BACKUP TO GCP")

            pinecone_upsert_data(pinecone_data, city_info["namespace"])

            logging.info(f"UPLOADED EMBEDDINGS TO PINECONE\n")

        except Exception as err:
            with open(f"error_log/{city_info['namespace']}.txt", "w") as file:
                file.write(traceback.format_exc())
                file.write(repr(err))


with DAG(dag_id=AIRFLOW_DAG_ID,
         description="ETL pipeline for historical weather collection",
         start_date=AIRFLOW_START_DATE,
         default_args={'depends_on_past': False},
         schedule=AIRFLOW_SCHEDULE) as dag:
    
    task1 = PythonOperator(
        python_callable=process_cities,
        task_id="task1"
    )
    task1
