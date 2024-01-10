import time
from datetime import datetime, timedelta
import sys
from loguru import logger
import warnings
from openai.error import APIError, PermissionError, AuthenticationError, InvalidAPIType, ServiceUnavailableError
from src.config import ES_INDEX
from src.elasticsearch_utils import ElasticSearchClient
from src.xml_utils import GenerateXML

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    gen = GenerateXML()
    elastic_search = ElasticSearchClient()
    dev_urls = [
        "https://lists.linuxfoundation.org/pipermail/bitcoin-dev/",
        "https://lists.linuxfoundation.org/pipermail/lightning-dev/"
    ]

    current_date_str = None
    if not current_date_str:
        current_date_str = datetime.now().strftime("%Y-%m-%d")

    start_date = datetime.now() - timedelta(days=30)
    start_date_str = start_date.strftime("%Y-%m-%d")
    logger.info(f"start_data: {start_date_str}")
    logger.info(f"current_date_str: {current_date_str}")

    for dev_url in dev_urls:
        data_list = elastic_search.extract_data_from_es(ES_INDEX, dev_url, start_date_str, current_date_str)
        dev_name = dev_url.split("/")[-2]
        logger.info(f"TOTAL THREADS RECEIVED FOR - {dev_name}: {len(data_list)}")

        delay = 5
        count_main = 0

        while True:
            try:
                gen.start(data_list, dev_url)
                break
            except (APIError, PermissionError, AuthenticationError, InvalidAPIType, ServiceUnavailableError) as ex:
                logger.error(str(ex))
                logger.error(ex)
                time.sleep(delay)
                count_main += 1
                if count_main > 5:
                    sys.exit(ex)
