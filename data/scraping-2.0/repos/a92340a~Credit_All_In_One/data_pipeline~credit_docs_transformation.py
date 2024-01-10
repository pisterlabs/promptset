import os
import sys
import time
import json
import pytz
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pymongo
import configparser
from apscheduler.schedulers.background import BackgroundScheduler

import google.cloud.logging
from google.oauth2.service_account import Credentials
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_KEY')

sys.path.append('../Credit_All_In_One/')
from my_configuration import _get_mongodb
import my_logger
from data_pipeline.etl_utils import fetch_distinct_card_name, fetch_distinct_card_alias_name


config = configparser.ConfigParser()
config.read('config.ini')

# Advanced Python Scheduler
scheduler = BackgroundScheduler()

# datetime
taiwanTz = pytz.timezone("Asia/Taipei") 


if config['environment']['ENV'] == 'production':
    # GCP logging
    gcp_key = json.load(open(os.getenv("KEY")))
    credentials = Credentials.from_service_account_info(gcp_key)
    client = google.cloud.logging.Client(credentials=credentials)
    client.setup_logging()

    # create a logger
    dev_logger = logging.getLogger("data_pipeline:credit_docs_transformation:docs_comparing_and_embedding")
else:
    # create a logger
    dev_logger = my_logger.MyLogger('etl')
    dev_logger.console_handler()
    dev_logger.file_handler(datetime.now(taiwanTz).strftime('%Y-%m-%d'))



mongo_db = _get_mongodb()
persist_directory = './chroma_db'
embedding = OpenAIEmbeddings() # default: "text-davinci-003"


def _docs_refactoring(data: list, today=datetime.now(taiwanTz).strftime('%Y-%m-%d')) -> list:
    """
    Compare the difference in card_content between yesterday and today.
    :param data: The data about this card between these 2 days. For example: data = [{'_id': ObjectId('651bd25a646a02fa2d4205b1'), 'source': ..., 'create_dt': '2023-10-03'}, ...]
    """
    if data: 
        # fetch card_content between yesterday and today
        compare = dict()
        for i in range(len(data)):
            # For example: compare = {'2023-10-03': card_content_1, {'2023-10-02': card_content_2}
            compare[data[i]['create_dt']] = data[i]['card_content']
        
        # build the Document when data is newest information today, or do nothing
        if len(compare) == 2:
            # Case 1: compare[today] != compare[yesterday] -> build a new doc for today's update information:
            if compare[today] != compare[(datetime.strptime(today, '%Y-%m-%d')-timedelta(days=1)).strftime('%Y-%m-%d')]:
                for index, content in enumerate(data):
                    if content['create_dt'] == today:
                        docs = [Document(
                            page_content=content['card_name']+':'+content['card_content']+'。'+content['card_link'],
                            metadata={'bank': content['bank_name'], 'card_name': content['card_name']},
                        )]
                        dev_logger.info(json.dumps({'msg':'Build a new Document and ready for ChromaDB: {}'.format(content['card_name'])}))
                        return docs
            # Case 2: compare[today] = compare[yesterday] -> do nothing due to the same information:
            else:
                dev_logger.info(json.dumps({'msg':'The card_content is the same. Do nothing!: {}'.format(data[0]['card_name'])}))
        elif len(compare) == 1:
            # Case 3: compare[today] -> build a new doc for today's information:
            if data[0]['create_dt'] == today:
                docs = [Document(
                    page_content=data[0]['card_name']+':'+data[0]['card_content']+'。'+data[0]['card_link'],
                    metadata={'bank': data[0]['bank_name'], 'card_name': data[0]['card_name']},
                )]
                dev_logger.info(json.dumps({'msg':'Build a new Document and ready for ChromaDB: {}'.format(data[0]['card_name'])}))
                return docs
            # Case 4: compare[yesterday] -> do nothoing due to the archived information:
            else:
                dev_logger.info(json.dumps({'msg':'The card_content is depreciated. Do nothing!: {}'.format(data[0]['card_name'])}))
    else:
        # Case 5: no date about this card in these 2 days -> do nothing:
        dev_logger.warning(json.dumps({'msg':'No data.'}))
        

def _insert_into_chroma(card_name:str, docs, persist_directory=persist_directory):
    """
    Docs embedding and converting into vectors
    :param card_name: data source of the card name, 
    :param docs: card content
    """
    vectordb = Chroma.from_documents(documents=docs, embedding=embedding, 
                                     persist_directory=persist_directory) 
    vectordb.persist()
    vectordb = None
    dev_logger.info(json.dumps({'msg':'Finish inserting into ChromaDB {}.'.format(card_name)}))


def docs_comparing_and_embedding(collection:str="official_website", card_list:list=fetch_distinct_card_name(), today=datetime.now(taiwanTz).strftime('%Y-%m-%d')):
    mongo_collection = mongo_db[collection]
    distinct_card_names = card_list
    
    dev_logger.info(json.dumps({'msg':'Schedulely fetch docs...'}))
    for card in distinct_card_names:
        cursor = mongo_collection.find({
            "$and": [
                {'card_name': card}, 
                {'create_dt': {'$gte': (datetime.strptime(today, '%Y-%m-%d')-timedelta(days=1)).strftime('%Y-%m-%d'), '$lte': today}}
            ]
        })
        data_comparision = list(cursor) 

        new_docs = _docs_refactoring(data=data_comparision, today=today)
        if new_docs:
            _insert_into_chroma(card, new_docs)


def docs_comparing_and_embedding_manually(collection:str="official_website", card_list:list=fetch_distinct_card_alias_name()):
    mongo_collection = mongo_db[collection]
    distinct_card_names = card_list
    
    max_create_dt = mongo_collection.find_one(sort=[('create_dt', pymongo.DESCENDING)])['create_dt']
    dev_logger.info(json.dumps({'msg':'Manually fetch docs at {}...'.format(max_create_dt)}))
    
    # list: fetch_distinct_card_name(), tuple: fetch_distinct_card_alias_name()
    for card_tuple in distinct_card_names:
        cursor = mongo_collection.find({
            "$and": [
                {'card_name': card_tuple[0] if isinstance(card_tuple, tuple) else card_tuple}, 
                {'create_dt': max_create_dt}
            ]
        })
        data_latest = list(cursor)

        for index, content in enumerate(data_latest):    
            if isinstance(card_tuple, tuple):
                # loop for all card alias name without duplicate original card name
                for card_alias_name in card_tuple[1]:
                    if card_alias_name != card_tuple[0]:
                        new_docs = [Document(
                            page_content=content['card_name']+':'+content['card_content']+'。'+content['card_link'],
                            metadata={'bank': content['bank_name'], 'card_name': card_alias_name},
                        )]
                        dev_logger.info(json.dumps({'msg':'Build a new Document with card alias name and ready for ChromaDB: {}'.format(card_alias_name)}))
                        if new_docs:
                            _insert_into_chroma(card_alias_name, new_docs, persist_directory=persist_directory)
        
            else:
                new_docs = [Document(
                    page_content=content['card_name']+':'+content['card_content']+'。'+content['card_link'],
                    metadata={'bank': content['bank_name'], 'card_name': content['card_name']},
                )]
                dev_logger.info(json.dumps({'msg':'Build a new Document with original card name and ready for ChromaDB: {}'.format(content['card_name'])}))
                if new_docs:
                    _insert_into_chroma(card_tuple, new_docs, persist_directory=persist_directory)



if __name__ == '__main__':
    #scheduler.add_job(docs_comparing_and_embedding, "interval", minutes=5)
    scheduler.add_job(
        docs_comparing_and_embedding,
        trigger="cron",
        hour="0, 4, 8, 12, 16, 20",
        minute=10,
        timezone=pytz.timezone("Asia/Taipei"),
    )

    scheduler.start()
    dev_logger.info(json.dumps({'msg':'Scheduler started ...'}))


    while True:
        time.sleep(5)
