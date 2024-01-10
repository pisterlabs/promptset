from meow.ai.openai import OpenaiHandler
from meow.baidu.baidu import BaiduHandler
from meow.audio.record import RecordHandler
from meow.database.db import DBHandler

from meow.utils.conf import get_conf_data, get_key_data

import meow.utils.context as mc

import logging

def generate_handler():
    mc.set_record_handler(None)
    mc.set_baidu_handler(None)
    mc.set_record_handler(None)
    conf_data = get_conf_data()
    key_data = get_key_data()

    # logging.debug(conf_data)
    
    openai_config = conf_data['openai']
    baidu_config = conf_data['baidu']
    audio_config = conf_data['audio']
    openai_api_key = key_data['OPENAI_API_KEY']
    baidu_key = key_data['BAIDU_KEY']
    
    logging.info('load conf data finished')


    retry_conf = conf_data['retry']
    mc.set_retry_conf(retry_conf['timewait'], retry_conf['max_retry_times'])
    logging.info('[1/5]Retry config set')


    record = RecordHandler(**audio_config)
    mc.set_record_handler(record)
    logging.info('[2/5]RecordHandler registered')

    
    openai = OpenaiHandler(openai_api_key, **openai_config)
    mc.set_openai_handler(openai)
    logging.info('[3/5]OpenaiHandler registered')


    baidu = BaiduHandler(*baidu_key, **baidu_config)
    mc.set_baidu_handler(baidu)
    logging.info('[4/5]BaiduHandler registered')


    db = DBHandler('database.sqlite')
    mc.set_db_manager(db)
    logging.info('[5/5]DBHandler registered')
    
    

    