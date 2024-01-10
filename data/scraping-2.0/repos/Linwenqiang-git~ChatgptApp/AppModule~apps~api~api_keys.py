import openai
from  utils.logger import logger

def set_api_key(api_key):         
    openai.api_key = api_key
    logger.info(f"set openai key success:{api_key}")    
