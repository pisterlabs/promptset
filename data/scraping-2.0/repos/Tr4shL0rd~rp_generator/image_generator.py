import os
import openai
from dotenv import load_dotenv
load_dotenv()
APIKEY = os.environ["APIKEY"]
openai.api_key = APIKEY
