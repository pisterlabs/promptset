import openai
from dotenv import load_dotenv
import os

load_dotenv()

# configure the openai authorizaiton
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# configure the weather api
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")
