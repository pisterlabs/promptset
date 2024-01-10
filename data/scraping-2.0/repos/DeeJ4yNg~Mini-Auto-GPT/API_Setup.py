import os
import openai

os.environ["OPENAI_API_KEY"] = ''
openai.api_key  = os.getenv('OPENAI_API_KEY')
