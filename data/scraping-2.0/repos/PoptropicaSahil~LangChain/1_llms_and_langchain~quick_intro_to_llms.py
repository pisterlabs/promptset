import warnings


# Filter out UserWarnings - should come before the warning causing thing
warnings.filterwarnings("ignore", 
                        category=UserWarning
                        )


from dotenv import load_dotenv
import os
import logging 

from langchain.llms import OpenAI



# Logging Configuration
logging.basicConfig(filename='langchain.log', 
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# load env variables
load_dotenv('.env')

# load API key from environment
openai_key = os.getenv("OPENAI_API_KEY")

# logging.INFO is an integer in itself
# use logging.info instead
logging.info(f"OpenAI key is {openai_key}")
logging.info("hi there")

llm = OpenAI(model_name="text-davinci-003", temperature=0)

text = "What would be a good company name for a company that makes colorful socks?"

result = llm(text)
logging.info(result)