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
activeloop_key = os.getenv("ACTIVELOOP_TOKEN")
activeloop_org_id = os.getenv("ACTIVELOOP_ORG_ID")


# logging.INFO is an integer in itself
# use logging.info instead
# logging.info(f"OpenAI key is {openai_key}")
logging.info("hi there check hello")


######################################################
######################################################
### Basic Prompting

llm = OpenAI(model_name="text-davinci-003", temperature=0.9)

# text = "What would be a good company name for a company that makes colorful socks?"
text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."

result = llm(text)
logging.info(result)
print(result)
