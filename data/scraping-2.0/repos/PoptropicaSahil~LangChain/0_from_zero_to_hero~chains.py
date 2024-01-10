import warnings


# Filter out UserWarnings - should come before the warning causing thing
warnings.filterwarnings("ignore", 
                        category=UserWarning
                        )


from dotenv import load_dotenv
import os
import logging 



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


from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

llm = OpenAI(model="text-davinci-003", temperature=0.9)

prompt = PromptTemplate(
    input_variables = ["product"],
    template = "What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
# result = chain.run("aluminium and copper wires")

user_input = input("What type of product ")
# user_input = "aluminium and copper wires"
result = chain.run(user_input)

# result = chain.run(input = "What type of product ") # does not work

logging.info(result)
print(result)