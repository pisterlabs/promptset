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



from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = OpenAI(model="text-davinci-003", temperature=0)

# Memory, such as ConversationBufferMemory, 
# acts as a wrapper around ChatMessageHistory, 
# extracting the messages and providing them to the chain for better context-aware generation.

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

# Start the conversation
conversation.predict(input="Tell me about yourself.")

# Continue the conversation
conversation.predict(input="What can you do?")
conversation.predict(input="How can you help me with data analysis?")

logging.info(conversation)

# Display the conversation
print(conversation)