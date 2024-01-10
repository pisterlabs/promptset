import os
from dotenv  import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage
)
 
## GENERAL CONFIGURATION
# Specify the path to the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')

# Load the .env file
load_dotenv(dotenv_path)

# Set the OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize model
chat_model: ChatOpenAI = ChatOpenAI(openai_api_key=openai_api_key)

sys_msg = SystemMessage(content = "You are an expert mentor for students who values self-regulated learning and its benefits for your mentee. You will assist the student with reflecting on what they learned last week. Your name is Sigma. Do not make-up answers.")
human_msg = HumanMessage(content = "Last week I read about self-regulated learning. I think it's a process that students implement to control their speed of learning. Is that right?")

prediction_msg = chat_model.predict_messages([human_msg, sys_msg])

print(prediction_msg.content)