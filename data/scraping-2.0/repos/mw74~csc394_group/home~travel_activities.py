import logging
import sys
import openai
import os
from llama_index import GPTTreeIndex, GPTListIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
from IPython.display import Markdown, display
from langchain.chat_models import ChatOpenAI
from amadeus import Client, ResponseError
from dotenv import load_dotenv


def run_travel_activities():
    load_dotenv()

    api_key = os.getenv('OPENAI_API_KEY')
    
    # Choose model and tune
    llm_predictor = LLMPredictor(llm=ChatOpenAI(model_name="gpt-4",
                                                request_timeout=300,
                                                temperature=1.0,
                                                top_p=1.0,
                                                frequency_penalty=0.0,
                                                presence_penalty=-1.0,
                                                max_tokens=150,
                                                openai_api_key=api_key))

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

    # Load directory which includes prompt.txt file detailing travel agent role obligations and user data.
    documents = SimpleDirectoryReader('src/prompt/').load_data()  # Updated to pull data from User model

    # Embed information in chatGPT model.
    davinci_index = GPTListIndex.from_documents(documents, service_context=service_context)

    # Initial data retrieval request to ChatGPT for destination airport code.
    query1 = "Respond with a list containing the float values for latitude then longitude of vacation my destination, do not add any other text besides the list containing the data."

    # Send request, save response
    response1 = davinci_index.query(query1)
    
    amadeus = Client(
        client_id=os.getenv('AMADEUS_API_KEY'),
        client_secret=os.getenv('AMADEUS_API_SECRET')
    )

    li = list(response1.response.replace("[", "").replace("]", "").replace(" ", "").split(","))

    with open('src/response/Activities.txt', 'w') as f:
        f.write(amadeus.shopping.activities.get(latitude=li[0], longitude=li[1]).body)

    activity_log = open("src/response/Activities.txt", "r").read()
    return activity_log


