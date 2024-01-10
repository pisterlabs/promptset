import os
import server
import openai
import asyncio
import aioconsole
import logging
from aiohttp import ClientSession
import puremagic
from fastapi import FastAPI, HTTPException
from typing import List
from prompting import system_message
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from fastapi.responses import JSONResponse
from azure.search.documents.indexes.models import (
    ComplexField,
    CorsOptions,
    SearchIndex,
    ScoringProfile,
    SearchFieldDataType,
    SimpleField,
    SearchableField
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

app = FastAPI()

preset_questions = [
    "What activity is she currently engaged in?",
    "Describe her facial expression or mood:",
    "How would you describe her style today?",
    "What are notable aspects of her attire or accessories?",
    "What initially caught your attention about her?",
    "How does her voice sound, if you've heard it?",
    "How is she positioned in the setting?",
    "Can you guess her current emotional state?",
    "Do you observe any interesting non-verbal cues?",
    "Any additional insights not captured in the photo or above questions?"
]

async def async_input(prompt: str = "") -> str:
    try:
        return await aioconsole.ainput(prompt)
    except Exception as e:
        logger.error(f"Error during input: {str(e)}")
        return ""

def get_answer(question: str) -> str:
    print(question)
    answer = input("Your answer: ")
    return answer

search_service_name = os.getenv('search_service_name')
endpoint = "https://wingmandatabase.search.windows.net"
admin_key = os.getenv('admin_key')
search_index_name = "wingmanindex"
index_name = "wingmanindex"
openai_api_key = os.environ['OPENAI_API_KEY']

search_client = SearchClient(endpoint=endpoint,
                             index_name=search_index_name,
                             credential=AzureKeyCredential(admin_key))

def log_request_data(request_data):
    """
    Log the request data being sent to OpenAI.
    """
    logger.info(f"Sending request to OpenAI: {request_data}")

def log_truncated_data(original_data, truncated_data):
    """
    Log the original and truncated data.
    """
    logger.info(f"Original request data: {original_data}")
    logger.info(f"Truncated request data: {truncated_data}")

async def search_chunks(query):
    try:
        # Logging the query for debugging
        logger.info(f"Searching database for: {query}")

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, lambda: [result for result in search_client.search(search_text=query, top=8)])
        return results
    except Exception as e:
        logger.error(f"Error searching chunks: {str(e)}")
        return []

async def insert_chunks(chunks):
    try:
        actions = [{"@search.action": "upload", "id": str(i+1), "content": chunk} for i, chunk in enumerate(chunks)]
        search_client.upload_documents(documents=actions)
    except Exception as e:
        logger.error(f"Error inserting chunks: {str(e)}")
        return []

@app.post("/uploadedimage")
async def receive_uploaded_image(image: dict):
    situation = ""
    history = []
    try:
        image_url = image.get('image_url', '')
        image_name = image.get('file_name', '')
        if not image_url or not image_name:
            raise ValueError("Invalid image data received.")

        image_description_task = asyncio.create_task(describe_image(image_url))

        image_description = await image_description_task

        history.append({"role": "assistant", "content": "I see the following in the image: " + image_description})
        situation = image_description + " " + situation

    except Exception as e:
        logger.error(f"Error receiving uploaded image: {str(e)}")
    return {"situation": "", "history": []}

async def describe_image(image_url):
    try:
        subscription_key = os.environ['COMPUTER_VISION_KEY']
        endpoint = os.environ['COMPUTER_VISION_ENDPOINT']
        computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

        loop = asyncio.get_running_loop()

        analysis = await loop.run_in_executor(None, lambda: computervision_client.analyze_image(
            image_url,
            visual_features=[
                VisualFeatureTypes.objects,
                VisualFeatureTypes.tags,
                VisualFeatureTypes.description,
                VisualFeatureTypes.color,
            ]
        ))
        if (len(analysis.description.captions) == 0):
            image_description = "No description detected."
        else:
            image_description = analysis.description.captions[0].text

        objects = [obj.object_property for obj in analysis.objects]
        tags = [tag.name for tag in analysis.tags]
        dominant_colors = analysis.color.dominant_colors

        return f"I see the following in the image: {image_description}. The image contains these objects: {', '.join(objects)}. The image has these dominant colors: {', '.join(dominant_colors)}."

    except Exception as e:
        logger.error(f"Error describing image: {str(e)}")
        return "Unable to process image."

async def process_question_answer(question: str, answer: str):
    history = []
    situation = ""

    logging.info(question)
    history.append({"role": "assistant", "content": question})
    history.append({"role": "user", "content": answer})
    situation += answer + " "

    return situation, history

def select_top_pickup_lines(response, num_lines):
    pickup_lines = []
    for choice in response['choices']:
        line = choice['message']['content']
        pickup_lines.append(line)
        if len(pickup_lines) >= num_lines:
            break
    return pickup_lines

async def generate_pickup_lines(situation, answers, history=[], num_lines=5):
    logger.debug(f"Initial situation in generate_pickup_lines: {situation}")

    relevant_data = await search_chunks(situation)
    logger.info(f"Relevant Data Chunks from DB: {relevant_data}")

    if relevant_data:
        for data in relevant_data:
            # Logging each data chunk structure for clarity
            logger.info(f"Data chunk structure: {data}")
            chunk_content = data.get('content', None)
            if chunk_content:
                logger.info(f"Appending chunk: {chunk_content}")
                situation += chunk_content + " "
            else:
                logger.warning(f"No content found in chunk with key 'content'.")
    else:
        logger.warning(f"No relevant data chunks retrieved for the situation: {situation}")

    logger.info(f"Situation after adding chunks: {situation}")

    # Debugging: print out the answers list
    logger.info(f"Answers list: {answers}")

    # Add answers to the situation
    for idx, answer in enumerate(answers):
        question = preset_questions[idx]
        situation += question + " " + answer + " "

    logger.info(f"Updated Situation after adding questions and answers: {situation}")

    messages = [
        {
            "role": "assistant",
            "content": f"{system_message}\nSituation: {situation}\nGenerate {num_lines} pickup lines:"
        }
    ]

    # Before sending the request, log the request data
    request_data = {
        "model": "gpt-4",
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.15,
        "n": 1,
    }
    logger.info(f"Sending request to OpenAI: {request_data}")

    retry_attempts = 3
    retry_delay = 2

    for attempt in range(retry_attempts):
        try:
            response = openai.ChatCompletion.create(**request_data)

            pickup_lines = select_top_pickup_lines(response, num_lines)

            return pickup_lines, history

        except openai.error.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {str(e)}")
            await asyncio.sleep(retry_delay)
        except Exception as e:
            logger.error(f"Error generating pickup lines: {str(e)}")
            return ["Error generating pickup lines."]
        
async def ask_preset_questions(session_id: str):
    try:
        situation = ""
        history = []
        
        # Iterate over preset questions
        for _ in range(len(server.preset_questions)):
            # Ask question
            question_response = await server.ask_question(session_id)
            question = question_response["question"]
            
            # Assume that we have a function get_answer that gets the answer to a question
            # You need to implement this function based on how you want to get the answer in your application
            answer = get_answer(question)
            
            # Post answer and process it
            post_answer_response = await server.post_answer(answer, session_id)
            # Assume that new_situation and new_history are returned in the response
            new_situation = post_answer_response["situation"]
            new_history = post_answer_response["history"]
            
            # Update situation and history
            situation += " " + new_situation
            history.extend(new_history)
        
        return situation, history
    except Exception as e:
        logging.error(f"Unexpected error occurred: {str(e)}")
        return JSONResponse(status_code=500, content={"message": f"Unexpected error occurred: {str(e)}"})

async def process_user_query(query, history, pickup_lines, questions, answers):
    # Logging for debugging purposes
    logger.debug("process_user_query function called.")
    logger.debug(f"Received query: {query}")
    logger.debug(f"Received history: {history}")
    logger.debug(f"Received questions: {questions}")
    logger.debug(f"Received answers: {answers}")

    # Ensure that the parameters are of the expected types
    if not isinstance(history, list):
        history = []
    if not isinstance(pickup_lines, list):
        pickup_lines = []
    if not isinstance(questions, list):
        questions = []
    if not isinstance(answers, list):
        answers = []

    # Constructing the situation from the questions, answers, and pickup lines
    situation = "Here is more of the conversation history:\n"
    for q, a in zip(questions, answers):
        situation += f"{q}: {a}\n"

    for pl in pickup_lines:
        situation += pl + "\n"
    
    system_message = {
        "role": "system",
        "content": "Drawing upon the provided conversation history, which encompasses pickup lines, user questions, and answers, craft a response to the user's command. As the world's premier dating assistant, your expertise lies in generating captivating pickup lines, dispensing invaluable dating advice, and offering insightful tips to men. Always remain attuned to the context and nuances of previous interactions to deliver the most fitting and impactful response."
    }

    # Conversation History with explicit labeling for pickup lines
    pickup_line_label = {
        "role": "assistant",
        "content": "These are the pickup lines:"
    }
    history_message = {
        "role": "assistant",
        "content": situation.strip()
    }

    # User Command
    user_command_message = {
        "role": "user",
        "content": "Here is the user's command, please answer this while taking into consideration the conversation history you are provided with: " + query
    }

    # Combine all messages
    all_messages = [system_message, pickup_line_label] + history + [user_command_message]

    # Log the messages being sent to OpenAI for debugging purposes
    logger.debug(f"Sending the following messages to OpenAI: {all_messages}")
    
    openai.api_key = openai_api_key
    
    # Send Messages to GPT-3 Chat Model
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=all_messages,
            max_tokens=400,
            n=1,
            temperature=0.15
        )

        assistant_message = response['choices'][0]['message']['content']
        history.append({"role": "assistant", "content": assistant_message})
    except openai.error.RateLimitError as e:
        logger.error(f"Rate limit exceeded: {str(e)}")
        assistant_message = "Rate limit exceeded."
    except Exception as e:
        logger.error(f"Error processing user query: {str(e)}")
        assistant_message = "Unable to process query."

    return assistant_message, history

def save_history_to_file(history):
    try:
        with open('history.txt', 'w') as file:
            for item in history:
                file.write(f"{item['role']}: {item['content']}\n")
    except Exception as e:
        logging.error(f"Error saving history to file: {str(e)}")

async def main():
    try:
        situation, history = await ask_preset_questions("1234")

        pickup_lines, history = await generate_pickup_lines(situation, history, answers, 5)
        logging.info("\n".join(pickup_lines))
        while True:
            query = await async_input("\nEnter your query: ")
            response, history = await process_user_query(query, history, pickup_lines, preset_questions, answers) # I've assumed you'll provide answers here, please adjust if it's different.
            logging.info(response)
    except KeyboardInterrupt:
        logging.info("\nInterrupted by user. Exiting.")
        save_history_to_file(history)
        exit()

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
