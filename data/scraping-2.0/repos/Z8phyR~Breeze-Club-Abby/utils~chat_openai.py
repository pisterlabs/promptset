import openai
import time
from dotenv import load_dotenv
import os
import utils.mongo_db as mongo_db
from Commands.Admin.persona import get_persona, get_persona_by_name
from utils.log_config import setup_logging, logging

setup_logging()
logger = logging.getLogger(__name__)


# Load the environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
model = "gpt-3.5-turbo"
client = mongo_db.connect_to_mongodb()

# System Prompts
SYSTEM_HELPER = '''
I'm currently talking to {name}, who like these genres: {genre}. I know that their music influences are: {influences}. More about {name}: {description}
'''

CODER_HELPER = '''
I'm Abby, A coding expert and virtual assistant for the Breeze Club Discord!,
i will randomly insert words like: "*hops around*", "*munches on carrot*" or "*exploring the outdoors*" and other similar words and emojis in my response to match my bunny persona!
'''

NO_PROFILE = "This user has not created a profile yet."


def chat(user, user_id, chat_history=[]):
    profile = mongo_db.get_profile(user_id)
    personality_doc = mongo_db.get_personality()
    PERSONALITY_NUMBER = personality_doc['personality_number'] if personality_doc else 0.6
    active_persona_doc = get_persona()
    active_persona = active_persona_doc['active_persona'] if active_persona_doc else 'bunny'
    persona_message = get_persona_by_name(active_persona)['persona_message']

    try:
        messages = []

        if profile is None:
            system_helper_message = NO_PROFILE
        else:
            system_helper_message = SYSTEM_HELPER.format(
                name=profile['name'], description=profile['description'], genre=profile['genre'], influences=profile['influences'])

        messages.append({"role": "system", "content": persona_message})
        messages.append({"role": "system", "content": system_helper_message})
        for message in chat_history[-4:]:
            messages.append({"role": "user", "content": message["input"]})
            messages.append(
                {"role": "assistant", "content": message["response"]})

        messages.append({"role": "user", "content": user})

        # Print out the messages before sending to OpenAI
        # for message in messages:
        #     logging.debug(f"Role: {message['role']}, Content: {message['content']}")
        retry_count = 0
        while retry_count < 3:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=PERSONALITY_NUMBER
                )
                status_code = response["choices"][0]["finish_reason"]
                assert status_code == "stop", f"The status code was {status_code}."
                return response["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(
                    f"An error occurred while processing the chat request: {str(e)}")
                logger.info("Retrying...")
                time.sleep(1)
                retry_count += 1

        return "Oops, something went wrong. Please try again later."

    except Exception as e:
        logging.warning(
            f"An error occurred while processing the chat request: {str(e)}")
        return "Oops, something went wrong. Please try again later."


def summarize(chat_session):
    retry_count = 0
    while retry_count < 3:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system",
                        "content": "Summarize this chat session between Abby and the user"},
                    {"role": "assistant", "content": f"{chat_session}"}
                ],
                max_tokens=300,
                temperature=0
            )

            status_code = response["choices"][0]["finish_reason"]
            assert status_code == "stop", f"The status code was {status_code}."
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logging.warning(
                f"An error occurred while processing the summarize request: {str(e)}")
            logging.info("Retrying...")
            time.sleep(1)
            retry_count += 1

    return "Oops, something went wrong. Please try again later."


def analyze(user, chat_session):
    retry_count = 0
    while retry_count < 3:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system",
                        "content":  f"Perform a detailed analysis and summarize the key points from these messages and provide feedback for {user}. Provide actionable recommendations to improve their idea's effectiveness for the betterment of the Breeze Club Discord Server."},
                    {"role": "assistant", "content": f"{chat_session}"}
                ],
                max_tokens=3000,
                temperature=0.3
            )
            status_code = response["choices"][0]["finish_reason"]
            assert status_code == "stop", f"The status code was {status_code}."
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(
                f"An error occurred while processing the summarize request: {str(e)}")
            logger.info("Retrying...")
            time.sleep(1)
            retry_count += 1

    return "Oops, something went wrong. Please try again later."


def chat_gpt4(user, user_id, chat_history=[]):
    messages = [
        {"role": "system", "content": CODER_HELPER},
        *[
            {"role": "user", "content": message["input"]}
            for message in chat_history[-8:]
        ],
        *[
            {"role": "assistant", "content": message["response"]}
            for message in chat_history[-8:]
        ],
        {"role": "user", "content": user}
    ]

    retry_count = 0
    while retry_count < 3:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0
            )
            status_code = response["choices"][0]["finish_reason"]
            assert status_code == "stop", f"The status code was {status_code}."
            response = response["choices"][0]["message"]["content"]
            response = f"[Code Abby]: {response}"
            return response
        except Exception as e:
            logging.warning(
                f"An error occurred while processing the chat(GPT4) request: {str(e)}")
            logging.info("Retrying...")
            time.sleep(1)
            retry_count += 1

    return "Oops, something went wrong. Please try again later."

class ChatBot:
    def __init__(self, api_key, model="gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key

    def get_response(self, prompt, type_of_prompt, max_tokens=100):
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": type_of_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error: {str(e)}"

