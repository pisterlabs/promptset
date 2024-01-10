import openai
import logging
import utils.audio
import utils.whisper
import utils.tts
import json
from utils.chatgpt_api import ChatGPTAPI  # Import the ChatGPTAPI class
import pdb

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_json_string(md_json_string):
    """
    Preprocesses a JSON string that is enclosed within Markdown backticks.

    Args:
        md_json_string (str): The JSON string with Markdown backticks.

    Returns:
        str: The cleaned JSON string.
    """
    # Split the string into lines
    print("preprocessing json string")
    lines = md_json_string.split('\n')

    # Remove the first and last lines if they contain backticks
    if lines[0].strip().startswith('```'):
        lines = lines[1:]
    if lines[-1].strip().endswith('```'):
        lines = lines[:-1]

    # Join the lines back into a single string
    cleaned_string = '\n'.join(lines).strip()

    return cleaned_string

def decode_json(json_string):
    try:
        data = json.loads(json_string)
        command = data.get("command")
        args = data.get("args", {})

        if command == "advice":
            print("Giving advice...")
            advice = args.get("content")
            # Call the advice handling function
            # Convert the response to speech
            utils.tts.text_to_speech(advice)
            logging.info("Text-to-speech conversion completed")
        elif command == "set_timer":
            print("Setting timer...")
            time = args.get("duration")
            # Convert the response to speech
            utils.tts.text_to_speech(f"Setting timer for {time} minutes")
            logging.info("Text-to-speech conversion completed")
        else:
            print(f"Unknown command: {command}")
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")

def read_api_key(file_path):
    """Read the API key from a file."""
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except IOError:
        logging.error("Unable to read API key. Check if the credentials.txt file exists and is readable.")
        return None

# Read API key from credentials.txt
api_key = read_api_key('credentials.txt')
if not api_key:
    logging.critical("API key not found. Exiting.")
    exit(1)

# Initialize ChatGPT API with the API key
chat_gpt_api = ChatGPTAPI(api_key)
# chat_gpt_api.add_system_message("You are KAI, a cooking assistant. Please give cooking advice to the user.")

# chat_gpt_api.add_system_message("""You are KAI, a cooking assistant. You only have two actions: advice and set_timer. You should only respond in JSON format as described below:
#                                 {
#                                 "command": "advice",
#                                 "parameters": {
#                                     "content": "The best temperature to cook a steak is medium rare"
#                                 },
#                                 }

#                                 or 
#                                 {
#                                 "command": "set_timer",
#                                 "parameters": {
#                                     "duration": "10 minutes",
#                                     "message": "Check the oven"
#                                 },
#                                 }
#                                 """)

chat_gpt_api.add_system_message("You are KAI, a cooking assistant. Please give cooking advice to the user. If giving the user a recipe, please ask the user if they would like to hear the steps one at a time. If they do, please provide one step of instructions until the user signals that they are ready for the next step.")
chat_gpt_api.add_system_message("""
These are your abilities:
ABILITIES = (
    'advice: Gives answers to the user, args: "code": "<full_code_string>"',
    'set_timer: Starts a timer, args: "duration": "<float, duration in minutes>"', "message": "<message>",
)
                                """)
chat_gpt_api.add_system_message("""
You should only respond in JSON format as described below:
                                {
                                "command": "advice",
                                "args": {
                                    "content": "The best temperature to cook a steak is medium rare"
                                },
                                }

                                or 
                                {
                                "command": "set_timer",
                                "args": {
                                    "duration": "10 minutes",
                                    "message": "Check the oven"
                                },
                                }
                                """)

# Main process
logging.info("Starting main process")

# File name for the recorded audio
file_name = "test.wav"

while(True):

    # # Record audio and save it to 'file_name'
    utils.audio.record_audio(file_name, 0.09, 2)
    logging.info("Audio recorded and saved as " + file_name)

    # Transcribe the recorded audio
    transcription = utils.whisper.transcribe_audio(file_name)
    logging.info("Transcription complete")

    # Log transcription
    logging.info("Transcription: " + transcription)

    # Send the transcription as a question to ChatGPT
    response = chat_gpt_api.ask_question(transcription)
    logging.info("Response received from ChatGPT")
    print("AFTER Response:")

    # Decode the JSON response
    response = preprocess_json_string(response)
    decode_json(response)

    # Log response
    logging.info("Response: " + response)

    # # Convert the response to speech
    # utils.tts.text_to_speech(response)
    # logging.info("SECOND Text-to-speech conversion completed")

    print("Press any key to talk to KAI...")
    input()


