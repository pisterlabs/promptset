import dotenv
from backend import handle_request
import openai


def text_interface(debug, speak):
    # Load the environment variables
    dotenv.load_dotenv()

    # Set the OpenAI API key
    API_KEY = dotenv.get_key(dotenv.find_dotenv(), "OPENAI_API_KEY")

    openai.api_key = API_KEY

    while True:
        message = input("> ")
        message = message.strip()

        # if the message is empty, don't send it to the API
        if message == "":
            continue

        if message == "exit":
            print("Thank you for using Jarvis!")
            break

        # send the message to the backend
        handle_request(message, debug=debug, speak=speak)