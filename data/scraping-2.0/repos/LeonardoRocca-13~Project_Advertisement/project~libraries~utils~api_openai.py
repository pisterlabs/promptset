from langchain.chat_models import ChatOpenAI
import os

from libraries.utils.settings import OPENAI_API_KEY_FILE_NAME, RESOURCES_FOLDER_NAME
from libraries.utils.get_path import get_path


def get_openai_model():
    # Get the API key for OpenAI from the file 'openai_api_key.txt' in the resources folder
    main_dir_path = get_path()
    file_path = os.path.join(main_dir_path, RESOURCES_FOLDER_NAME, OPENAI_API_KEY_FILE_NAME)
    try:
        with open(file_path, 'r') as key_file:
            api_key = key_file.readline()
    except FileNotFoundError:
        print("Please enter your OpenAI API key in the file 'openai_api_key.txt' in the project/resources folder.")
        exit(1)

    os.environ["OPENAI_API_KEY"] = api_key

    # Define the large language model and the relative temperature
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=1
    )

    return llm
