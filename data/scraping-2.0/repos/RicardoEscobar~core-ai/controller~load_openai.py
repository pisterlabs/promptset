import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
from elevenlabs import set_api_key


def load_openai():
    """Load the OpenAI API key from the .env file and return an OpenAI client"""
    # Load environment variables from .env file
    load_dotenv()

    # Create an OpenAI client
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Set the Eleven API key
    ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
    set_api_key(ELEVEN_API_KEY)

    # This is needed to load the path where the mpv player is located."C:\Users\Ricardo\Downloads\mpv.exe"
    # E:\\downloads (PC)
    # C:\\Users\\Ricardo\\Downloads (Laptop)
    os.environ["PATH"] += os.pathsep + os.getenv("MPV_PATH")

    # Add project root to PYTHONPATH
    root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root_folder not in sys.path:
        sys.path.append(root_folder)

    return client


if __name__ == "__main__":
    openai_client = load_openai()

