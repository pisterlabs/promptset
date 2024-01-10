from openai import OpenAI
from config import Config

def delete_thread():
    # Initialize the OpenAI client with your API key
    client = OpenAI()
    OPENAI_API_KEY = Config.OPENAI_API_KEY

    # Prompt the user to enter the thread ID
    thread_id = input("Please enter the thread ID to delete: ")

    # Attempt to delete the thread
    try:
        response = client.beta.threads.delete(thread_id)
        print("Thread deleted successfully.")
        print(response)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    delete_thread()