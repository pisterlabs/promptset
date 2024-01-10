import openai
import os
import yaml
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception as e:
    pass

client = OpenAI()


class OpenAIAssistant:
    def __init__(self, api_key=None):
        """
        Initialize the Assistant Manager with an OpenAI API key.

        :param api_key: OpenAI API key. If None, it tries to fetch from environment variables.
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("API key is not set. Please provide an API key or set the OPENAI_API_KEY environment variable.")
        openai.api_key = self.api_key

    def create_assistant_from_yaml(self, yaml_file):
        """
        Creates an OpenAI assistant from a configuration specified in a YAML file.

        :param yaml_file: Path to the YAML file containing the assistant configuration.
        :return: The created Assistant object or None if creation failed.
        """
        config = self.load_yaml_config(yaml_file)
        if config is None:
            return None

        return self.create_assistant(config)

    def load_yaml_config(self, yaml_file):
        """
        Loads the YAML configuration file.

        :param yaml_file: Path to the YAML file.
        :return: Loaded configuration as a dictionary or None if failed.
        """
        try:
            with open(yaml_file, 'r') as stream:
                return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("Error loading YAML configuration:", exc)
        except FileNotFoundError:
            print(f"The file {yaml_file} was not found.")
        return None

    def create_assistant(self, config):
        """
        Creates an assistant with the given configuration.

        :param config: Configuration dictionary for the assistant.
        :return: The created Assistant object or None if creation failed.
        """
        try:
            assistant = client.beta.assistants.create(**config)
            print("Assistant created successfully:", assistant.id)
            return assistant
        except Exception as e:
            print("An error occurred while creating the assistant:", e)
        return None
    
    def create_thread(self, assistant_id, initial_message):
        """
        Creates a new thread with an initial message.

        :param assistant_id: ID of the Assistant to use.
        :param initial_message: Initial message content as a string.
        :return: The created Thread object or None if creation failed.
        """
        try:
            thread = client.beta.threads.create(
                assistant_id=assistant_id,
                messages=[{
                    "role": "user",
                    "content": initial_message
                }]
            )
            print("Thread created successfully:", thread.id)
            return thread
        except Exception as e:
            print("An error occurred while creating the thread:", e)
        return None

    def chat_with_assistant(self, thread_id, assistant_id, user_message):
        """
        Appends a message to the thread and runs the assistant to get a response.

        :param thread_id: ID of the Thread to chat in.
        :param assistant_id: ID of the Assistant to use.
        :param user_message: User's message content as a string.
        :return: Response from the Assistant or None if an error occurred.
        """
        try:
            # Append the user's message to the thread
            client.beta.threads.messages.create(
                thread_id=thread_id,
                message={
                    "role": "user",
                    "content": user_message
                }
            )

            # Run the assistant to generate a response
            run = client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id
            )
            return run.messages[-1].content if run.messages else None
        except Exception as e:
            print("An error occurred while chatting with the assistant:", e)
        return None