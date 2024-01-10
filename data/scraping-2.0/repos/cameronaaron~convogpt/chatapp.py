import os
import openai
import json
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)

class DataStore(ABC):
    """Abstract base class for data storage."""

    @abstractmethod
    def load(self, user_id):
        """Load data for a user."""
        pass

    @abstractmethod
    def save(self, user_id, data):
        """Save data for a user."""
        pass

class JsonDataStore(DataStore):
    """Class for JSON file data storage."""

    def __init__(self, filename):
        self.filename = filename

    def load(self, user_id):
        """Load data for a user from a JSON file."""
        try:
            with open(self.filename, 'r') as f:
                all_data = json.load(f)
                return all_data.get(user_id, {})
        except (IOError, json.JSONDecodeError):
            return {}

    def save(self, user_id, data):
        """Save data for a user to a JSON file."""
        try:
            with open(self.filename, 'r') as f:
                all_data = json.load(f)
        except (IOError, json.JSONDecodeError):
            all_data = {}

        all_data[user_id] = data

        with open(self.filename, 'w') as f:
            json.dump(all_data, f)

class ChatApp:
    """Main application class."""

    def __init__(self, user_id, datastore):
        self.user_id = user_id
        self.datastore = datastore
        self.load_data()

    def load_data(self):
        """Load user data from data store."""
        try:
            data = self.datastore.load(self.user_id)
            self.user_profile = data.get('user_profile', {})
            self.conversation_history = data.get('conversation_history', [])
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            self.user_profile = {}
            self.conversation_history = []

    def save_data(self):
        """Save user data to data store."""
        data = {
            'user_profile': self.user_profile,
            'conversation_history': self.conversation_history
        }
        try:
            self.datastore.save(self.user_id, data)
        except Exception as e:
            logging.error(f"Error saving data: {e}")

    def generate_response(self, user_message, message_type):
        """Generate a response using OpenAI's GPT-3.5-turbo model."""
        try:
            if not user_message or not message_type:
                raise ValueError("User message and message type are required.")
            instruction = f"Generate a {message_type} response to the following message: {user_message}"

            context = {
                'user_profile': self.user_profile,
                'conversation_history': self.conversation_history
            }

            messages = [
                {"role": "system", "content": json.dumps(context)},
                {"role": "user", "content": instruction}
            ]

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=150
            )

            self.conversation_history.append(response.choices[0].message['content'])
            self.save_data()
            return response.choices[0].message['content']

        except (ValueError, openai.Error) as e:
            logging.error(f"Error generating response: {e}")
            return None

def main():
    logging.info("Disclaimer: This application is designed to help generate responses for social interactions. It should not be used to deceive or mislead others. Always be honest and respectful in your communications.")
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    if not openai.api_key:
        raise ValueError("OpenAI API key is required.")
    user_id = input("Enter your user ID: ")
    if not user_id:
        raise ValueError("User ID is required.")
    chat_app = ChatApp(user_id, JsonDataStore('data.json'))
    user_message = input("Enter the message you received: ")
    message_type = input("Enter the type of message (e.g. flirty,professional, friendly): ")
    suggested_response = chat_app.generate_response(user_message, message_type)
    print("Suggested response: " + suggested_response)

if __name__ == '__main__':
    main()