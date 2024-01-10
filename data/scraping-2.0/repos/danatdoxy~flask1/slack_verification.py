
import sys
from openai import OpenAI
import logging
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class OpenAIChatHandler:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)

    def send_message_to_openai(self, slack_event: dict):
        logging.info('Sending message to openai')

        # Extract the message content from the Slack event
        user_message = slack_event.get("text", "")

        # Create the system and user messages for OpenAI
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ]

        # Send the messages to OpenAI and get the response
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        # Return the OpenAI response
        return completion.choices[0].message.content

    def send_thread_to_openai(self, chat_array):
        # Send the messages to OpenAI and get the response
        logging.info('Sending thread to openai')
        completion = self.client.chat.completions.create(
            model="gpt-4",
            messages=chat_array
        )
        # Access the 'content' attribute directly from the 'message' object
        response = completion.choices[0].message.content  # Instead of ['content']
        logging.info(f'Chat response is:\n {response}')
        return response

class SlackHandler:
    def __init__(self, token: str):
        self.client = WebClient(token=token)

    def get_thread_history(self, channel: str, thread_ts: str):
        logging.info('Fetching messaging history')
        messages = []
        try:
            response = self.client.conversations_replies(channel=channel, ts=thread_ts, limit=200)

            # Add a check for response and response_metadata
            if response and 'messages' in response:
                messages.extend(response['messages'])

                while 'response_metadata' in response and 'next_cursor' in response['response_metadata']:
                    next_cursor = response['response_metadata']['next_cursor']
                    if not next_cursor:  # Check if next_cursor is empty
                        break

                    response = self.client.conversations_replies(
                        channel=channel,
                        ts=thread_ts,
                        cursor=next_cursor,
                        limit=200
                    )
                    if 'messages' in response:
                        messages.extend(response['messages'])

        except SlackApiError as e:
            logging.error(f"Error fetching conversation replies: {e.response['error']}")

        return messages

    def post_thread_message(self, channel: str, thread_ts: str, message: str, ephemeral: bool = False, user_id: str = None): # This function will post a message to a thread either as a user or as an ephemeral message
        logging.info('Sending message to slack')
        try:
            if ephemeral:
                self.client.chat_postEphemeral(channel=channel, user=user_id, text=message, thread_ts=thread_ts)
            else:
                self.client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=message) #A message will be posted to the thread

        except SlackApiError as e:
            logging.error(f"Error posting message: {e.response['error']}")

    def build_chat_array(self, messages, system_message="You are a helpful assistant.", as_string=False):
        logging.info('Building chat array')

        # Define the initial system message
        chat_array = [{"role": "system", "content": system_message}]

        if as_string:
            combined_messages = []
            real_name_index = {}  # Use a dictionary for efficient lookup

            for message in messages:
                user_id = message.get("user", "")
                if user_id not in real_name_index:
                    # Get the user's real name from the users.info API
                    try:
                        user_info = self.client.users_info(user=user_id)
                        real_name = user_info.get("user", {}).get("real_name", user_id)  # Default to user_id if real_name not found
                        real_name_index[user_id] = real_name
                    except SlackApiError as e: # If the user is a bot, the users.info API will return an error. So we'll just call it "Bot"
                        real_name_index[user_id] = "Bot"
                else:
                    real_name = real_name_index[user_id]

                try:
                    text = message.get("text", "")
                    combined_messages.append(f"{real_name}: {text}")
                except Exception as e:
                    logging.error(f"Error processing message: {e}")

            # Join combined_messages array into a single string
            combined_messages_string = "\n".join(combined_messages)
            chat_array.append({"role": "user", "content": combined_messages_string})
            logging.info(f'Chat array built as string:\n {chat_array}')

        else:
            for message in messages:
                try:
                    role = "assistant" if message.get("subtype") == "bot_message" else "user"
                    text = message.get("text", "")
                    chat_array.append({"role": role, "content": text})
                except Exception as e:
                    logging.error(f"Error processing message: {e}")
            logging.info(f'Chat array built as user/assistant message array')

        return chat_array