"""MemberBarry is a wrapper class for interacting with the OpenAI API."""
import logging
import os

# import SimpleNamespace
from types import SimpleNamespace

import backoff
import dotenv
import openai
import tiktoken

from aidatabase import AIDatabase

from constants import (
    DEFAULT_SYSTEM_PROMPT,
    GENERATE_SUMMARY_SYSTEM_PROMPT,
    GENERATE_SUMMARY_USER_PROMPT,
    USE_SUMMARY_USER_PROMPT
)

from config import (
    CONTEXT_TOKEN_LIMIT,
    OPENAI_MAX_TOKENS,
    OPENAI_MODEL,
    OPENAI_STT_ENGINE,
    OPENAI_TEMPERATURE
)

from system_prompt import SYSTEM_PROMPT


# Load local environment variables
dotenv.load_dotenv()

# Constants and Keys for OpenAI and Buffer size
openai.api_key = os.environ.get('OPENAI_API_KEY')
openai.organization = os.environ.get('OPENAI_ORG_ID')


class MemberBarry:
    """A wrapper class for interacting with the OpenAI API and retaining
    context."""

    def __init__(
            self, system_prompt=None, session_id=None, openai_model=None,
            stt_engine=None, temp=None, max_tokens=None, db_filename=None):
        """Initialize the OpenAI_API_Wrapper class."""
        # System prompt
        if not system_prompt:
            self.system_prompt = DEFAULT_SYSTEM_PROMPT
        # TODO: Add a warning if a session_id is provided that the session
        # system prompt will be used.
        else:
            self.system_prompt = system_prompt

        # Local context
        # This stores the immediate running context.
        self.running_context = []

        # This stores the number of times the running context has been cleared.
        # (i.e. summarized and stored)
        self.running_context_pass = 0

        # This stores a continuing running summary of the whole conversation.
        self.running_summary = ""

        # This stores the embeddings and metadata for the running context.
        # self.embedding_log = []

        # Attach the database
        self.db = AIDatabase(session_id, db_filename=db_filename)

        # Load the conversation from the database
        self.load_conversation()

        # Set the OpenAI API parameters
        self.stt_engine = stt_engine if stt_engine else OPENAI_STT_ENGINE
        self.openai_model = openai_model if openai_model else OPENAI_MODEL
        self.temp = temp if temp else OPENAI_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens else OPENAI_MAX_TOKENS

        # Tokenizer ~ 100 tokens ~ 75 words | 1 token ~ 4 chars
        self.tokenizer = tiktoken.encoding_for_model(OPENAI_MODEL)

        # Token count
        self.tokens_used = SimpleNamespace(prompt=0, completion=0, total=0)

        # Check if the token count is already beyond the limit
        if self.get_token_count_running_context() > CONTEXT_TOKEN_LIMIT:
            # Generate a new running summary
            self.generate_running_context_summary()

    def load_conversation(self, session_id=None):
        """Load a conversation from the database."""
        # If there are existing summaries, load the latest one and reload any
        # further running context that was stored but not summarized.
        latest_summary = self.db.get_most_recent_summary(session_id=session_id)
        if latest_summary:
            # Set the running summary to the latest summary
            self.running_summary = latest_summary['summary']
            # Set the running context pass to the latest summary pass
            self.running_context_pass = latest_summary['pass']+1

        # Then load only the latest set of context conversations.
        self.running_context = \
            self.db.create_context_chain(
                session_id=session_id, context_pass=self.running_context_pass)

        if not self.running_context:
            self.reset_running_context()

    def transcribe_audio(self, filename):
        """Transcribe audio to text using OpenAI.

        Args:
            filename (str): Path to audio file.

        Returns:
            str: The transcribed text.
        """
        logging.info('Transcribing audio...')

        with open(filename, 'rb') as file:
            # Transcribe audio to text using OpenAI
            transcript = openai.Audio.transcribe(
                self.STT_ENGINE,
                file=file
            )
            transcript_text = transcript.text

        return transcript_text

    def simple_summarize_text(self, text, system_prompt=None):
        """Summarize text using OpenAI.

        Args:
            text (str): The text to summarize.
            system_prompt (str): The system prompt to send to the OpenAI API.
                Defaults to "Please summarize the text."

        Returns:
            str: The summary from the OpenAI API.
        """
        # Log what's happening
        logging.info('Summarizing text...')
        logging.debug(f'\ninput text: \n{text}\n')

        # Set the system prompt to a default if none is provided
        if not system_prompt:
            system_prompt = "Please summarize the text."

        # Send the prompt to the OpenAI API and return the response
        summary = self.send_prompt(text, system_prompt)
        logging.debug(f'\nsummarize_text() response: \n{summary}\n')

        return summary

    def send_prompt(self, text, system_prompt=None, use_full_context=False,
                    context=None):
        """Send a prompt to the OpenAI API and return the response.

        Args:
            text (str): The text to send to the OpenAI API.
            system_prompt (str): The system prompt to send to the OpenAI API.
            use_full_context (bool): Whether to use the full context in the
            instance.
            context (list): A raw list of messages to send to the OpenAI API.
        """
        if system_prompt and use_full_context:
            raise ValueError(
                """You can't use the full context when you are also sending a
                system prompt. The stored full context already contains a
                system prompt. Please remove your system prompt or send an
                explicit ad-hoc context.""")

        # Create a list to store the messages
        messages = []

        # Add a default system prompt if none is provided
        if system_prompt:
            # Add the system prompt at the front
            messages.append({"role": "system", "content": system_prompt})

        long_term_memory = self.db.get_similar_convos(text)

        if long_term_memory:
            # Add the long term memory to the messages
            for memory in long_term_memory:
                if memory not in self.running_context:
                    messages.append(memory)

        # Check if the full context should be used
        if use_full_context:
            full_context = []

            # If there is a running summary, add it to the full context
            if self.running_summary:
                # Create a message to summarize the entire conversation
                full_context.append({
                    "role": "user",
                    "content": USE_SUMMARY_USER_PROMPT + self.running_summary
                })
                full_context.append({
                    "role": "assistant",
                    "content": "OK."
                })

            # Add each message in the running context to the full context
            for msg in self.running_context:
                # There is debate as to how the system prompt should be
                # handled: sending it first, last, or just sending system
                # directives as a standard user prompt. We'll be sending it
                # first here.
                #
                # See: https://community.openai.com/t/the-system-role-how-it-influences-the-chat-behavior/87353  # noqa
                if msg['role'] == 'system':
                    full_context.insert(0, msg)
                else:
                    full_context.append(msg)

            messages += full_context

        # Check if explicit context is provided
        if context:
            # Add the context to the messages
            for message in context:
                messages.append(message)

        # Ensure the transcript_text is at the end
        messages.append({"role": "user", "content": text})

        logging.debug(f"\nsend_prompt() message: \n{messages}\n\n")

        # Send the message to the OpenAI API and return the response
        response = self._raw_send_message(messages)

        # Add the query and response to the running context
        self.add_to_running_context(user=text, assistant=response)

        # Insert the conversation into the db
        insert_id = self.db.insert_convo(
            system_prompt if system_prompt else self.system_prompt,
            text,
            response,
            self.running_context_pass)

        self.db.add_embedding(
            user_message=text,
            assistant_response=response,
            convo_id=insert_id)

        return response

    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def _raw_send_message(self, messages):
        """Send a message to the OpenAI API and return the response.

        Args:
            messages (list): A list of messages to send to the OpenAI API.

        Returns:
            str: The response from the OpenAI API.

        Raises:
            openai.error.RateLimitError: If the OpenAI API returns a rate limit
                error. This will trigger a backoff and retry.
        """
        try:
            # Summarize transcribed text using the OpenAI API
            response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=messages,
                temperature=self.temp,
                # max_tokens=self.max_tokens
            )

            # Update the token counts for the session
            self.tokens_used.prompt += response.usage.prompt_tokens
            self.tokens_used.completion += response.usage.completion_tokens
            self.tokens_used.total += response.usage.total_tokens

            # Grab the content from the response
            response = response.choices[0].message.content

        except Exception as exc:
            logging.error(f"Error: {exc}")
            return "There was an error. Please try again."

        # Add the response to the running context
        return response

    def add_to_running_context(self, **kwargs):
        """Add the given text to the running context.

        Args:
            kwargs (dict): A dictionary of key-value pairs. The key is the
                role and the value is the content.

        Example:
            add_to_running_context(
                user="Hello.",
                assistant="Greetings, flesh bag.")
        """
        for key, value in kwargs.items():
            self.running_context.append({'role': key, 'content': value})

    def add_to_running_summary(self, text):
        """Add the given text to the running summary."""
        # Add a space, just in case...
        self.running_summary += f" {text}"

    def set_running_summary(self, text):
        """Set the running summary to the given text."""
        self.running_summary = text

    def reset_running_context(self):
        """Reset the running context to contain only the current system
        prompt."""
        self.running_context = [
            {'role': 'system', 'content': self.system_prompt}
        ]

    def clear_running_summary(self):
        """Clear the running summary."""
        self.running_summary = ""

    def generate_running_summary_message(self):
        """Generate a new running summary message."""
        # Combine the entire running summary into a single message
        messages = [
            {
                'role': 'system',
                'content': GENERATE_SUMMARY_SYSTEM_PROMPT
            },
        ]

        # add the running summary to the messages
        if self.running_summary:
            messages.append(
                {
                    'role': 'user',
                    'content': USE_SUMMARY_USER_PROMPT + self.running_summary
                }
            )
            messages.append(
                {
                    'role': 'assistant',
                    'content': 'OK.'
                }
            )

        # Combine the entire running context into the messages
        for message in self.running_context:
            if message['role'] != 'system':
                messages.append(message)

        # Add a final directive to summarize the entire conversation
        messages.append(
            {
                'role': 'user',
                'content': GENERATE_SUMMARY_USER_PROMPT
            }
        )

        return messages

    def generate_running_summary(self):
        """Generate a new running summary."""
        # Generate the message for the running summary
        messages = self.generate_running_summary_message()

        # Send the message to the OpenAI API and return the response
        summary = self._raw_send_message(messages)

        # Clear the running context
        self.reset_running_context()

        # Set the running summary to the response
        self.set_running_summary(summary)

        # Insert the summary into the db
        self.db.insert_summary(
            summary,
            summary_type="summary",
            context_pass=self.running_context_pass)

        # Increment the running context pass
        self.running_context_pass += 1

    def generate_running_context_summary(self):
        """Generate a running context summary and use it to replace the current
        running context."""

        # Generate the message for the running context
        messages = self.generate_running_summary_message()

        # Send the message to the OpenAI API and return the response
        summary = self._raw_send_message(messages)

        self.reset_running_context()
        self.add_to_running_context(
            user=USE_SUMMARY_USER_PROMPT + summary,
            assistant='OK.')

        # Insert the summary into the db
        self.db.insert_summary(
            summary,
            summary_type="context")

    def clear_full_running_summaries(self):
        """Clear the running summaries."""
        self.reset_running_context()
        self.clear_running_summary()

    def get_token_count(self, text):
        """Return the number of tokens in the given text."""
        return len(self.tokenizer.encode(text))

    def get_token_count_running_context(self):
        """Return the number of tokens in the running context."""
        if not self.running_context:
            return 0

        # Get the token count for the running context
        count = self.get_token_count(
            " ".join([m['content'] for m in self.running_context]))

        # Log the token count
        logging.debug(f'token count running context: {count}')

        return count

    def get_token_count_running_summary(self):
        """Return the number of tokens in the running summary."""
        return self.get_token_count(self.running_summary)

    def get_token_count_full_context(self):
        """Return the number of tokens in the full context."""
        msg = self.generate_running_summary_message()

        return self.get_token_count(
            " ".join([message['content'] for message in msg]))

    def get_session_id(self):
        """Return the session_id."""
        return self.db.get_session_id()


# Run an example of the OpenAI_API_Wrapper class
if __name__ == "__main__":

    import argparse

    # Parse "session_id" if passed
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session_id", help="An optional session_id to use for the chat.")
    args = parser.parse_args()

    # Create an instance of the OpenAI_API_Wrapper class
    api = MemberBarry(
        session_id=args.session_id,
        system_prompt=SYSTEM_PROMPT)

    print(f'This chat session ID is: {api.get_session_id()}')

    # Create a loop that breaks on a specific input
    while True:
        # Get input from the user
        text = input("Enter text: ")

        # Check if the input is empty
        if not text:
            break

        # Send the prompt to the OpenAI API
        response = api.send_prompt(text, use_full_context=True)

        # Print the response
        print(response)

        # check if the api's running context has too many tokens.
        if api.get_token_count_running_context() > CONTEXT_TOKEN_LIMIT:
            # Generate a new running summary
            api.generate_running_summary()
