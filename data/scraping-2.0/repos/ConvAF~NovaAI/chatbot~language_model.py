import os
import openai
import json
from pathlib import Path
from flask import current_app

from chatbot.chat import ChatHistory

ENGINES = {
    'ada': 'text-ada-001',
    'babbage': 'text-babbage-001',
    'curie': 'text-curie-001',
    'davinci': 'text-davinci-002'
}

class LanguageModel():
    """ Language Generation Model
    """
    def __init__(self):

        # Init openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.engine = os.getenv("OPENAI_ENGINE")
        self.temperature = 0.5
        self.max_tokens = 50
        

    def add_response_to_chat_history(self, chat_history: ChatHistory):
        """ Generate a response from the bot and append to chat history.
        """
        # if chat_history is None or len(chat_history)==0:
        #     return chat_history
        
        reply_raw_text = self.get_response_from_GPT3(chat_history)

        reply_text = self.clean_reply_text(reply_raw_text,
                                        tag_bot = chat_history.tag_bot,
                                        tag_user = chat_history.tag_user
                                        )

        if reply_text:
            chat_history.add_bot_message(reply_text)
        return chat_history


    def get_response_from_GPT3(self, chat_history):
        """ Get a reply from GPT3 
        
        Returns:
        --------
         - reply: str
            A text string containing just the reply from the model.
            Example: "I'm fine, how are you?"
        """
        # prompt_with_dialog = self.create_prompt_with_dialog(chat_history, prompt_text)
        prompt_with_dialog = chat_history.get_as_prompt_with_dialog()

        # Add the stop sequences (such as "Human:" and "Nova:")
        # stop_sequences = [f'{tag}:' for tag in [chat_history.tag_user, chat_history.tag_bot]]
        stop_sequences = [f'{tag}:' for tag in [chat_history.tag_user]]

        response = openai.Completion.create(
                engine=self.engine,
                prompt=prompt_with_dialog,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop_sequences
            )

        if response and ('choices' in response) and len(response['choices']):
            reply_raw = response['choices'][0]['text']
            return reply_raw

        return ''
        # reply = None
        # if response and ('choices' in response) and len(response['choices']):

        #     reply_raw = response['choices'][0]['text']
        #     import sys
        #     print("-----------------\nReply raw\n\n", reply_raw, file=sys.stdout)
    
        #     reply = self.clean_reply_text(reply_raw,
        #                                   tag_bot = chat_history.tag_bot,
        #                                   tag_user = chat_history.tag_user
        #                                   )

        # return reply

    def clean_reply_text(self, reply_raw, tag_bot, tag_user):
        " Clean up the reply reply_raw a bit "

        reply = reply_raw.strip()

        # Remove new line characters
        reply = reply.replace(f"\n", "")

        # Get rid of "Bot: " at beginning of message
        reply = reply.replace(f"{tag_bot}: ", "")
        reply = reply.replace(f"{tag_bot}: ".lower(), "")

        # Sometimes, a partial reply of a user is included. Stop answer there
        # Example: "Bot: Hello, how are you? User: "
        if f'{tag_user}:' in reply:
            idx = reply.find(f'{tag_user}:')
            reply = reply[:idx].strip()

        return reply
        

    # def create_prompt_with_dialog(self, chat_history, prompt_text) -> str:
        """ Create a prompt to get a response from GPT-3.
        
        Combines the base prompt and the recent chat history
        to a prompt with dialog for GPT3 to create the next sentence.
        """
        # prompts = current_app.prompts
        # assert chat_type in prompts.keys()
        # prompt = prompts[chat_type]
        # prompt_text = prompt['text']

        # Limit the chat_history to the past 100 messages
        # messages = chat_history.messages[-10:]
        # # Exclude messages that have a correction
        # dialog = "\n".join([f"{message['sender'].title()}: {message['text']}" for message in messages
        #                          if not ('correction' in message.keys())])

        # prompt_with_dialog = "\n".join([prompt_text, dialog])
        # return prompt_with_dialog