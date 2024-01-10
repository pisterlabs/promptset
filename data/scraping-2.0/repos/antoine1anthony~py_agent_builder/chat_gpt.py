# chat_gpt.py
import openai
import logging

# Define custom exceptions
class ChatGPTError(Exception):
    pass

class ChatGPT:
    DEFAULT_PARAMS = {
        "temperature": 0.75,
        "frequency_penalty": 0.2,
        "presence_penalty": 0
    }

    def __init__(self, api_key, chatbot, retries=3):
        self.api_key = api_key
        self.chatbot = chatbot
        self.conversation = []
        self.retries = retries
        openai.api_key = self.api_key

    def chat(self, user_input, log_file, bot_name):
        self.conversation.append({"role": "user", "content": user_input})
        response = self.chatgpt_with_retry(self.conversation, self.chatbot, user_input)
        self.conversation.append({"role": "assistant", "content": response})
        # Save to log file
        with open(log_file, 'a') as f:
            f.write(f"User: {user_input}\n")
            f.write(f"{bot_name}: {response}\n\n")
        # Remove oldest message from the conversation after 4 turns
        if len(self.conversation) > 4:
            self.conversation.pop(0)
        return response


    def is_repetitive(self, response):
        # Check the last few responses for repetition. Adjust the range as needed.
        for message in self.conversation[-3:]:
            if message['content'] == response:
                return True
        return False

    def chatgpt(self, conversation, chatbot, user_input, **kwargs):

        params = {**self.DEFAULT_PARAMS, **kwargs}
        
        messages_input = conversation.copy()
        
        prompt = [{"role": "system", "content": chatbot}]
        
        messages_input.insert(0, prompt[0])

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=params["temperature"],
            frequency_penalty=params["frequency_penalty"],
            presence_penalty=params["presence_penalty"],
            messages=messages_input)
        
        chat_response = completion['choices'][0]['message']['content']

        # Check if the response is repetitive
        if self.is_repetitive(chat_response):
            # If the response is repetitive, try again
            return self.chatgpt(conversation, chatbot, user_input, **kwargs)
        else:
            return chat_response

    def chatgpt_with_retry(self, conversation, chatbot, user_input, **kwargs):
        for i in range(self.retries):
            try:
                return self.chatgpt(conversation, chatbot, user_input, **kwargs)
            except openai.api.error.APIError as e:
                logging.warning(f"Error in chatgpt attempt {i + 1}: {e}. Retrying...")
            except Exception as e:
                logging.error(f"Unexpected error in chatgpt attempt {i + 1}: {e}. No more retries.")
                raise ChatGPTError from e
        return None
