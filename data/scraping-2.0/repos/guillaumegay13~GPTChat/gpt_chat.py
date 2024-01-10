import openai
import time

class GPTChat:
    """
    A class for generating responses to messages using OpenAI's GPT-4 language model.
    
    Attributes:
        api_key (str): The OpenAI API key to use for authentication.
        completion_model (str): The name of the GPT model to use for generating responses.
        chat_history (list): A list of tuples representing the chat history, where each tuple contains the prompt and response.
        
    Methods:
        send_message(message, systemContent, temperature, top_p, max_history): Sends a message to the GPT-4 model and returns the generated response.
    """

    def __init__(self, api_key):
        """
        Initializes a new instance of the GPTChat class.

        Args:
            api_key (str): The OpenAI API key to use for authentication.
        """
        openai.api_key = api_key
        self.completion_model = "gpt-4-1106-preview"
        self.chat_history = []
        self.model_max_length = 128000 # 128K context window

    def send_message(self, message, systemContent, temperature, top_p):
        """
        Sends a message to the GPT-4 chat model, using the maximum tokens available,
        while keeping the entire latest response in memory. Also prints the time taken
        for the API call to complete.
        """

        # Prepare the system content and add previous history
        messages_to_send = [{"role": "system", "content": systemContent}]
        # Extend the history instead of appending to avoid nested lists
        messages_to_send.extend([{"role": history_entry["role"], "content": history_entry["content"]} for history_entry in self.chat_history])

        # Add the new user message to the context
        messages_to_send.append({"role": "user", "content": message})

        # Start timing the API call
        start_time = time.time()

        # Send the message to the API
        try:
            response = openai.ChatCompletion.create(
                model=self.completion_model,
                temperature=temperature,
                top_p=top_p,
                messages=messages_to_send,
                max_tokens=4096  # max response token for GPT-4 turbo
            )
        except openai.error.InvalidRequestError as e:
            raise e  # Raising the error to be handled by the caller

        # Calculate the duration of the API call
        duration = time.time() - start_time
        print(f"API call duration: {duration:.2f} seconds")

        # Extract the response content
        message_response = response.choices[0].message['content'].strip()

        # Update the chat history with the new user message
        self.chat_history.append({"role": "user", "content": message})

        # Update the chat history with the new response
        self.chat_history.append({"role": "assistant", "content": message_response})

        # Return the new response
        return message_response