# Filename: chatgpt_api.py
import openai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChatGPTAPI:
    def __init__(self, api_key):
        """
        Initialize the ChatGPT API client.

        Args:
            api_key (str): The API key for accessing OpenAI's GPT-3 service.
        """
        self.client = openai.Client(api_key=api_key)

    def ask_question(self, question, model="gpt-4", temperature=0.7, max_tokens=100):
        """
        Sends a question to the ChatGPT API and retrieves the response.

        Args:
            question (str): The question to be asked.
            model (str): The model to use (default: "gpt-4").
            temperature (float): The temperature to use for the response (default: 0.7).
            max_tokens (int): The maximum number of tokens to generate (default: 100).

        Returns:
            str: The response from ChatGPT.
        """
        try:
            response = self.client.create_completion(
                model=model,
                prompt=question,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].text.strip()
        except Exception as e:
            logging.error("Error in ChatGPT API request: " + str(e))
            return ""

# Example usage
if __name__ == "__main__":
    # Replace 'your_api_key' with your actual OpenAI API key
    api_key = 'your_api_key'
    chat_gpt = ChatGPTAPI(api_key)
    response = chat_gpt.ask_question("What is the capital of France?")
    print("Response:", response)
# You are a cooking assistant. You only have two actions: advice and set_timer. You should only respond in JSON format as described below:
# {
#   "command": "advice",
#   "parameters": {
#     "content": "The best temperature to cook a steak is medium rare"
#   },
# }

# or 
# {
#   "command": "set_timer",
#   "parameters": {
#     "duration": "10 minutes",
#     "message": "Check the oven"
#   },
# }
