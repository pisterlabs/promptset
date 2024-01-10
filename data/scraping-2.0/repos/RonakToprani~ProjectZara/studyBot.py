import openai

class Chatbot:
    def __init__(self, name, api_key):
        # Initialize chatbot's name and state
        self.name = name
        self.state = "INITIAL"

        # Set up OpenAI API with API key
        openai.api_key = #will change later 

    def get_response(self, input_text):
        if self.state == "INITIAL":
            # Greet the user and prompt for input
            response = "Hi, I'm " + self.name + ", a friendly StudyBot designed to help you with homework problems. How can I assist you today?"
            self.state = "WAITING_INPUT"
        elif self.state == "WAITING_INPUT":
            # Send user input to OpenAI API to generate a response
            response = openai.Completion.create(
                engine="davinci",
                prompt=input_text,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.7
            ).choices[0].text.strip()
        else:
            # Return an error message and reset the chatbot's state
            response = "I'm sorry, there seems to be an issue. Please try again later."
            self.state = "INITIAL"
        return response
