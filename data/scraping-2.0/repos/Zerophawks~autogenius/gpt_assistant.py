# Importing necessary libraries and modules
import pyautogen as pg
import openai.gpt as gpt


# Defining the GPTAssistant class
class GPTAssistant(pg.GenericAgent):
    # Initialization of the class
    def __init__(self, name="GPTAssistant"):
        super(GPTAssistant, self).__init__(name)

    # Method to handle assistant requests
    def process_request(self, request):
        # Processing request using GPT model
        response = gpt.generate(request)
        return response
