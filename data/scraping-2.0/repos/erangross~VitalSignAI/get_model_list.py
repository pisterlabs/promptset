# Import the required libraries
import openai
import os

# Set the OpenAI API key from the environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Function to list model IDs
def list_model_ids():
    # Get the list of available models from the OpenAI API
    models = openai.Model.list()
    
    # Use a list comprehension to extract the model IDs
    return [model['id'] for model in models['data']]

# Main entry point, only executed when run as a script
if __name__ == "__main__":
    # Call the list_model_ids function and iterate through the result
    for model_id in list_model_ids():
        # Print each model ID
        print(model_id)
