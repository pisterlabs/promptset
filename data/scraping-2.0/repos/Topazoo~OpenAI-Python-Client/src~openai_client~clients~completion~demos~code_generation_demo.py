from ..client import Code_Completion_Client
import json

# Simple code generation app :)
if __name__ == "__main__":
    # API Key is read from OPENAI_API_KEY
    client = Code_Completion_Client()

    # Get a prompt to generate code with
    prompt = "Create a Python dictionary of all US states without any non-code text"

    # Print the tweet to classify
    print(f"Creating code: '{prompt}'")

    # Send to the model with examples
    response = client.run_prompt(prompt)

    # Load AI generated code into Python object
    us_states = json.loads(response)

    # Print result for California
    print(f"Getting result for key: CA")
    print(us_states["CA"])
