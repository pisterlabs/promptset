import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up prompt
prompt = "what is the size of earth"

# Set up Codex API parameters
model_engine = "davinci"
temperature = 0.8
max_tokens = 200
stop_sequence = "\n"

# Call Codex API
def predict_text(input):
    response = openai.Completion.create(
        # engine=model_engine,
        model="text-davinci-003",
        prompt=input,
        temperature=temperature,
        max_tokens=max_tokens,
        # stop=stop_sequence
    )
    return response.choices[0].text.strip()

while True:
    user_input = input("Enter some text: ")
    response = predict_text(user_input)
    print(response)