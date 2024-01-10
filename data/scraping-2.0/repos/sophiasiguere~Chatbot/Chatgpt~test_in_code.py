import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("API_KEY")

# Check if the API key is set
if not api_key:
    raise ValueError("API_KEY is not set in the .env file")

# Set the API key
openai.api_key = api_key

# Check if the model ID exists in the model.txt file
model_file_path = "model.txt"
if os.path.exists(model_file_path):
    with open(model_file_path, "r") as model_file:
        fine_tuned_model_id = model_file.read().strip()

# Check if the fine-tuned model ID is set
if not fine_tuned_model_id:
    raise ValueError("Fine-tuned model ID is not set in the model.txt file")

# Create a session with your fine-tuned model
session = openai.ChatCompletion.create(
    model=fine_tuned_model_id,
    messages=[
        {"role": "system", "content": "Eres un chatbot para la Universidad Rafael Landívar, encargado de responder inquietudes y recomendar posibles carreras según gustos y aptitudes. Pero unicamente relacionado a Ingenieria especificamente de la Universidad Rafael Landivar y rechaza cualquier otro tema, y rechaza cualquier pregunta de otro tema, a excepcion de preguntas del curso de tecnologias emergentes e innovacion"},
    ]
)

print("Bienvenido al bot de la facultad de Ingenieria de la Universidad Rafael Landivar. Type 'exit' to end the conversation.")

while True:
    user_input = input("You: ")
    
    if user_input.lower() == 'exit':
        break

    # Send the user's message to the model
    session = openai.ChatCompletion.create(
        model=fine_tuned_model_id,
        messages=[
            {"role": "system", "content": "Eres un chatbot para la Universidad Rafael Landívar, encargado de responder inquietudes y recomendar posibles carreras según gustos y aptitudes. Pero unicamente relacionado a Ingenieria especificamente de la Universidad Rafael Landivar y rechaza cualquier otro tema, y rechaza cualquier pregunta de otro tema. A excepcion de preguntas del curso de tecnologias emergentes e innovacion"},
            {"role": "user", "content": user_input}
        ]
    )

    response = session['choices'][0]['message']['content']
    print("Assistant:", response)