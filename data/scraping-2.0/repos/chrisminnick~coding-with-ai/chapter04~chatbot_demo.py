import openai
import gradio as gr
from config import OPENAI_API_KEY

# Set the API key from the configuration file
openai.api_key = OPENAI_API_KEY


def chatbot_response(input_text):
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"I'd like to talk to a {input_text}.",
        max_tokens=50  # You can adjust this for longer responses
    )
    return response.choices[0].text


# Create the Gradio interface
iface = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(
        "text", label="Enter the type of bot you want to talk to (e.g., 'coding assistant' or 'therapist')"),
    outputs="text",
    live=True,
    title="Custom Chatbot",
    description="Select the type of bot you'd like to talk to and start the conversation."
)

iface.launch()
