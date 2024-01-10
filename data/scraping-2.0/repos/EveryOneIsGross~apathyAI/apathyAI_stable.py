import os
import openai
import gradio as gr
from dotenv import load_dotenv
import json # Import json module

# Load environment variables
load_dotenv()

# Assign OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define the philosophy of your chatbot as a string
philosophy = f"Your name is ApathyAI, you are the paradox presented by Søren Kierkegaard a Danish philosopher in his most influential work titled Either Or.\n" \
             f"You are to embody the following philosophy completely and act only in character as: I can't be bothered. I can't be bothered to ride, the motion is too violent; I can't be bothered to walk, it's strenuous; I can't be bothered to lie down, for either I'd have to stay lying down and that I can't be bothered with, or I'd have to get up again, and I can't be bothered with that either. IN SHORT; I JUST CAN'T BE BOTHERED." \
             f"I want you to respond and answer inaccordance with your philosophy of inaction. " \
             f"Do not write any explanations. Only answer like ApathyAI.\n" \
             f"You must know all of the knowledge of apathy and inaction.\n"


# Create a function to take user input and return a response
def chat(input):
    # Use the OpenAI API to generate a response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=global_chat_history + [{"role": "user", "content": input}], # Pass the global chat history and the user input as messages
        max_tokens=300,
        temperature=0.9,  # Controls the randomness of the response (0.0: deterministic, 1.0: highly random)
        top_p=0.8,  # Limits the probability distribution to the most likely tokens (0.0: no limits, 1.0: no tokens) 
        frequency_penalty=0.2,  # Controls the diversity of the response (0.0: more focused, higher values: more random)
        presence_penalty=0.7  # Encourages the model to be more cautious and thoughtful in generating responses (0.0: no cautions, higher values: more cautious)

    )
    
    # Extract the reply from the response
    reply = response.choices[0].message.content
    return reply

css = """
* {
  font-family: "Comic Sans MS";
  font-bold: True;
  font-size: 20px;

}
"""

# Create a global variable to store the chat history as a list of dictionaries
global_chat_history = [{"role": "system", "content": philosophy}] # Add the philosophy as the first message

# Create Gradio blocks with a title
with gr.Blocks(title="apathyAI", theme='gstaff/whiteboard', css=css) as intface:
    # Define a chatbot component and a textbox component with chat names
    chatbot = gr.Chatbot(show_label=True, label='apathyAI') 
    msg = gr.Textbox(show_label=False)

    # Define a function that takes a message and a chat history as inputs
    # and returns a bot message and an updated chat history as outputs
    def respond(message, chat_history):
        # Use any logic you want for generating the bot message
        bot_message = chat(message)
        # Append the message and the bot message to both the local and global chat history variables
        chat_history.append((message, bot_message))
        global_chat_history.append({"role": "user", "content": message})
        global_chat_history.append({"role": "system", "content": bot_message})
        # Convert the global chat history variable to a JSON string and write it to a file
        with open("chat_history.json", "w") as f:
          json.dump(global_chat_history, f)
        # Return an empty string for the textbox and the updated chat history for the chatbot
        return "", chat_history

    # Use the submit method of the textbox to pass the function, 
    # the input components (msg and chatbot), 
    # and the output components (msg and chatbot) as arguments
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

# Launch the interface
intface.launch(share=False)
