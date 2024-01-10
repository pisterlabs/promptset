import logging
import openai
import os
import sys
import time
import streamlit
from azure.identity import DefaultAzureCredential
from dotenv import dotenv_values
from dotenv import load_dotenv
from streamlit_chat import message

# Read environment variables
assistant_profile = """
I am a chatbot backed by AKS and Azure OpenAi
"""
title = os.environ.get("TITLE", "Chatbot backed by AKS and Azure OpenAI")
image_file_name = os.environ.get("IMAGE_FILE_NAME", "mychatbot.png")
system = os.environ.get("SYSTEM", assistant_profile)
api_type = os.environ.get("AZURE_OPENAI_TYPE", "azure")
api_base = os.getenv("AZURE_OPENAI_BASE")
api_key = os.getenv("AZURE_OPENAI_KEY")
api_version = os.environ.get("AZURE_OPENAI_VERSION", "2023-05-15")
engine = os.getenv("AZURE_OPENAI_DEPLOYMENT")
model = os.getenv("AZURE_OPENAI_MODEL")
temperature = float(os.environ.get("TEMPERATURE", 0.5))
default_credential = None
image_width = 80

def main():
  # config azure openai
  configure_azure_openai()
  
  # customize streanlit ui
  customize_streamlit_ui()
  
  col1, col2 = streamlit.columns([1, 6])

  # Display the robot image
  with col1:
    streamlit.image(image = os.path.join("images", image_file_name), width = image_width)

  # Display the header
  with col2:
    streamlit.header(title)

  col3, col4, col5 = streamlit.columns([6, 1, 1])

  # Create text input in column 1
  with col3:
    user_input = streamlit.text_input(" ", key = "user", on_change = user_change)

  # Create send button in column 2
  with col4:
    streamlit.button(label = "Post")

  # Create cleam button in column 3
  with col5:
    streamlit.button(label = "Clean", on_click = clean_click)

  if streamlit.session_state['generated']:
    for i in range(len(streamlit.session_state['generated']) - 1, -1, -1):
       streamlit.markdown("**:blue[{}]**".format(streamlit.session_state['past'][i]))
       streamlit.markdown(streamlit.session_state['generated'][i])
       streamlit.markdown("""---""")

def configure_azure_openai():
  # Set default Azure credential
  default_credential = DefaultAzureCredential() if api_type == "azure_ad" else None

  # Authenticate to Azure OpenAI
  if api_type == "azure":
    openai.api_key = api_key
  elif api_type == "azure_ad":
    openai_token = default_credential.get_token("https://cognitiveservices.azure.com/.default")
    openai.api_key = openai_token.token
    if 'openai_token' not in streamlit.session_state:
      streamlit.session_state['openai_token'] = openai_token
  else:
    raise ValueError("Invalid API type. Please set the AZURE_OPENAI_TYPE environment variable to azure or azure_ad.")

  # Configure Azure OpenAI
  openai.api_type = api_type
  openai.api_version = api_version
  openai.api_base = api_base 

def customize_streamlit_ui():
  # Customize Streamlit UI using CSS
  streamlit.set_page_config(page_title='MyChatbot on AKS and AzureOpenAI')
  streamlit.markdown("""
  <style>

  div.stButton > button:first-child {
    background-color: #eb5424;
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 0.5rem;
    padding: 0.5rem 1rem;
    border: none;
    box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.15);
    width: 500 px;
    height: 42px;
    transition: all 0.2s ease-in-out;
  } 

  div.stButton > button:first-child:hover {
    transform: translateY(-3px);
    box-shadow: 0 1rem 2rem rgba(0,0,0,0.15);
  }

  div.stButton > button:first-child:active {
    transform: translateY(-1px);
    box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.15);
  }

  div.stButton > button:focus:not(:focus-visible) {
    color: #FFFFFF;
  }

  @media only screen and (min-width: 1000px) {
    /* For desktop: */
    div {
      font-family: 'Roboto', sans-serif;
    }

    div.stButton > button:first-child {
      background-color: #0066cc;
      color: white;
      font-size: 20px;
      font-weight: bold;
      border-radius: 0.5rem;
      padding: 0.5rem 1rem;
      border: none;
      box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.15);
      width: 500 px;
      height: 42px;
      transition: all 0.2s ease-in-out;
      position: relative;
      bottom: -32px;
      right: 0px;
    } 

    div.stButton > button:first-child:hover {
      transform: translateY(-3px);
      box-shadow: 0 1rem 2rem rgba(0,0,0,0.15);
    }

    div.stButton > button:first-child:active {
      transform: translateY(-1px);
      box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.15);
    }

    div.stButton > button:focus:not(:focus-visible) {
      color: #FFFFFF;
    }

    input {
      border-radius: 0.5rem;
      padding: 0.5rem 1rem;
      border: none;
      box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.15);
      transition: all 0.2s ease-in-out;
      height: 40px;
    }
  }

  footer {visibility: hidden;}
  </style>
  """, unsafe_allow_html=True)

  # Initialize Streamlit session state
  if 'prompts' not in streamlit.session_state:
    streamlit.session_state['prompts'] = [{"role": "system", "content": system}]

  if 'generated' not in streamlit.session_state:
    streamlit.session_state['generated'] = []

  if 'past' not in streamlit.session_state:
    streamlit.session_state['past'] = []

  if 'user' not in streamlit.session_state:
    streamlit.session_state['user'] = ""

def refresh_openai_token():
  if streamlit.session_state['openai_token'].expires_on < int(time.time()) - 30 * 60:
      streamlit.session_state['openai_token'] = default_credential.get_token("https://cognitiveservices.azure.com/.default")
      openai.api_key = streamlit.session_state['openai_token'].token

# Send user prompt to Azure OpenAI 
def generate_response(prompt):
  try:
    streamlit.session_state['prompts'].append({"role": "user", "content": prompt})

    if openai.api_type == "azure_ad":
      refresh_openai_token()

    completion = openai.ChatCompletion.create(
      engine = engine,
      model = model,
      messages = streamlit.session_state['prompts'],
      temperature = temperature,
    )
    
    message = completion.choices[0].message.content
    return message
  except Exception as e:
    logging.exception(f"Exception in generate_response: {e}")

# Reset Streamlit session state to start a new chat from scratch
def clean_click():
  streamlit.session_state['prompts'] = [{"role": "system", "content": system}]
  streamlit.session_state['past'] = []
  streamlit.session_state['generated'] = []
  streamlit.session_state['user'] = ""

# Handle on_change event for user input
def user_change():
  # Avoid handling the event twice when clicking the Send button
  chat_input = streamlit.session_state['user']
  streamlit.session_state['user'] = ""
  if (chat_input == '' or
      (len(streamlit.session_state['past']) > 0 and chat_input == streamlit.session_state['past'][-1])):
    return
  
  # Generate response invoking Azure OpenAI LLM
  if chat_input !=  '':
    output = generate_response(chat_input)
    
    # store the output
    streamlit.session_state['past'].append(chat_input)
    streamlit.session_state['generated'].append(output)
    streamlit.session_state['prompts'].append({"role": "assistant", "content": output})

if __name__ == '__main__':
  main()