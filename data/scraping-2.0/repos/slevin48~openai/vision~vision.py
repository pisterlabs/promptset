from openai import OpenAI
import base64, requests, os
import streamlit as st
from PIL import Image

st.set_page_config(page_title='Vision 48',page_icon='👀')

avatar = {"assistant": "👀", "user": "🐱"}

# Set the API key for the openai package
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=st.secrets['OPEN_AI_KEY'],
)

# OpenAI API Key
api_key = st.secrets['OPEN_AI_KEY']

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def chat_vision(messages):

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": messages,  
        "max_tokens": 500
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response.json()['choices'][0]['message']

# Initialization
if 'convo' not in st.session_state:
    st.session_state.convo = []

if 'advanced_mode' not in st.session_state:
  st.session_state.advanced_mode = False

# create the folder vision/img if it doesn't exist
if not os.path.exists('vision/img'):
  os.makedirs('vision/img')


st.sidebar.title('GPT-4 vision 🤖👀')

# Add a toggle to enable advanced mode
advanced_mode = st.sidebar.toggle('Advanced mode')
if advanced_mode:
  password = st.sidebar.text_input('Enter password', type='password')
  if password == st.secrets['PASSWORD']:
    st.session_state.advanced_mode = True
  else:
    st.sidebar.warning('Incorrect password')

if st.session_state.advanced_mode:
    with st.sidebar.expander('?'):
        # Example of image download
        st.markdown('### What is in the image?')
        # Download the image from the URL
        img = 'DallE/img/funny corgi in a cartoon style.png'
        # Display the downloaded image
        image = Image.open(img)
        st.image(image, caption='Funny corgi in a cartoon style')
        with open(img, 'rb') as file:
            st.download_button(
                label='Download image',
                data=file,
                file_name='funny_corgi.png',
                mime='image/png',
            )
   # Add an upload field to the sidebar for images
    uploaded_file = st.file_uploader('Upload an image')
    if uploaded_file is not None:
        # # Convert the file to an image and display it
        image = Image.open(uploaded_file)
        st.image(image,caption=uploaded_file.name)
        # Save image to disk
        with open('vision/img/'+ uploaded_file.name, 'wb') as f:
            f.write(uploaded_file.getvalue())

# Display the response in the Streamlit app
for line in st.session_state.convo:
    # st.chat_message(line.role,avatar=avatar[line.role]).write(line.content)
    if line['role'] == 'user':
      with st.chat_message('user',avatar=avatar['user']):
        st.write(line['content'][0]['text'])
        # st.image(line['content'][1]['image_url']['url'])
    elif line['role'] == 'assistant':
      st.chat_message('assistant',avatar=avatar['assistant']).write(line['content'])

# Create a text input widget in the Streamlit app
disabled = not st.session_state.advanced_mode
# st.sidebar.write(disabled)
prompt = st.chat_input('Ask question about images 👀',disabled=disabled)

if prompt:
  # Append the text input to the conversation
  with st.chat_message('user',avatar='🐱'):
    st.write(prompt)
  
  # Getting the base64 string
  image_path = 'vision/img/'+ uploaded_file.name
  base64_image = encode_image(image_path)  
  st.session_state.convo.append({'role': 'user',
                                'content': [{ "type": "text","text": prompt},
                                            {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]
                                })
  # Query the chatbot with the complete conversation
  with st.chat_message('assistant',avatar='🤖'):
     result = chat_vision(st.session_state.convo)
     st.write(result['content'])
  # Add response to the conversation
  st.session_state.convo.append(result)


# Debug
if st.sidebar.toggle('Debug mode',key='debug_mode'):
    st.sidebar.write(st.session_state.convo)