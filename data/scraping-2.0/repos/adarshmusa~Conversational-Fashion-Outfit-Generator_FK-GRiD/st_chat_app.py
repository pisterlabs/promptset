import streamlit as st

import os
import openai

os.environ['OPENAI_API_KEY'] = "" # enter OpenAI key here
openai.api_key = "" # enter OpenAI key here

# Setup LLM

from llama_index import LLMPredictor, GPTVectorStoreIndex, PromptHelper
from langchain.chat_models import ChatOpenAI

from llama_index import SimpleDirectoryReader
from llama_index import GPTVectorStoreIndex
import json

documents = SimpleDirectoryReader('./fashion_data').load_data()

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo"))
# define prompt helper
# set maximum input size
max_input_size = 4096
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
#prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap) #new 2 line code below
num_output = 4096
prompt_helper = PromptHelper(max_input_size, num_output, chunk_overlap_ratio= 0.2, chunk_size_limit=1024)

custom_LLM_index = GPTVectorStoreIndex(
    documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
)


# define Chatbot as a class

class Chatbot:
    def __init__(self, api_key, index):
        self.index = index
        openai.api_key = api_key
        self.chat_history = [{"role": "system",
                              "content": "You are a friendly fashion outfit recommendation assistant which recommends outfits after taking information regarding age, gender, body shape, occasion and location. Update outfits according to the user's preferences and opinions accordingly. Also, it should collect user name and information and provide them with the user name and information when asked."}]

    def generate_response(self, user_input):
        prompt = "\n".join([f"{message['role']}: {message['content']}" for message in self.chat_history])
        prompt += f"\nUser: {user_input}"
        query_engine = custom_LLM_index.as_query_engine()
        response = query_engine.query(user_input)

        message = {"role": "assistant", "content": response.response}
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append(message)
        return message

    def load_chat_history(self, filename):
        try:
            with open(filename, 'r') as f:
                self.chat_history = json.load(f)
        except FileNotFoundError:
            pass

    def save_chat_history(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.chat_history, f)

bot = Chatbot(openai.api_key, index=custom_LLM_index)

bot.load_chat_history("chat_history.json")

# Terminal/Command-Line Based Chatbot Implementation
# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ["bye", "goodbye"]:
#         print("Bot: Goodbye!")
#         bot.save_chat_history("chat_history.json")
#         break
#     response = bot.generate_response(user_input)
#     print(f"Bot: {response['content']}")

# GUI using streamlit

import streamlit as st
import random
import time
from PIL import Image
import requests
from io import BytesIO
from st_clickable_images import clickable_images
from bs4 import BeautifulSoup


def imglink(word):
    url = f"https://www.google.com/search?q={word}&tbm=isch"  # the URL of the search result page

    response = requests.get(url)  # make a GET request to the URL
    soup = BeautifulSoup(response.text, "html.parser")  # parse the HTML content with BeautifulSoup

    # find the first image link by searching for the appropriate tag and attribute
    img_tag = soup.find("img", {"class": "yWs4tf"})

    if img_tag is not None:
        img_link = img_tag.get("src")
        print(img_link)  # print the first image link
        return img_link
    else:
        print("No image found on the page.")


# GUI created using streamlit

st.title("👘 TrendAI - Your Fashion Expert")
st.subheader("Welcome to TrendAI, your go-to fashion assistant to help you get trendy outfits catered to your unique preferences!\nMade with ❤️ by Ashutosh and Adarsh.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
prompt = st.chat_input("Suggest an outfit for a 20 year old woman for a picnic.")
if prompt:
    if prompt.lower() not in ["bye", "goodbye"]:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            response = bot.generate_response(prompt)
            assistant_response = response['content']
            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            # prompt design for outfit-component extraction for displaying respective components
            prompt_prefix = "Extract the fashion product names along with their attributes, if any, from the content given and provide them in a single line seperated by comma. Otherwise return the string \"nothing\" ."
            prompt_with_input = prompt_prefix + response['content']
            response = bot.generate_response(prompt_with_input)
            print(f"Bot: {response['content']}")

            outfit_components = response['content'].split(',')  # Split outfit components
            if outfit_components not in ["nothing"]:
                images_list=[]

                for component in outfit_components:
                    # Call the imglink function here for each component
                    img_link = imglink(component)
                    images_list.insert(0, img_link)

                # clicked = clickable_images(
                #     images_list,
                #     titles=[f"Image #{str(i)}" for i in range(len(images_list))],
                #     div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
                #     img_style={"margin": "5px", "height": "200px"},
                # )

                # container_style = (
                #     "display: flex; justify-content: center; align-items: center;"
                #     "margin: 20px; height: 300px;"
                # )

                # for i in range(len(images_list)):
                #     print(images_list[i])

                #for i in range(len(images_list)):
                #    st.image(images_list[i], caption=outfit_components[i], use_column_width="never", clamp=True)

                clicked = clickable_images(images_list, titles=[f"Image #{str(i)}" for i in range(len(images_list))],
                                           div_style={"display": "flex", "justify-content": "center",
                                                      "flex-wrap": "wrap"},
                                           img_style={"margin": "5px", "height": "200px"},
                                           )

            #st.markdown(f"Image #{clicked} clicked" if clicked > -1 else "No image clicked")

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    else:
        # Display assistant response in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            bot.save_chat_history("chat_history.json")
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = "Goodbye!"
            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})