import wget
from openai import OpenAI
import wget
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import os


def generate_story_board(prompt, model="dall-e-3", size="1024x1024", quality="standard", n=1):
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=n,
    )
    image_url = response.data[0].url
    return image_url

def download_image_from_url(image_url, save_path="output_story_board.png"):
    if os.path.exists(save_path):
        # Get the directory and filename from the save_path
        directory, filename = os.path.split(save_path)
        
        # Get the filename without extension
        filename_without_ext = os.path.splitext(filename)[0]
        
        # Generate a new filename by appending a number to the original filename
        i = 1
        while os.path.exists(os.path.join(directory, f"{filename_without_ext}_{i}.png")):
            i += 1
        
        # Update the save_path with the new filename
        save_path = os.path.join(directory, f"{filename_without_ext}_{i}.png")
    
    try:
        # Download the image using wget
        wget.download(image_url, save_path)
        print("Image downloaded successfully!")
    except Exception as e:
        print(f"Error downloading image: {str(e)}")
    
    return save_path
    

def visualize_image(image_path):
    st.image(image_path)


def upload_image_frame_reference():
    pass    

def main():
    st.title("StoryBoard-Maker")
    # prompt = st.text_input("Enter your story board prompt:")
    PROMPT_STORY_BOARD = "I want to visualize a storyboard about this\
    boy who picks up a book in the library and the book when set\
    on the table turns into a portal into a different dimension."


    PROMPT_SYSTEM = "You are an expert animator and sketch artist for movies and animations.\
        You will be prompted with images or scripts and your job is to generate perfect story\
        boards that match the user's requirements. You should pay special attention to\
        how many frames the user wants as well as which the style the user wants the\
        storyboard to be made of."
        
    num_frames = st.sidebar.number_input("#frames in the story board", step=1) # default 4
    style_description = st.sidebar.text_input("Describe the style of your story board") # "Cartoon minimalist black and white"
    story_board_description = st.text_input("Give the description of your story board in detail.") # "A man walks into a bar. The man sits down at the stool. The man orders a drink. The drink arrives."
    
    PROMPT_SAMPLE_TASK = f"Create storyboard in the style of {style_description} for this {story_board_description}.\
        The final storyboard should be strictly done in {num_frames} frames"
    
    gen_board = st.button("Generate story board")
    
    if gen_board:
        image_url = generate_story_board(PROMPT_SAMPLE_TASK)
        st.session_state["image_path"] = download_image_from_url(image_url)
        st.write("Your story board is ready!")
    if st.button("Visualize Story Board"):
        image_path = st.session_state.get("image_path")
        visualize_image(image_path)

if __name__=="__main__":
    
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

    if openai_api_key!="":
        os.environ["OPENAI_API_KEY"] = openai_api_key

        client = OpenAI()

        main()
    
