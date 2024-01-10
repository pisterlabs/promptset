import streamlit as st
from openai import OpenAIError
from util_toast import get_toast_message
import base64
import requests

def encode_image(image_path):
    """
    Encode the image to base64.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



def generate_character_description(image_path):
    """
    Generates a character description using GPT-4 vision.
    :param image_path: Path to the uploaded image
    :return: Character description
    """

    tst_start = get_toast_message("start", "Character Description")
    
    openai_api_key = st.secrets["openai"]["api_key"]
    model = "gpt-4-vision-preview"
    url = "https://api.openai.com/v1/chat/completions"
    content_type = "application/json"
    authorization = f"Bearer {openai_api_key}"

    sys_instructions = """
    You are an expert at describing images and creating effective prompts to use with DALL-E 3 image generator. Your sole job is to create a description for the main character of a childrens storybook from an image provided by a user. You will:
    - Identify the main person from the image
    - Create a magical, descriptive, detailed description about that main person in the image factoring into account looks, age, clothes, expressions, etc.
    - Review the description 3 times - each time enhancing for quality, details, and likeness
    - Perform one final check that the description is the most effective prompt that can be used by DALLE-3
    - RETURN / OUTPUT: You will return the response in the following format: "Do not modify or diversify this prompt: {{Character Description}} followed by the prompt followed by the character description 

    You will follow these set of rules ALWAYS:
    1. You will perform all steps internally
    2. You will never engage or consult with the user at any point unless it is submitting the final output. 
    3. You will create a description fit for a children's storybook
    4. Your description should be based on the image provided
    5. The stylistic nature of the description should create a character similar to those in Disney or Pixar movies / books
    6. You will only return the description as a text string and nothing else.
    """

    user_instructions = "Describe the main person in this image."

    base64_image = encode_image(image_path)
    img_url = f"data:image/jpeg;base64,{base64_image}"

    messages = [
        {"role": "system", "content": sys_instructions},
        {"role": "user", "content": [
            {"type": "text", "text": user_instructions},
            {"type": "image_url", "image_url": {"url": img_url}}
        ]}
    ]

    headers = {"Content-Type": content_type, "Authorization": authorization}
    payload = {"model": model, "messages": messages, "max_tokens": 300}
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        response_data = response.json()
        character_description = response_data['choices'][0]['message']['content']
        
        # Display toast message
        tst_end = get_toast_message("end", "Character Description")

        return character_description
    else:
        st.error("Failed to generate character description.")
        return None