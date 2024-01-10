from sys_templates import *
import openai
import re
import streamlit as st
import replicate
from elevenlabs import voices, generate, stream
import base64
from pydub.utils import mediainfo
import time

class SimpleStoryAdapter:
    def __init__(self, story):
        self.story = story

    def get_latest_messages(self):
        return self.story.messages[-2:]
    
    def get_latest_message_from_system(self):
        for message in reversed(self.story.messages):
            if message["sender"] == "system":
                return message["content"]
        return None

    def submit_choice(self, choice):
        self.story.add_choice(choice)

    def advance_story(self, choice):
        self.story.advance_story(choice)
        
class Story:
    def __init__(self, user_preferences):
        self.messages = []  # Start with an initial system message
        self.user_preferences = user_preferences

    def add_choice(self, choice):
        self.messages.append({"sender": "user", "content": choice})
        self.advance_story(choice)

    def get_last_system_message(self):
        for message in reversed(self.messages):
            if message["sender"] == "system":
                return message["content"]
        return "Initial segment to kickstart the narrative."

    def format_system_message(self):
        # Check if the summary attribute exists in the object; if not, initialize with a placeholder
        if not hasattr(self, "summary"):
            self.summary = "First chunk, begin the story!"
        
        last_chunk = self.get_last_system_message()
        user_choice = self.messages[-1]["content"] if self.messages and self.messages[-1]["sender"] == "user" else "No decision made yet."
        
        # print("CURRENT TEMPLATE:", system_template.format(preferences=self.user_preferences, story_so_far=self.summary, last_chunk=last_chunk, user_choice=user_choice))
        return system_template.format(preferences=self.user_preferences, story_so_far=self.summary, last_chunk=last_chunk, user_choice=user_choice)

    def generate_text(self, prompt):
        system_message = prompt["system_message"]
        user_message = prompt["previous_choice"]

        response_content = GPT(system_message, user_message)

        return response_content

    def get_summary(self, new_chunk):
        if not hasattr(self, "summary"):
            self.summary = "First chunk, begin the story!"  # Default for the first iteration
        
        prompt = summary_template.format(previous_summary=self.summary, new_chunk=new_chunk)
        new_summary = self.generate_summary(prompt)
        self.summary = new_summary
        return new_summary

    def generate_summary(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.1,  # You may adjust this as per your requirements
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Generate the updated summary."}
            ]
        )
        return response['choices'][0]['message']['content']

    def advance_story(self, choice):
        system_message = self.format_system_message()
        prompt = {
            "system_message": system_message,
            "previous_choice": choice
        }
        new_chunk = self.generate_text(prompt)
        self.get_summary(new_chunk)  # Update the summary after generating a new chunk
        self.messages.append({"sender": "system", "content": new_chunk})
        # print("System message added:", self.messages[-1]["content"])

class AlignmentSim(Story):
    def __init__(self, user_preferences):
        super().__init__(user_preferences)

    def get_last_system_message(self):
        for message in reversed(self.messages):
            if message["sender"] == "system":
                return message["content"]
        return "Initial segment to initiate the alignment simulation."

    def format_system_message(self):
        if not hasattr(self, "progress_summary"):
            self.progress_summary = "Start of the alignment simulation."
        
        last_alignment_chunk = self.get_last_system_message()
        user_last_decision = self.messages[-1]["content"] if self.messages and self.messages[-1]["sender"] == "user" else "No decision made yet."
        
        return alignment_template.format(preferences=self.user_preferences, alignment_progress_so_far=self.progress_summary, last_alignment_chunk=last_alignment_chunk, user_last_decision=user_last_decision)

    def get_progress_summary(self, new_chunk):
        if not hasattr(self, "progress_summary"):
            self.progress_summary = "Start of the alignment simulation."
        
        prompt = alignment_summary_template.format(previous_summary=self.progress_summary, new_chunk=new_chunk)
        new_progress_summary = self.generate_summary(prompt)
        self.progress_summary = new_progress_summary
        return new_progress_summary

    def advance_simulation(self, choice):
        system_message = self.format_system_message()
        prompt = {
            "system_message": system_message,
            "previous_choice": choice
        }
        new_chunk = self.generate_text(prompt)
        self.get_progress_summary(new_chunk)
        self.messages.append({"sender": "system", "content": new_chunk})


def extract_options_from_chunk(chunk):
    pattern = r"OPTION (\d): ([^\n]+)"
    matches = re.findall(pattern, chunk)
    
    if len(matches) != 2:
        # Handle the situation where we don't get two clear options
        return None, None
    
    option_1, option_2 = matches[0][1], matches[1][1]
    return option_1, option_2

def GPT(system_message, user_message):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=0.5,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        st.write(f"Error with OpenAI call: {e}")
        return None

# if GPT3.5 shits the bed:

# def get_image_from_prompt(prompt, client):
#     output = client.run(
#         "stability-ai/sdxl:2b017d9b67edd2ee1401238df49d75da53c523f36e363881e057f5dc3ed3c5b2",
#         input={"prompt": prompt},
#     )
#     return output

# def generate_illustration(story_chunk, style, client):
#     # Simply append the style to the story chunk to create the illustration prompt
#     illustration_prompt = story_chunk + "\n" + style
#     print("ILLUSTRATION PROMPT:", illustration_prompt)
    
#     # Pass the combined prompt to get the image
#     image_output = get_image_from_prompt(illustration_prompt, client)
    
    return image_output

def get_audio_duration(data: bytes) -> float:
    with open("temp_audio.mp3", "wb") as f:
        f.write(data)
    info = mediainfo("temp_audio.mp3")
    duration = float(info["duration"])
    return duration

def autoplay_audio(data: bytes):
    b64 = base64.b64encode(data).decode()
    md = f"""
    <audio id="audioElement" controls autoplay="true" style="display: none;">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)
    
    # Estimate audio duration using average bitrate of 192 kbps
    bits = len(data) * 8
    duration_seconds = (bits / 192000) * 1.53 

    # Return the estimated duration for further use
    return duration_seconds

def get_image_from_prompt(prompt, client):
    output = client.run(
        "stability-ai/sdxl:2b017d9b67edd2ee1401238df49d75da53c523f36e363881e057f5dc3ed3c5b2",
        input={"prompt": prompt},
    )
    return output

def generate_illustration_prompt(story_chunk, style):
    # Format the illustrator system message with the user-specified style.
    system_message = illustrator.format(style=style)
    
    # Make the call to GPT-3.5-turbo
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.3,  # Adjust as per your requirements
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": story_chunk}
        ]
    )
    return response['choices'][0]['message']['content']

def generate_illustration(story_chunk, style, client):
    # Generate the detailed illustration prompt from the story chunk
    illustration_prompt = generate_illustration_prompt(story_chunk, style)
    # print("ILLUSTRATION PROMPT:", illustration_prompt)
    # Pass the generated prompt to get the image
    image_output = get_image_from_prompt(illustration_prompt + 'NO TEXT, NO WORDS', client)
    
    return image_output

def generate_story_prompt(preferences):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.25,  # You may adjust this as per your requirements
        messages=[
            {"role": "system", "content": preferences},
            {"role": "user", "content": "Generate the preferences."}
        ]
    )
    return response['choices'][0]['message']['content']

def generate_title(title_prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.25,  # You may adjust this as per your requirements
        messages=[
            {"role": "system", "content": title_prompt},
            {"role": "user", "content": "Generate the title."}
        ]
    )
    return response['choices'][0]['message']['content']

# alignment stuff!!!!!!

def generate_alignment_prompt(preferences):
    # print('INPUTS:', preferences)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.25,  # You may adjust this as per your requirements
        messages=[
            {"role": "system", "content": preferences},
            {"role": "user", "content": "Generate the preferences."}
        ]
    )
    return response['choices'][0]['message']['content']

