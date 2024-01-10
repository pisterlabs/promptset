import os
import base64
import asyncio
import httpx
from openai import OpenAI
from openai import AsyncOpenAI
import json
import requests
from io import BytesIO
from dotenv import load_dotenv
from PIL import Image
import replicate
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
import utils.download_from_url as download
from pathlib import Path
import streamlit as st


# Load environment variables from .env file
load_dotenv()
async_client = AsyncOpenAI(
        # This is the default and can be omitted
        api_key=st.secrets["OPENAI_API_KEY"],
    )
client = OpenAI(
        # This is the default and can be omitted
        api_key=st.secrets["OPENAI_API_KEY"],
    )

def generate_example_prompt():
    """
    Generate an example prompt from the OpenAI API.
    :param api_key: OpenAI API key.
    :return: JSON formatted three example prompts
    """
    # get system prompt from txt file
    systemprompt = open("systemprompt_gen_examples.txt", "r")
    systemprompt = systemprompt.read()
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {
                    "role": "system",
                    "content": systemprompt
                },
                {
                    "role": "user",
                    "content": "Generate three very short example prompts for the Dreammachine"
                }
            ],
            response_format={ "type": "json_object" },
            temperature=1.0,
        )
        
        return response.choices[0].message.content

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def generate_story(story_prompt):
    """
    Generate a story with title, voiceover, and image prompts in JSON format.

    :param api_key: OpenAI API key.
    :param story_prompt: The initial prompt for the story.
    :return: JSON formatted story with title, voiceover, and image prompts.
    """


    
    # get system prompt from txt file
    systemprompt = open("systemprompt.txt", "r")
    systemprompt = systemprompt.read()

    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {
                    "role": "system",
                    "content": systemprompt
                },
                {
                    "role": "user",
                    "content": story_prompt
                }
            ],
            response_format={ "type": "json_object" },
        )
        
        return response.choices[0].message.content

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
def generate_voiceover(voiceover_text):
    """
    Generate a voiceover from text using the OpenAI API.
    """
    # make sure the directory exists
    os.makedirs("media/voiceover", exist_ok=True)
    speech_file_path = Path(__file__).parent / f"media/voiceover/speech_{voiceover_text[:25]}.mp3"
    response = client.audio.speech.create(
    model="tts-1-hd",
    voice="onyx",
    input=voiceover_text
    )

    response.stream_to_file(speech_file_path)
    path_str = str(speech_file_path)
    return path_str

async def get_image_from_DALL_E_3_API(user_prompt, image_dimension="1792x1024", image_quality="hd", model="dall-e-3", nb_final_image=1, response_format="b64_json"):
    response = await async_client.images.generate(
        model=model,
        prompt=user_prompt,
        size=image_dimension,
        quality=image_quality,
        n=nb_final_image,
        response_format=response_format,
    )

    # Assuming the response is a JSON object with the base64-encoded image
    # Adjust the following line based on the actual structure of the response
# Convert response to JSON
    image_data_json = json.loads(response.json())  # Adjust based on response structure

    # Print the JSON response to understand its structure but only the first 200 characters
    print(json.dumps(image_data_json, indent=2)[:200])

    # Extract base64 string correctly based on the JSON structure
    # Example: image_data_base64 = image_data_json['someKey']['nestedKey']
    image_data_base64 = image_data_json['data'][0]['b64_json']

    image_data = base64.b64decode(image_data_base64)

    # Open the image and save it to a file
    image = Image.open(BytesIO(image_data))
    file_path = f"media/images/img_{user_prompt[:50]}.png"

    # Ensure the directory exists or create it
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    image.save(file_path)
    return file_path

async def get_video_from_Replicate_API(image_path, video_length="25_frames_with_svd_xt"):
    url = await replicate.async_run(
        "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438",
        input={"input_image": open(image_path, "rb"), "video_length": video_length}
    )
    print(url)
    # Generate a unique output path based on the input image name
    base_name = os.path.basename(image_path)
    output_path = f"media/videos/video_{base_name[:-4]}.mp4"

    async with httpx.AsyncClient() as client:
        timeout = httpx.Timeout(100.0)  # Set timeout to 10 seconds
        response = await client.get(url, follow_redirects=True, timeout=timeout)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as video_file:
                async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                    video_file.write(chunk)
            return output_path
        else:
            print(f"Failed to download video. Status code: {response.status_code}")
            return None

def combine_videos_and_audio(video_paths, audio_path, output_path):
    """
    Combine multiple videos into one and add an audio track using MoviePy.

    :param video_paths: A list of paths to the video files.
    :param audio_path: Path to the audio file.
    :param output_path: Path where the output video will be saved.
    """
    # Load all the video clips
    video_clips = [VideoFileClip(path) for path in video_paths]

    # Concatenate the video clips
    final_clip = concatenate_videoclips(video_clips)

    # Load the audio file
    audio_clip = AudioFileClip(audio_path)

    # Set the audio of the concatenated clip as the audio clip
    final_clip = final_clip.set_audio(audio_clip)

    os.makedirs("media/videos", exist_ok=True)

    # Write the result to the output file
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

    # Close the clips
    final_clip.close()
    for clip in video_clips:
        clip.close()
    audio_clip.close()

    return output_path

async def generate_video(story_prompt):
    story_json = generate_story(story_prompt)
    print(story_json)
    story_json_dict = json.loads(story_json)

    title = story_json_dict["title"]
    st.header(title)
    col1, col2, col3 = st.columns(3)
    video_logline = story_json_dict.get("video_logline", "")
    col1.subheader("Video Logline")
    col1.write(video_logline)
    voiceover_text = story_json_dict["voiceover_text"]
    col2.subheader("Voiceover Text")
    col2.write(voiceover_text)
    visual_style = story_json_dict.get("visual_style", "")
    col3.subheader("Visual Style")
    col3.write(visual_style)

    speech_path = generate_voiceover(voiceover_text)
    display_generated_media('voiceover', lambda *args, **kwargs: speech_path, col2, col3)
    
    video_paths = []    
    

    if "clips" in story_json_dict:
        async def process_clip(clip_value, clip_index):
            
            col1, col2, col3 = st.columns(3)
            full_prompt = clip_value.get("image_prompt", "") + ' + ' + visual_style
            st.subheader(f"Clip {clip_index}")
            col1.write(full_prompt)
            # Generate image
            img_path = await get_image_from_DALL_E_3_API(full_prompt)
            display_generated_media(f'image_{clip_index}', lambda *args, **kwargs: img_path, col2, col3)
            # Generate video from image
            video_path = await get_video_from_Replicate_API(img_path)
            display_generated_media(f'video_{clip_index}', lambda *args, **kwargs: video_path, col2, col3)
            return video_path

        # Create tasks for each clip
        tasks = [process_clip(clip_value, i) for i, clip_value in enumerate(story_json_dict["clips"].values())]

        # Wait for all video tasks to complete
        video_paths = await asyncio.gather(*tasks)


    st.subheader("Final Video")

    if video_paths:
        output_video_path = f"media/videos/{title}.mp4"
        combine_videos_and_audio(video_paths, speech_path, output_video_path)
        print(f"Combined video and audio saved to {output_video_path}")
        st.write(f"Combined video and audio saved to {output_video_path}")        
        return output_video_path

    else:
        print("No videos were generated.")
        st.write("No videos were generated.")



def display_generated_media(key, generate_media_function, col_image, col_video, *args, **kwargs):
    # Generate media if not already done or if regeneration is requested
    if key not in st.session_state or st.session_state.get(f'regenerate_{key}', False):
        st.session_state[key] = generate_media_function(*args, **kwargs)
        st.session_state[f'regenerate_{key}'] = False

    media_path = st.session_state[key]

    # Display the generated media based on its file type
    if media_path:
        file_extension = os.path.splitext(media_path)[1].lower()
        if file_extension in ['.jpg', '.jpeg', '.png']:
            col_image.subheader("Generated Image:")
            col_image.image(media_path)
        elif file_extension == '.mp3':
            st.subheader("Generated Audio:")
            st.audio(media_path)
        elif file_extension == '.mp4':
            col_video.subheader("Generated Video:")
            col_video.video(media_path)

def list_video_files(directory):
    """List video files in the given directory."""
    video_extensions = ['.mp4', '.avi', '.mov']  # Add more extensions as needed
    return [file for file in os.listdir(directory) if os.path.splitext(file)[1] in video_extensions]

def display_videos(video_files):
    """Display videos in rows of 3."""
    with st.expander("Example Gallery"):
        for i in range(0, len(video_files), 3):
            cols = st.columns(3)
            for j in range(3):
                index = i + j
                if index < len(video_files):
                    cols[j].video(os.path.join('media/videos', video_files[index]))




def main():
    st.set_page_config(page_title="X-Fiction", page_icon="🎥")
    st.title("X-Fiction Video Generation 🎥")
    
    st.image("media/header3.png")
    
    st.write("This is a demo. Enter a story prompt and click the 'Start Generation' button to generate a video. Generations take about 12 minutes. Progress can be followed live. Powered by Stable Video Diffusion.")
    
    # Initialize or use existing session state
    if 'current_story_prompt' not in st.session_state:
        st.session_state['current_story_prompt'] = "Reactions from hell (Genre: Satire)"    

    
    # Default story prompt
    story_prompt = st.text_input("Enter a story prompt", st.session_state['current_story_prompt'])

    col1, col2, col3 = st.columns(3)

    if col3.button("Generate Example Prompt"):
        example_prompts_json = generate_example_prompt()
        if example_prompts_json:
            example_prompts = json.loads(example_prompts_json)
            col1, col2, col3 = st.columns(3)
            for i, (prompt_key, prompt_value) in enumerate(example_prompts.items()):
                
                if i == 0:
                    col1.subheader("Prompt 1")
                    col1.write(prompt_value)
                elif i == 1:
                    col2.subheader("Prompt 2")
                    col2.write(prompt_value)
                elif i == 2:
                    col3.subheader("Prompt 3")
                    col3.write(prompt_value)
    if col1.button("Start Generation (0,80€)"):
        with st.spinner("Generating video..."):
            video_path = asyncio.run(generate_video(story_prompt))
            col1, col2 = st.columns(2)
            col1.success("Video generated!")
            col2.video(video_path)   
    # Directory containing the videos
    video_directory = 'media/videos'

    # Get a list of video files
    videos = list_video_files(video_directory)

    # Display videos
    display_videos(videos)
        
   

if __name__ == "__main__":
    main()