import os
from dotenv import load_dotenv
from typing import List, Optional
import shutil
from mutagen.mp3 import MP3
from moviepy import editor
from PIL import Image
from moviepy.editor import TextClip, CompositeVideoClip, ImageClip
from moviepy.editor import *
import logging
from moviepy.video.io.VideoFileClip import VideoFileClip
import glob
from PIL import Image
import openai
import requests

# Load the environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), '.env')
load_dotenv(env_path)

class Frames:
    INSTAGRAM_REEL = (1080, 1920)  # size in pixels
    YOUTUBE_REEL = (1920, 1080)    
    TIKTOK_REEL = (1080, 1920)     
    INSTAGRAM_POST = (1080, 1080)  

class VideoGenerator:

    # If the app is ran in docker, the db folder is copied into the app folder
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    storage_dir = "app/db/storage" if os.path.exists(os.path.join(root_dir, "app/db/storage")) else "db/storage"

    video_storage_path = os.path.join(root_dir, storage_dir, "videos")
    image_storage_path = os.path.join(root_dir, storage_dir, "images")
    audio_storage_path = os.path.join(root_dir, storage_dir, "audios")
    subtitle_storage_path = os.path.join(root_dir, storage_dir, "subtitles")

    def __init__(self, 
                 video_path: str = video_storage_path,
                 audio_path: str = audio_storage_path,
                 image_path: str = image_storage_path,
                 subtitle_path: str = subtitle_storage_path,
                 openai_api_key: Optional[str] = None, 
                 stable_diff_api_key: Optional[str] = None):
        """
        :param src: List[str] would be a list of image file locations [db/storage/images/image1.png, ] or it can be
        a string "generate" which would use DALLE or Stable diffusion to generate new sets of images.

        :param video_path: Where the newly generated video is stored
        :param audio_path: Where the newly generated audio is stored
        :param image_path: Where the newly generated audio is stored
        :param openai_api_key - api key for OpenAI
        :param stable_diff_api_key - api key for Stable Diffusion
        """

        self.video_path = video_path
        self.audio_path = audio_path
        self.image_path = image_path
        self.subtitle_path = subtitle_path

        openai.api_key = os.environ.get("OPENAI_KEY", openai_api_key)
        
    def upload_images(self, image_files: List[str], destination_folder: str):
        """
        :param image_files: List of paths of images to upload
        :param destination_folder: Folder to which images will be uploaded
        """
        for image_file in image_files:
            shutil.copy(image_file, destination_folder)
    
    
    def resize_image(self, image_path: str, size: tuple = Frames.INSTAGRAM_REEL) -> str:
        """Resize an image to the specified size and save it.

        Args:
            image_path: The path to the image to resize.
            size: The desired size as a tuple (width, height).

        Returns:
            The path to the saved image.
        """
        img = Image.open(image_path)
        img = img.resize(size)
        new_image_path = image_path.rsplit('.', 1)[0] + '_resized.' + image_path.rsplit('.', 1)[1]
        img.save(new_image_path)
        return new_image_path

    def read_audio_file(self, audio_file_path: str):
        """
        :param audio_file_path: Path of the audio file to read
        :return: Length of the audio file in seconds
        """
        audio = MP3(audio_file_path)
        return audio.info.length

    def create_video(self, image_files: List[str], audio_file_path: str = None, video_size: tuple = Frames.INSTAGRAM_REEL):
        """
        :param image_files: List of paths of images to use for the video
        :param audio_file_path: Path of the audio file to use for the video
        :param video_size: Tuple , defaults to size for IG reel
        """
        if not audio_file_path:
            # TODO: Current saving of audio is to a file called 'test.mp3' so if its not provided we will just grab that one. This needs to be updated.
            audio_file_path = os.path.join(self.audio_path, "test.mp3")

        # Calculate duration per image
        audio_length = self.read_audio_file(audio_file_path)
        duration_per_image = audio_length / len(image_files)
        
        # Open, resize and save images as gif
        images = [Image.open(image).resize(video_size, Image.ANTIALIAS) for image in image_files]
        images[0].save("temp.gif", save_all=True, append_images=images[1:], duration=int(duration_per_image)*1000)

        # Set output file name
        output_file_name = os.path.splitext(os.path.basename(audio_file_path))[0] + '.mp4'
        output_video_path = os.path.join(self.video_path, output_file_name)
        
        # Combine audio and gif to create video
        video = editor.VideoFileClip("temp.gif")
        audio = editor.AudioFileClip(audio_file_path)
        final_video = video.set_audio(audio)
        final_video.write_videofile(output_video_path, fps=30, codec="libx264")
        
        # Delete temporary gif
        os.remove("temp.gif")
    
    def generate_video_static(self, audio_file_path: str = None, static_image: Optional[str] = None):
        """
        :param audio_file_path: Path of the audio file to use for the video
        :param static_image: Path of the static image, defaults to black
        """
        # Check static image
        if not static_image:
            static_image = os.path.join(self.image_path, "black_image.png")
            
        if not audio_file_path:
            # TODO: Current saving of audio is to a file called 'test.mp3' so if its not provided we will just grab that one. This needs to be updated.
            audio_file_path = os.path.join(self.audio_path, "test.mp3")

        # Load the audio file
        audio = AudioFileClip(audio_file_path)

        # Load the static image file and convert it to a clip with the duration of the audio
        img_clip = ImageClip(static_image, duration=audio.duration)

        # Set the audio of the video to the audio clip
        video = img_clip.set_audio(audio)

        # Create file output path
        audio_name = os.path.splitext(os.path.basename(audio_file_path))[0] + ".mp4"
        video_file_path = os.path.join(self.video_path, audio_name)

        # Write the final video file
        video.write_videofile(video_file_path, codec='libx264', temp_audiofile='temp-audio.m4a', remove_temp=True, audio_codec='aac', fps=24)
        

    def generate_subtitles(self, auido_file_path: str, subtitle_file_path: str, language='en'):
        """
        :param language: Language code of the audio file's language (default is 'en' for English)
        """
        # Generate a subtitle file name (without path) from audio file
        subtitle_file_name = os.path.splitext(os.path.basename(subtitle_file_path))[0] + ".srt"

        # If subtitle file does not exist in the directory, generate it
        if not glob.glob(f"{self.subtitle_storage_path}/{subtitle_file_name}"):
            # TODO: Figure out how to generate subtitles. This seems to be the fix
            # https://stackoverflow.com/questions/66977227/could-not-load-dynamic-library-libcudnn-so-8-when-running-tensorflow-on-ubun
            pass
    
    def generate_images_with_dalle(self, api_key: str, prompt: str, size: tuple = Frames.INSTAGRAM_POST):

        openai.api_key = api_key

        size_for_openai = "1024x1024"
        
        try:
            generation_response = openai.Image.create(
                prompt=prompt,
                n=1,
                size=size_for_openai,
                response_format="url"
            )

            # save the image
            generated_image_name = "generated_image.png"  # any name you like; the filetype should be .png
            generated_image_filepath = os.path.join(self.image_path, generated_image_name)

            generated_image_url = generation_response["data"][0]["url"]  # extract image URL from response
            generated_image = requests.get(generated_image_url).content  # download the image

            with open(generated_image_filepath, "wb") as image_file:
                image_file.write(generated_image)
              # write the image to the file
            
            print("Sucess!")

            return generated_image_filepath
        
        except Exception as e:
            print(e)

if __name__ == "__main__":

    img = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "db/storage/images/black_image.png")
    aud = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "db/storage/audios/test.mp3")
    vg = VideoGenerator()
    # vg.generate_video_static(aud, img)

    # Location of all the images
    image_path =  os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "db/storage/images/")
    image_list = [file for file in os.listdir(image_path) if file.startswith("rand")]

    # vg.create_video(image_files=image_list,
    #                 audio_file_path=aud)

    prompt = "rusty old phone booth"

    vg.generate_images_with_dalle(prompt=prompt)
    
