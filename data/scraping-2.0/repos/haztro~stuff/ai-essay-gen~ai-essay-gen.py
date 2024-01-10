import openai
import os
import io
import warnings
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import cv2
import numpy as np
from moviepy.editor import *
from google.cloud import texttospeech

GC_KEY_FILENAME = "googlecloud_api_key.json"

class EssayGen(): 

    def __init__(self):
        self.openai_key = ''
        self.stable_diffusion_key = ''
        self.essay_prompt = ''
        self.image_prompt = ''
        self.image_style_prompt = ''
        self.essay_title = ''
        self.dir_path = ''
        self.text = []
        self.image_prompt_main = ''
        self.image_prompt_start = ''
        self.text_raw = ''

        self.read_keys()
        self.load_prompts()


    def generate_essay(self):
        self.generate_text()
        self.generate_audio()
        self.generate_images()
        self.generate_video()
        print("Done")


    def read_keys(self):
        with open('../api_keys/openai_api_key') as f:
            self.openai_key = f.readlines()[0].strip()

        with open('../api_keys/stablediffusion_api_key') as f:
            self.stable_diffusion_key = f.readlines()[0].strip()
        
        # For gpt
        openai.api_key = self.openai_key
        # For stable diffusion
        os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
        os.environ['STABILITY_KEY'] = self.stable_diffusion_key
        # For google cloud text-to-speech
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.abspath("../api_keys/"+GC_KEY_FILENAME)


    def load_prompts(self):
        with open('essay_prompt.txt', 'r') as f:
            self.essay_prompt = f.read()

        with open('image_prompt.txt', 'r') as f:
            self.image_prompt = f.read()

        with open('image_prompt_main.txt', 'r') as f:
            self.image_prompt_main = f.read()

        with open('image_style_prompt.txt', 'r') as f:
            self.image_style_prompt = f.read()


    def generate_text(self):
        print("Generating essay...")

        # Generate text
        completions = openai.Completion.create(
            engine="text-davinci-003",
            prompt=self.essay_prompt,
            max_tokens=512,
            temperature=0.85,
            presence_penalty=0.6,
            n=1,
            stop=None
        )

        self.text_raw = completions.choices[0].text
        self.text = completions.choices[0].text.split("\n")
        self.essay_title = self.text.pop(0).strip()
        self.dir_path = '../' + self.essay_title
        
        print("Title: " + self.essay_title)
        print("Generating audio...")

        # Setup directory 
        os.mkdir(self.dir_path)
        os.mkdir(self.dir_path + "/images")
        os.mkdir(self.dir_path + "/audio")


    def generate_audio(self):
        # Generate audio for essay title 
        self.gen_and_save_audio(self.essay_title, self.dir_path + '/title.mp3')

        # Generate rest of the audio
        file = open(self.dir_path + "/text.txt", "w")
        i = 0
        for p in self.text:
            if len(p) != 0 and p:
                file.write(p + "\n")
                self.gen_and_save_audio(p.strip(), self.dir_path + "/audio/" + str(i)+'.mp3')
                i += 1
        file.close()


    def gen_and_save_audio(self, text, filename):
        # Instantiates a client
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-GB", name="en-GB-Neural2-B"
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        with open(filename, "wb") as out:
            out.write(response.audio_content)


    def generate_images(self):
        print("Generating images...")

        # Setup connection to API
        stability_api = client.StabilityInference(
            key=os.environ['STABILITY_KEY'],
            verbose=True,
            engine="stable-diffusion-v1-5",
        )

        completions = openai.Completion.create(
            engine="text-davinci-003",
            prompt=self.image_prompt_main.format(self.text_raw),
            max_tokens=100,
            temperature=0.8,
            n=1,
            stop=None
        )

        self.image_prompt_start = completions.choices[0].text.strip()
        print(self.image_prompt_start)

        i = 0
        for p in self.text:
            if len(p) != 0 and p:
                self.generate_sd_image(stability_api, p.strip(), i)
                i += 1


    def generate_sd_image(self, client, prompt, index):
        # Generate image prompt with GPT
        completions = openai.Completion.create(
            engine="text-davinci-003",
            prompt=self.image_prompt.format(prompt.strip()),
            max_tokens=200,
            temperature=0.85,
            n=1,
            stop=None
        )

        image_prompt = self.image_prompt_start + "," + self.make_image_prompt_string(completions.choices[0].text)

        print(image_prompt)
        
        # Use image prompts to generate images using stable diffusion
        saved = False
        while not saved:
            try:
                # Set up our initial generation parameters.
                answers = client.generate(
                    prompt=image_prompt + "," + self.image_style_prompt,
                    steps=150,
                    cfg_scale=8.0,
                    width=512,
                    height=512,
                    samples=1,
                    sampler=generation.SAMPLER_K_DPMPP_2M
                )
                
                for resp in answers:
                    for artifact in resp.artifacts:
                        if artifact.finish_reason == generation.FILTER:
                            print("SAFETY FILTER ACTIVATED")
                            warnings.warn("Safety filter activated!")
                        elif artifact.type == generation.ARTIFACT_IMAGE:
                            img = Image.open(io.BytesIO(artifact.binary))
                            img.save(self.dir_path + "/images/" + str(index) + ".png")
                            saved = True
            except Exception:
                print("ERROR. Retrying...")


    def make_image_prompt_string(self, text):
        lines = text.strip().split("\n")
        data = {}
        for line in lines:
            key, value = line.split(":")
            key = key.strip()
            value = value.strip()
            data[key] = value

        output_string = ",".join(data.values())
        return output_string


    def generate_video(self):
        print("Generating video...")

        # Get the list of all files in the directory
        images = os.listdir(self.dir_path + '/images')
        audio = os.listdir(self.dir_path + '/audio')
        images.sort()
        audio.sort()
        images = [self.dir_path + '/images/'+ f for f in images]
        audio = [self.dir_path + '/audio/'+ f for f in audio]

        clips = []
        for i in range(len(images)):
            audio_clip = AudioFileClip(audio[i])
            image_clip = ImageClip(images[i])
            # use set_audio method from image clip to combine the audio with the image
            video_clip = image_clip.set_audio(audio_clip)
            # specify the duration of the new clip to be the duration of the audio clip
            video_clip.duration = audio_clip.duration
            # set the FPS to 1
            video_clip.fps = 1
            clips.append(video_clip)

        # Concatenate the clips into a single video
        video = concatenate_videoclips(clips)
        # Save the video to a file
        video.write_videofile(self.dir_path + '/video.mp4', temp_audiofile='temp-audio.m4a', remove_temp=True, codec="libx264", audio_codec="aac")


    def upload_to_youtube(self):
        pass




def run():
    obj = EssayGen()
    obj.generate_essay()


if __name__ == "__main__":
    run()

