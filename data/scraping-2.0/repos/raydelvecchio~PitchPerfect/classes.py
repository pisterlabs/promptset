import requests
from moviepy.editor import *
# import noisereduce as nr
# import librosa
# import soundfile as sf
import shutil
import openai
import re
import os
import json
from dotenv import load_dotenv
load_dotenv('.env')


class Pulze:
    def __init__(self):
        openai.api_key = os.environ['Pulze_Key']
        openai.api_base = "https://api.pulze.ai/v1"
        self.messages = []
        self.audience = ""

    def __call_llm(self, prompt: str) -> str:
        """
        Calls the LLM with Pulze and returns the response. Stores all prompts/responses in context.
        """
        self.messages.append({"role": "user", "content": prompt})

        chat_response = openai.ChatCompletion.create(
            model="pulze-v0",
            max_tokens=2500,
            messages=self.messages
        )

        llm_response = chat_response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": llm_response})

        return llm_response

    def configure_initial_prompts(self, audience: str) -> str:
        """
        Sets up the hyperprompt and context, then loads them into the LLM's memory for it to recall later.
        """
        self.audience = audience
        hyperprompt = "You are a professional speaking coach who helps people improve their presentations skills. I will give you a transcript of a presentation, including the text as well as a field pause_time, representing the amount of time between each spoken word. Please give helpful and concise tips to the person pitching to improve the content of the presentation. Also, try to list specific timing and cadence issues if you see them in words where there is too long, too little, or an otherwise abnormal pause. Provide your feedback in the form of ONE bulleted list."
        context_prompt = f"The intended audience for this presentation is {audience}; please consider this in your feedback. The next prompt I give you will be this presentation, including text and pause_time for each word. Do you understand? Respond only yes or no."
        response = self.__call_llm(hyperprompt + context_prompt)
        return response

    def generate_feedback(self, transcript: str, length: int, shorten: bool = True) -> tuple:
        """
        Given the DeepGram transcript, generate feedback for the user. Returns a tuple of (feedback, new script). Takes
        in a length parameter that dictates roughly how long the response should be, in words. If we want to shorten
        the length to a set amount, we iteratively prompt it to make it shorter!
        """
        feedback = self.__call_llm(transcript)
        del self.messages[0:2]  # remove the hyper-prompt to save on tokens; not needed anymore
        rewrite_prompt = f"You are presenting to {self.audience}. Given your improvements, please write a new script for this presenter to follow that makes all necessary changes to the presentation. Ensure that you highlight the most important part of the presentation a lot. ONLY include the script for the presentation, making sure to NOT include things like captions, pauses, or who is speaking. Please include line breaks where pauses are relevant."
        new_script = self.__call_llm(rewrite_prompt)

        if shorten:
            shorten_prompt = f"It is VERY important that this script is shortened to at MOST {length} words. Focus on key points about the business, but remember we're presenting to {self.audience}. Please rewrite the script to match this requirement, again ONLY outputting the shorter script."
            del self.messages[0:2]  # remove the original transcript / feedback from the messages list (context)
            new_script = self.__call_llm(shorten_prompt)

        new_script = re.sub(r'\[(.*?)\]|\((.*?)\)', '', new_script)
        return feedback, new_script

    def reset_context(self):
        self.messages = []


class DeepGram:
    def __init__(self):
        self.key = os.environ['Deepgram_Key']

    def transcribe(self, filename: str) -> tuple:
        """
        Returns a summary of the response received from transcribing audio with the DeepGram API, as well as the
        number of words in the transcript.
        """
        url = (f"https://api.deepgram.com/v1/listen?filler_words=true"
               f"&summarize=v2")
        headers = {
            "content-type": "audio/mpeg",
            "Authorization": f"Token {self.key}"
        }

        with open(filename, 'rb') as f:
            payload = f.read()

        response: dict = requests.post(url, data=payload, headers=headers).json()
        data = response['results']['channels'][0]['alternatives']
        for entry in data:
            entry.pop('confidence', None)
            last_end = 0
            for word_dict in entry['words']:
                word_dict.pop('confidence', None)
                word_dict.pop('punctuated_word', None)
                word_dict['pause_time'] = round(float(word_dict['start']) - last_end, 2)  # pause between this/last word
                last_end = float(word_dict['end'])
                word_dict.pop('start', None)
                word_dict.pop('end', None)

        num_words = len(data[0]['transcript'].split(" "))
        return str(data[0]).replace("\"", "").replace("'", ""), num_words


class InputProcessor:
    def __init__(self):
        pass

    @staticmethod
    def process_input(filename: str, filter_noise=False) -> tuple:
        """
        Given an input video file, split them up into their video and audio components. Returns a tuple of
        (audio file, video file). Can optionally filter out noise if we set filter_noise=True. If the input is an
        AUDIO file (mp3), simply just return the audio file and nothing about the video file.
        """
        if '.mp3' in filename:
            audio_name = f'splits/{filename}'
            shutil.copy2(filename, audio_name)

            # if filter_noise:
            #     y, sr = librosa.load(audio_name, sr=None)
            #     reduced_noise = nr.reduce_noise(y=y, sr=int(sr))
            #     sf.write(audio_name, reduced_noise, int(sr))

            return audio_name, ""
        else:
            name = re.sub(r'\..*$', '', filename)

            video = VideoFileClip(filename)

            audio = video.audio
            audio_name = f'splits/{name}.mp3'
            audio.write_audiofile(audio_name)

            video_without_audio = video.without_audio()
            video_name = f'splits/{name}.mp4'
            video_without_audio.write_videofile(video_name, audio_codec='none')

            audio.close()
            video.close()
            video_without_audio.close()

            # if filter_noise:
            #     y, sr = librosa.load(audio_name, sr=None)
            #     reduced_noise = nr.reduce_noise(y=y, sr=int(sr))
            #     sf.write(audio_name, reduced_noise, int(sr))

            return audio_name, video_name


class Labs11:
    def __init__(self, username, store_file='keystore.json'):
        self.store_file = store_file
        self.username = username
        self.key = os.environ['11labs_Key']
        if not os.path.exists(store_file):
            with open(store_file, "w") as file:
                json.dump({}, file, indent=4)

    def __create_new_voice(self, audio_file: str) -> str:
        """
        Given an audio file, create a new voice in 11labs and return the voice ID. First checks our datastore to
        ensure we aren't generating multiple audio voices in 11labs for each user.
        """
        with open(self.store_file, "r") as file:
            data = dict(json.load(file))
        if self.username in data:
            return data[self.username]

        url = "https://api.elevenlabs.io/v1/voices/add"
        headers = {
            "Accept": "application/json",
            "xi-api-key": f"{self.key}"
        }
        data = {
            'name': f"Username_{self.username}",
            # 'labels': '{"accent": "American"}',
            'description': 'Ted AI hackathon generated voice'
        }
        files = [
            ('files', (audio_file, open(audio_file, 'rb'), 'audio/mpeg')),
        ]
        response = requests.post(url, headers=headers, data=data, files=files)
        voice_id = response.json()['voice_id']

        # updating our user -> voice ID dictionary
        with open(self.store_file, "r") as file:
            data = json.load(file)
        data[self.username] = voice_id
        with open(self.store_file, "w") as file:
            json.dump(data, file, indent=4)

        return voice_id

    def __generate_audio(self, voice_id: str, script: str, chunk_size: int = 1024) -> str:
        """
        Given a voice ID and a script, generate an MP3 file of that voice saying the given script.
        Returns filename of the generated audio's location.
        """
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": f"{self.key}"
        }
        data = {
            "text": script,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.6,
                "similarity_boost": 0.75
            }
        }

        response = requests.post(url, json=data, headers=headers)
        filename = f"audio_outputs/{self.username}_Output.mp3"
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        return filename

    def generate_custom_response(self, audio_file: str, script: str) -> str:
        """
        First creates a voice in 11labs with the given audio file, then speaks it with a script.
        """
        voice_id = self.__create_new_voice(audio_file)
        return self.__generate_audio(voice_id, script)


class Tester:
    @staticmethod
    def test_11labs():
        script = "Today I will be pitching my startup Crux. Thank you! I love analyzing pitch decks."
        audio_file = "test_files/test_audio1.mp3"
        username = "Ray"
        labs = Labs11(username)
        labs.generate_custom_response(audio_file, script)

    @staticmethod
    def test_input_processor_video():
        video_file = "test_files/Bad_Pitch.mov"
        InputProcessor.process_input(video_file)

    @staticmethod
    def test_input_processor_audio():
        audio_file = "test_files/test_audio1.mp3"
        InputProcessor.process_input(audio_file)

    @staticmethod
    def test_transcribe():
        audio_file = "test_files/test_audio1.mp3"
        dg = DeepGram()
        print(dg.transcribe(audio_file))

    @staticmethod
    def test_pulze():
        transcript = """{transcript: This is Ke. Today, we are pitching our new app called Pitch Perfect. That helps you as a presenter, improve your presentation skills and also improve the materials that youre presenting. Pitch gets to the crux of whats important in your presentation, and it helps you amplify that more effectively., words: [{word: this, speak_time: 0.16, pause_time: 0.6}, {word: is, speak_time: 0.08, pause_time: 0.0}, {word: ke, speak_time: 0.24, pause_time: 0.24}, {word: today, speak_time: 0.24, pause_time: 1.52}, {word: we, speak_time: 0.16, pause_time: 0.24}, {word: are, speak_time: 0.4, pause_time: 0.0}, {word: pitching, speak_time: 0.5, pause_time: 0.0}, {word: our, speak_time: 0.32, pause_time: 0.46}, {word: new, speak_time: 0.4, pause_time: 0.0}, {word: app, speak_time: 0.5, pause_time: 0.0}, {word: called, speak_time: 0.24, pause_time: 0.14}, {word: pitch, speak_time: 0.24, pause_time: 0.08}, {word: perfect, speak_time: 0.48, pause_time: 0.08}, {word: that, speak_time: 0.32, pause_time: 0.57}, {word: helps, speak_time: 0.48, pause_time: 0.0}, {word: you, speak_time: 0.32, pause_time: 0.0}, {word: as, speak_time: 0.16, pause_time: 0.0}, {word: a, speak_time: 0.32, pause_time: 0.0}, {word: presenter, speak_time: 0.4, pause_time: 0.0}, {word: improve, speak_time: 0.24, pause_time: 0.4}, {word: your, speak_time: 0.5, pause_time: 0.0}, {word: presentation, speak_time: 0.32, pause_time: 0.14}, {word: skills, speak_time: 0.5, pause_time: 0.0}, {word: and, speak_time: 0.24, pause_time: 0.77}, {word: also, speak_time: 0.48, pause_time: 0.0}, {word: improve, speak_time: 0.24, pause_time: 0.0}, {word: the, speak_time: 0.4, pause_time: 0.0}, {word: materials, speak_time: 0.4, pause_time: 0.0}, {word: that, speak_time: 0.16, pause_time: 0.0}, {word: youre, speak_time: 0.32, pause_time: 0.0}, {word: presenting, speak_time: 0.24, pause_time: 0.0}, {word: pitch, speak_time: 0.5, pause_time: 0.81}, {word: gets, speak_time: 0.24, pause_time: 0.21}, {word: to, speak_time: 0.16, pause_time: 0.0}, {word: the, speak_time: 0.24, pause_time: 0.0}, {word: crux, speak_time: 0.5, pause_time: 0.0}, {word: of, speak_time: 0.32, pause_time: 0.61}, {word: whats, speak_time: 0.48, pause_time: 0.0}, {word: important, speak_time: 0.24, pause_time: 0.0}, {word: in, speak_time: 0.24, pause_time: 0.0}, {word: your, speak_time: 0.5, pause_time: 0.0}, {word: presentation, speak_time: 0.24, pause_time: 0.14}, {word: and, speak_time: 0.08, pause_time: 0.16}, {word: it, speak_time: 0.24, pause_time: 0.0}, {word: helps, speak_time: 0.32, pause_time: 0.0}, {word: you, speak_time: 0.5, pause_time: 0.0}, {word: amplify, speak_time: 0.24, pause_time: 0.21}, {word: that, speak_time: 0.5, pause_time: 0.0}, {word: more, speak_time: 0.32, pause_time: 0.21}, {word: effectively, speak_time: 0.48, pause_time: 0.0}]}
         """
        audience = "investors"
        p = Pulze()
        init = p.configure_initial_prompts(audience)
        print(init)
        fix, better = p.generate_feedback(transcript, 50)
        print(fix)
        print("")
        print(better)
        print("")
        print(p.messages)


if __name__ == "__main__":
    pass
