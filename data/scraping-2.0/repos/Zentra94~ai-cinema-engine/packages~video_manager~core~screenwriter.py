import re
from datetime import datetime
import google.cloud.texttospeech as tts
from google.oauth2 import service_account
from configs import PATH_STATICS_FONTS

from moviepy.editor import (AudioFileClip,
                            AudioClip,
                            ImageClip,
                            VideoFileClip,
                            concatenate_videoclips,
                            CompositeAudioClip)
import openai
import replicate
import requests
import numpy as np
from io import BytesIO
import os
from natsort import natsorted
from rake_nltk import Rake
import pickle
import spacy
from PIL import (Image, ImageDraw, ImageFont)
import textwrap
import soundfile as sf

from typing import List

REPLICATE_STABILITY_VERSION = "f178fa7a1ae43a9a9af01b833b9d2ecf97b1bcb0acfd2dc5dd04895e042863f1"
REPLICATE_STABILITY_ENGINE = "stability-ai/stable-diffusion"
REPLICATE_RIFFUSION_VERSION = "riffusion/riffusion:8cf61ea6c56afd61d8f5b9ffd14d7c216c0a93844ce2d82ac1c9ecc9c7f24e05"


def _add_static_image_to_audio(image_path, audio_path, output_path):
    """Create and save a video file to `output_path` after combining a static image that
    is located in `image_path` with an audio file in `audio_path`."""

    # create the audio clip object
    audio_clip = AudioFileClip(str(audio_path))
    # create the image clip object
    image_clip = ImageClip(str(image_path), ismask=False)
    # use set_audio method from image clip to combine the audio with the image
    video_clip = image_clip.set_audio(audio_clip)
    # specify the duration of the new clip to be the duration of the audio clip
    video_clip.duration = audio_clip.duration
    # set the FPS to 1
    video_clip.fps = audio_clip.duration * 2
    # write the resulting video clip
    video_clip.write_videofile(str(output_path))


def _paragraphs_splitter(text, rules_kw=None, min_length=5):
    """ Splits a given text into paragraphs based on the specified rules and minimum
    paragraph length.

    Args:
        text (str): The text to be split into paragraphs.
        rules_kw (list[str], optional): A list of keywords that indicate the end of a
            paragraph (default=["\\n", ",", ".", ";"]).
        min_length (int, optional): The minimum number of words in a paragraph
            (default=5).

    Returns:
        dict: A dictionary where the keys are "p1", "p2", ..., and the values are the
            corresponding paragraphs of the input text.
    """

    if rules_kw is None:
        rules_kw = ["\n", ",", ".", ";"]
    paragraphs_dict = {}
    text_split = text.split()

    n_paragraphs = 0
    counter = 0
    new_paragraph = []
    for i, word in enumerate(text_split):
        new_paragraph.append(word)
        counter += 1
        if counter > min_length:
            for kw in rules_kw:
                if kw in word:
                    counter = 0
                    n_paragraphs += 1
                    paragraphs_dict["p{}".format(n_paragraphs)] = " ".join(
                        new_paragraph)
                    new_paragraph = []

                    break
        if i + 1 == len(text_split) and len(new_paragraph) > 0:
            n_paragraphs += 1
            paragraphs_dict["p{}".format(n_paragraphs)] = " ".join(new_paragraph)

    return paragraphs_dict


def _enhance_thumbnail(image,
                       title="Video generated 100% with Artificial Intelligence.",
                       path_fonts=PATH_STATICS_FONTS / "Playfair.ttf",
                       title_font_size=120,
                       alpha=100,
                       header_color=(255, 255, 0)
                       ):
    title = title.strip()
    image.putalpha(alpha)
    draw_image = ImageDraw.Draw(image)
    W, _ = image.size
    img_width = int(W * 0.03)

    # Define font sizes

    title_font = ImageFont.truetype(str(path_fonts), title_font_size)

    header_offset = 15
    # Wrap text and add it to the thumbnail
    for line in textwrap.wrap(title, width=int(img_width * 0.6)):
        _, _, w, _ = draw_image.textbbox((0, 0), line, font=title_font)

        xy = ((W - w) / 2, header_offset)

        draw_image.text(xy, line,
                        font=title_font,
                        fill=header_color,
                        spacing=10,
                        align="center")
        header_offset += title_font_size
    return draw_image._image


class ScreenWriter:
    """A class for generating video scripts using AI-powered text generation engines."""

    def __init__(self,
                 base_path: str,
                 gcp_sa_key: str,
                 replicate_api_key: str,
                 open_ai_key: str,
                 engine: str = "gpt-3.5-turbo",
                 verbose: int = 1,
                 replicate_stability_engine: str = REPLICATE_STABILITY_ENGINE,
                 replicate_stability_version: str = REPLICATE_STABILITY_VERSION,
                 replicate_riffusion_version: str = REPLICATE_RIFFUSION_VERSION,
                 image_height: int = 768,
                 image_width: int = 768,
                 timeout: int = 60 * 60,
                 default_tags: List[str] = None,
                 prompt_engineering_mapping_list: List[str] = None,
                 prompt_engineering_default_list: dict = None,
                 prompt_engineering_nlp=spacy.load('en_core_web_sm')):
        """Initialize ScreenWriter.

        Args:
            base_path (str): The base path for the generated video script files.
            gcp_sa_key (str): The path to the Google Cloud Platform service account key
                file.
            replicate_api_key (str): The API key for the Replicate API.
            open_ai_key (str): The API key for the OpenAI API.
            engine (str, optional): The AI-powered text generation engine to use.
                Defaults to "gpt-3.5-turbo".
            verbose (int, optional): The verbosity level of the class. Defaults to 0.
            replicate_stability_engine (str, optional): The Replicate engine to use.
                Defaults to REPLICATE_ENGINE.
            replicate_stability_version (str, optional): The version of the Replicate
                engine to use. Defaults to REPLICATE_VERSION.
            image_height (int, optional): The height of the images used in the video
                script. Defaults to 768.
            image_width (int, optional): The width of the images used in the video
                script. Defaults to 768.
            timeout (int, optional): The maximum amount of time in seconds to wait for a
                response from the text generation engine. Defaults to 3600.
            default_tags (List[str], optional): The default tags to use for the video
                script. Defaults to None.
        """

        if prompt_engineering_mapping_list is None:
            prompt_engineering_mapping_list = ['NOUN', 'VERB', 'ADJ', "PROPN"]

        if prompt_engineering_default_list is None:
            prompt_engineering_default_list = {'PROPN': ["person"],
                                               "NOUN": ["facts"],
                                               'VERB': ["smiling"],
                                               'ADJ': ["happy"]}

        if default_tags is None:
            default_tags = ["artificial intelligence", "future", "machines",
                            "stable diffusion", "chatGPT", "youtubeislife",
                            "subscriber", "youtubeguru", "youtubecontent",
                            "newvideo", "subscribers", "youtubevideo",
                            "youtub", "youtuber", "youtubevideos"]

        self.tags = default_tags
        self.description = ""

        openai.api_key = open_ai_key
        openai.timeout = timeout

        self.engine = engine
        self.verbose = verbose
        self.base_path = base_path / "m{}".format(
            datetime.now().strftime("%Y%m%d%H%M%S"))
        self.replicate_client = replicate.Client(api_token=replicate_api_key)
        self.replicate_model = self.replicate_client.models.get(
            replicate_stability_engine)
        self.replicate_version = self.replicate_model.versions.get(
            replicate_stability_version)
        self.replicate_riffusion_version = replicate_riffusion_version

        self.image_dimensions = "{}x{}".format(image_width, image_height)

        self.gcp_service_account = service_account.Credentials.from_service_account_file(
            gcp_sa_key)

        self.title = None
        self.paragraphs = {}
        self.cover = None

        self.prompt_engineering_mapping_list = prompt_engineering_mapping_list
        self.prompt_engineering_default_list = prompt_engineering_default_list
        self.prompt_engineering_nlp = prompt_engineering_nlp

    def generate_text(self, prompt, **kwargs):
        """Generates a response to the given prompt using OpenAI's chatbot API.

        Args:
            prompt (str): The prompt to use for generating the response.
            **kwargs: Additional arguments to pass to the OpenAI API.

        Returns:
            str: The generated response to the prompt.

        Raises:
            OpenAIError: If there was an error with the OpenAI API.
        """

        messages = [{"role": "user",
                     "content": prompt}]

        response = openai.ChatCompletion.create(
            model=self.engine,
            messages=messages,
            **kwargs)

        return response["choices"][0]["message"]["content"]

    def generate_title_from_prompt(self,
                                   prompt,
                                   **kwargs
                                   ):
        title = self.generate_text(prompt=prompt,
                                   **kwargs)

        if self.verbose > 0:
            print(title)

        return re.sub(r'[\n\t\r\'\"]', '', title)

    def generate_content_from_prompt(self, prompt, **kwargs):
        content = self.generate_text(prompt=prompt,
                                     **kwargs)

        if self.verbose:
            print(content)

        return content

    def generate_image_from_prompt(self, prompt, **kwargs):

        pred = self.replicate_version.predict(prompt=prompt,
                                              image_dimensions=self.image_dimensions,
                                              safety_checker=None,
                                              **kwargs)

        image = Image.open(BytesIO(requests.get(pred[0]).content))

        if self.verbose > 1:
            image.show()

        return image

    def text_to_speech(self, text, voice_name=None):
        """Synthesizes a given text into speech using Google Cloud Text-to-Speech API.

        Args:
            text (str): The input text to synthesize into speech.
            voice_name (Optional[str]): The name of the voice to use for synthesizing
                speech. If None, a default voice is used.

        Returns:
            bytes: The synthesized audio content as bytes.

        Raises:
            ValueError: If the given text is empty or too long.
            google.api_core.exceptions.InvalidArgument: If there is an error with the
                Google Cloud Text-to-Speech API.
        """

        if voice_name is None:
            voice_name = "en-US-News-N"

        language_code = "-".join(voice_name.split("-")[:2])

        text_input = tts.SynthesisInput(text=text)
        voice_params = tts.VoiceSelectionParams(
            language_code=language_code, name=voice_name)
        audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)

        client = tts.TextToSpeechClient(credentials=self.gcp_service_account)
        response = client.synthesize_speech(input=text_input, voice=voice_params,
                                            audio_config=audio_config)

        return response.audio_content

    def update_tags_from_content(self, content,
                                 max_tags=5,
                                 rake_nltk_var=Rake(),
                                 ):

        clean_content = " ".join(re.sub('[^a-zA-Z ]+', '', content).split())
        rake_nltk_var.extract_keywords_from_text(clean_content)
        keyword_extracted = rake_nltk_var.get_ranked_phrases()
        keyword_extracted = [k for k in set(keyword_extracted)]
        keyword_extracted = sorted(keyword_extracted, key=len)
        max_tags = min(max_tags, len(keyword_extracted))
        self.tags += keyword_extracted[:max_tags]

    def update_description_from_content(self, content,
                                        description_kwargs=None,
                                        content_to_description_prompt=None):

        if description_kwargs is None:
            description_kwargs = {"max_tokens": 64 * 2,
                                  "temperature": 0.9}

        if content_to_description_prompt is None:
            content_to_description_prompt = "Summarize the content:"

        description_prompt = "{} {}".format(content_to_description_prompt, content)

        description = self.generate_content_from_prompt(prompt=description_prompt,
                                                        **description_kwargs)

        description = """Video Generate using 100% Artificial Intelligence. \n {}
        """.format(description)

        self.description = description

    def prompt_engineering(self,
                           text,
                           max_tokens=32,
                           quality="HD, dramatic lighting, detailed, realistic",
                           init_prompt="describes an image",
                           update_prompt_engineering_default_list=False):

        mapping_kw = {}
        doc = self.prompt_engineering_nlp(text)
        for mk in self.prompt_engineering_mapping_list:
            tokens = [token.text for token in doc if token.pos_ == mk]
            if len(tokens) == 0:
                tokens = self.prompt_engineering_default_list[mk]
            mapping_kw[mk] = tokens

        if update_prompt_engineering_default_list:

            for mk, values in mapping_kw.items():
                if len(values) > 0:
                    self.prompt_engineering_default_list[mk] = values

        propns = " and ".join(mapping_kw["PROPN"])
        nouns = " and ".join(mapping_kw["NOUN"])
        verbs = " and ".join(mapping_kw["VERB"])
        adjectives = " and ".join(mapping_kw["ADJ"])

        screen_writer_prompt = f"""{init_prompt} that has as subject {propns} 
        nouns: {nouns}, verbs: {verbs} and adjectives {adjectives}"""

        prompt = self.generate_text(prompt=screen_writer_prompt,
                                    max_tokens=max_tokens)

        prompt += f" {quality}."

        return prompt

    def generate_wav_file_music_from_prompt(self,
                                            prompt_a,
                                            prompt_b="Classic music and Electronic",
                                            alpha=0.5,
                                            denoising=0.9,
                                            num_inference_steps=30,
                                            seed_image_id="vibes",
                                            path=None,
                                            min_duration_sec=60 * 5):
        if path is None:
            path = str(self.base_path / "bg_music.wav")

        replicate_input = {
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
            "alpha": alpha,
            "denoising": denoising,
            "num_inference_steps": num_inference_steps,
            "seed_image_id": seed_image_id
        }

        duration = 0
        counter = 0
        samplerate = 0
        datas = []

        while duration < min_duration_sec:
            pred = self.replicate_client.run(self.replicate_riffusion_version,
                                             input=replicate_input)

            response = requests.get(pred.get("audio")).content
            bytes_content = BytesIO(response)
            data, samplerate = sf.read(bytes_content)
            datas.append(data)

            duration += len(data) / samplerate
            counter += 1

            if self.verbose > 0:
                print(f"Generating music: try nb. {counter}. total duration: {duration}")

        final_music = np.concatenate(datas)

        sf.write(path, final_music, samplerate=samplerate, format="wav")

        print(f"Music save in {path}")

        return path

    def fit(self,
            title_prompt,
            music_path=None,
            title_kwargs=None,
            content_kwargs=None,
            description_kwargs=None,
            image_kwargs=None,
            rules_kw=None,
            min_length=5,
            voice_name=None,
            title_to_content_prompt="Generate a Youtube plain script about",
            content_to_description_prompt=None):

        """Trains the ScreenWriter model using the provided parameters and generates a
        final video based on the script.

        Args:
            title_prompt (str): The initial prompt for generating the video title.
            music_path (str): The path to the audio file to use as background music.
            title_kwargs (Optional[dict]): The parameters to use for generating the
                video title. Defaults to None.
            content_kwargs (Optional[dict]): The parameters to use for generating the
                video content. Defaults to None.
            description_kwargs (Optional[dict]): The parameters to use for generating
                the video description. Defaults to None.
            image_kwargs (Optional[dict]): The parameters to use for generating the
                video images. Defaults to None.
            rules_kw (Optional[dict]): The parameters to use for splitting the generated
                content into paragraphs. Defaults to None.
            min_length (int): The minimum length of a paragraph. Defaults to 5.
            voice_name (Optional[str]): The name of the voice to use for the
                text-to-speech synthesis. Defaults to None.
            title_to_content_prompt (str): The prompt to use for generating the video
                content based on the title. Defaults to "Generate a YouTube script
                about".
            content_to_description_prompt (Optional[str]): The prompt to use for
                generating the video description based on the content. Defaults to None.
        """

        if music_path is None:
            music_path = self.base_path / "bg_music.wav"

        if content_kwargs is None:
            content_kwargs = {"max_tokens": 256 * 2,
                              "temperature": 0.4,
                              }
        if title_kwargs is None:
            title_kwargs = {"max_tokens": 32,
                            "temperature": 0.9}

        if image_kwargs is None:
            image_kwargs = {"num_inference_steps": 50}

        os.mkdir(path=self.base_path)
        paragraphs_path = self.base_path / "paragraphs"

        os.mkdir(path=paragraphs_path)

        self.title = self.generate_title_from_prompt(prompt=title_prompt,
                                                     **title_kwargs)

        content_prompt = "{} {}".format(title_to_content_prompt, self.title)

        self.paragraphs["p0"] = self.title

        content = self.generate_content_from_prompt(prompt=content_prompt,
                                                    **content_kwargs)

        self.update_description_from_content(content=content,
                                             description_kwargs=description_kwargs,
                                             content_to_description_prompt=content_to_description_prompt)

        self.update_tags_from_content(content=content)

        content = _paragraphs_splitter(text=content,
                                       rules_kw=rules_kw,
                                       min_length=min_length)

        self.paragraphs.update(content)

        # Combine speech + images

        for p, txt in self.paragraphs.items():

            prompt = self.prompt_engineering(text=txt,
                                             update_prompt_engineering_default_list=p == "p0")

            new_image = self.generate_image_from_prompt(prompt=prompt,
                                                        **image_kwargs)

            image_path = paragraphs_path / "{}_image.png".format(p)
            new_image.save(image_path)

            if p == "p0":
                self.cover = _enhance_thumbnail(image=new_image)
                cover_path = paragraphs_path / "cover.png".format(p)
                self.cover.save(cover_path)

            new_speech = self.text_to_speech(text=txt,
                                             voice_name=voice_name)

            speech_path = paragraphs_path / "{}_speech.wav".format(p)

            with open(speech_path, "wb") as out:
                out.write(new_speech)

            output_path = paragraphs_path / "{}_image_speech.mp4".format(p)

            _add_static_image_to_audio(image_path=image_path,
                                       audio_path=speech_path,
                                       output_path=output_path)
            if self.verbose > 0:
                print("Paragraph {} saved in {}".format(p, output_path))

        # Merge Videos

        consolidate_list = []

        for root, dirs, files in os.walk(paragraphs_path):
            files = natsorted(files)
            for file in files:
                if file.endswith("_image_speech.mp4"):
                    file_path = os.path.join(root, file)
                    video = VideoFileClip(file_path)
                    consolidate_list.append(video)

        final_clip = concatenate_videoclips(consolidate_list)
        full_speech = final_clip.audio
        full_speech.write_audiofile(str(self.base_path / "speech.wav"), fps=44100)

        # Add Background Music
        # TODO: Add AI music engine:
        #  https://replicate.com/riffusion/riffusion/api

        self.generate_wav_file_music_from_prompt(prompt_a=self.title,
                                                 path=music_path)

        audio_clip = AudioFileClip(str(music_path))
        audio_clip = audio_clip.volumex(0.05)
        audio_clip = audio_clip.subclip(final_clip.start, final_clip.end)
        final_audio = CompositeAudioClip([final_clip.audio, audio_clip])
        final_clip = final_clip.set_audio(final_audio)
        final_clip.write_videofile(str(self.base_path / "final.mp4"))

        with open(self.base_path / "ScreenWriter.pkl", 'wb') as of:
            pickle.dump(self, of)
