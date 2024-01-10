from datetime import datetime
import os
import time
import openai
import json
import urllib.request
from gtts import gTTS
import psutil
from pydub import AudioSegment
from pydub.playback import play

from PIL import Image

from mutagen.mp3 import MP3
import imageio
from moviepy.editor import TextClip, CompositeVideoClip, concatenate_videoclips, VideoFileClip, AudioFileClip, ImageClip
from moviepy.video.tools.subtitles import SubtitlesClip
from pathlib import Path


config = json.load(open("config.json", "r"))


class InfiniteContent:
    def __init__(self, seed, display, api_key) -> None:
        self.show = seed
        self.seed = f"write me a story titled '{seed}' with dialogue"
        self.continuation = ""
        self.context = []
        self.scene_paths = []
        self.display = display
        self.filename = f"{self.show}-{self.get_unique_id()}.mp4".replace(" ", "_")
        self.scene_count = 0

        openai.api_key = api_key

    def generate_image(self, text, model=None):
        if len(text) > 200:
            text = text[:200]
        response = openai.Image.create(
            prompt=f"scene from {self.show}: {text}",
            n=1,
            size="x".join([str(x) for x in config["size"]]),
        )
        filename = f"image/{self.get_unique_id()}.png"
        urllib.request.urlretrieve(response["data"][0]["url"], filename)

        return filename

    def generate_gpt3_response(self, text, model=None):
        completions = openai.Completion.create(
            engine='text-davinci-003',
            temperature=config["temperature"],
            prompt=text,
            max_tokens=1000,
            n=1,
            stop=None,
            model=model
        )

        return completions.choices[0].text

    def start_training(self, dataset):
        upload_response = openai.File.create(
            file="\n".join([json.dumps(datapoint)
                            for datapoint in dataset]).encode('utf-8'),
            purpose='fine-tune'
        )

        file_id = upload_response.id
        fine_tune_response = openai.FineTune.create(
            training_file=file_id, model=self.open_ai_model if self.open_ai_model else "davinci")

        return fine_tune_response.id

    def get_training_status(self, fine_tuning_id):
        response = openai.FineTune.retrieve(
            id=fine_tuning_id)

        if response.fine_tuned_model is not None:
            return response.fine_tuned_model, True

        response = openai.FineTune.list_events(
            id=fine_tuning_id)

        return response.data, False

    def get_voice(self, text, voice):
        try:
            voice_obj = gTTS(text=text, lang='en', slow=False)
        except AssertionError:
            return None

        file = f"audio/{self.get_unique_id()}.mp3"
        voice_obj.save(file)

        return file

    def play(self, voice, image, text):
        print(f"\n\n{text}\n\n")
        img = Image.open(image)
        img.show()
        if voice is not None:
            audio = AudioSegment.from_file(open(voice, 'rb'), format="mp3")
            play(audio)
        else:
            time.sleep(2)
        # hide image
        for proc in psutil.process_iter():
            if proc.name() == "display":
                proc.kill()
        img.close()

    def generate_scene(self, voice, image, text):
        if voice is not None:
            audio = MP3(voice)
            duration = audio.info.length
            audio = AudioFileClip(voice)
        else:
            duration = 1000

        video = ImageClip(image, duration=duration)

        final_video = video.set_audio(audio) if voice is not None else video

        # def gen(txt): return TextClip(txt, font='Arial', fontsize=24,
        #                               color='white', method='caption', size=tuple(config['size']))
        # subs = [((0, duration), text)]
        # subtitle_clip = SubtitlesClip(subs, gen)
        # final_video = CompositeVideoClip(
        #     [final_video, subtitle_clip.set_pos(('center', 'bottom'))])
        path = f"scenes/{self.get_unique_id()}.mp4"
        final_video.write_videofile(fps=60, codec="libx264", filename=path)
        self.scene_paths.append(path)

    def join_scenes(self):
        final_video = concatenate_videoclips(
            [VideoFileClip(x) for x in self.scene_paths])
        final_video.write_videofile(
            fps=60, codec="libx264", filename=f"results/{self.filename}")

    def get_unique_id(self):
        return f"{self.show}_{datetime.timestamp(datetime.now())}".replace('.', '')

    def main(self, episode_no=10):
        for ep in range(episode_no):
            completion = self.generate_gpt3_response(
                self.continuation+self.seed)

            self.continuation = ""
            lines = completion.split("\n")
            for n, line in enumerate(lines):
                if line.replace(" ", "") == "":
                    continue
                if len(line.split(":")) == 0:
                    if len(self.context) > 3:
                        self.context.pop(0)
                    self.context.append(line)

                voice = self.get_voice(line, None)
                context_text = '\n'.join(self.context)
                image = self.generate_image(
                    f"{self.seed}\n{context_text}\n{line}")
                if self.display:
                    self.play(
                        voice=voice,
                        image=image,
                        text=line
                    )

                self.generate_scene(voice, image, line)
                self.scene_count += 1

                if n > (len(lines) - 4):
                    self.continuation += f"\n{line}"

            self.continuation += "\n\n continue"

        self.exit()

    def exit(self):
        self.join_scenes()

        if not config["keep_images"]:
            os.system("rm image/*")
        if not config["keep_audio"]:
            os.system("rm audio/*")
        if not config["keep_scenes"]:
            os.system("rm scenes/*")


if __name__ == "__main__":
    content = InfiniteContent(input("Seed: "), input(
        "Show images? enter for no") != "", input("API key:"))
    content.main(episode_no=int(input("Ep no: ")))
