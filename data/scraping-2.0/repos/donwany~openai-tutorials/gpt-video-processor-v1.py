import cv2
import base64
from openai import OpenAI
from moviepy.editor import VideoFileClip, AudioFileClip
from dotenv import load_dotenv
import os
import time

load_dotenv()


class VideoProcessor:
    def __init__(self, api_key: str, model: str = 'gpt-4-vision-preview'):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def calculate_video_length(self, video):
        """calculate video length"""
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        return length / fps

    def read_video_frames(self, video):
        """read video frames"""
        base64_frames = []
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        return base64_frames

    def create_voiceover_script(self, video_length, base64_frames):
        """create voice over"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        f"These are frames of a video. Create a short voiceover script in the style of a football commentator for {video_length:.2f} seconds. Only include the narration. Don't talk about the view",
                        *map(lambda x: {"image": x, "resize": 768}, base64_frames[0::25]),
                    ]
                }
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content

    def generate_audio_file(self, model, voice, text, file_path):
        """generate audio file"""
        response = self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text
        )
        response.stream_to_file(file_path)


def main():
    api_key = os.getenv("api_key")
    video_processor = VideoProcessor(api_key)

    video_path = "football.m4v"
    video = cv2.VideoCapture(video_path)

    video_length_seconds = video_processor.calculate_video_length(video)
    print(f'Video length: {video_length_seconds:.2f} seconds')

    base64_frames = video_processor.read_video_frames(video)
    video.release()
    print(len(base64_frames), "frames read.")

    voiceover_script = video_processor.create_voiceover_script(video_length_seconds,
                                                               base64_frames)
    time.sleep(2)  # Ensure a brief pause between chat completion and audio generation

    speech_file_path = "football.mp3"
    video_processor.generate_audio_file("tts-1", "onyx", voiceover_script, speech_file_path)

    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(speech_file_path)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile("football_with_commentator.m4v", codec='libx264', audio_codec='aac')

    video_clip.close()
    audio_clip.close()
    final_clip.close()


if __name__ == '__main__':
    main()
