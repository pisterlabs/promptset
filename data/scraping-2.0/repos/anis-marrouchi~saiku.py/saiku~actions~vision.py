import os
import cv2
import base64
import requests
import youtube_dl
import openai
from pathlib import Path
from typing import List

class VisionAction:
    def __init__(self, agent):
        self.agent = agent
        self.name = 'openai_vision'
        self.description = 'Analyze a video or image using OpenAI Vision.'
        self.parameters = [
            {'name': 'source', 'type': 'string', 'required': True,
             'description': 'Local path or URL of the video/image.'},
            {'name': 'request', 'type': 'string', 'required': True,
             'description': 'The user request to be sent to OpenAI Vision.'}
        ]

    async def run(self, args: dict) -> str:
        source = args['source']
        openai_request = args['request']
        is_url = source.startswith('http://') or source.startswith('https://')
        file_path = source if not is_url else self.download_media(source)

        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            base64_frames = [self.encode_image_to_base64(file_path)]
        else:
            frames_path = 'tmp/video/frames'
            os.makedirs(frames_path, exist_ok=True)
            self.extract_frames(file_path, frames_path)
            base64_frames = self.encode_frames_to_base64(frames_path)

        return await self.analyze_media(base64_frames, openai_request)

    def download_media(self, url: str) -> str:
        if 'youtube.com' in url or 'youtu.be' in url:
            os.makedirs('tmp/videos', exist_ok=True)
            ydl_opts = {'outtmpl': 'tmp/videos/%(title)s.%(ext)s'}
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                info = ydl.extract_info(url, download=False)
                return ydl.prepare_filename(info)
        else:
            response = requests.get(url)
            response.raise_for_status()
            os.makedirs('tmp/downloads', exist_ok=True)
            file_path = f'tmp/downloads/{Path(url).name}'
            with open(file_path, 'wb') as file:
                file.write(response.content)
            return file_path

    def extract_frames(self, video_path: str, output_path: str):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(f'{output_path}/frame_{frame_count:03d}.jpg', frame)
            frame_count += 1
        cap.release()

    def encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def encode_frames_to_base64(self, output_path: str) -> List[str]:
        return [self.encode_image_to_base64(f'{output_path}/{f}')
                for f in os.listdir(output_path) if f.endswith('.jpg')]

    async def analyze_media(self, base64_frames: List[str], openai_request: str) -> str:
        

        prompt_messages = [
            {
                "role": "user",
                "content": [
                    openai_request,  # Using the openai_request string as part of the prompt
                    *map(lambda x: {"image": x, "resize": 768}, base64_frames[::50]),
                ],
            },
        ]

        params = {
            "model": "gpt-4-vision-preview",
            "messages": prompt_messages,
            "max_tokens": 200,
        }

        response = openai.chat.completions.create(**params)
        return response.choices[0].message.content
