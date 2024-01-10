from ai4teaching import DocumentReader
from ai4teaching import log
from ai4teaching.utils import make_sure_directory_exists
import os

class VideoReader(DocumentReader):
    def __init__(self):
        super().__init__()
        self.step_name = "video_reader"
        self.original_document_type = "video"
        
    def read(self, video_uri, tmp_download_dir="tmp"):
        log(f"Reading content from video >{video_uri}<", type="debug")

        make_sure_directory_exists(tmp_download_dir)

        # Is this a local file? Or a YouTube-URL?
        if video_uri.startswith("http") and "youtube" in video_uri:
            audio_file = self._get_audio_from_youtube(video_uri, tmp_download_dir)
        else:
            # Local file
            audio_file = self._get_audio_from_local_file(video_uri, tmp_download_dir)

        # Transcribe audio to text
        transcript_segments = self._transcribe_audio(audio_file)

        # Read the text from the video file
        output_json = self._get_json_output_template(video_uri, self.original_document_type, self.step_name)
        output_json["segments"] = transcript_segments
        output_json["metadata"]["video_length"] = transcript_segments["metadata"]["duration"]

        # Remove temporary file and folder
        os.remove(audio_file)
        os.rmdir(tmp_download_dir)
        
        return output_json

    def _get_audio_from_youtube(self, youtube_url, tmp_download_dir):
        log(f"Downloading audio from YouTube >{youtube_url}<", type="debug")
        from pytube import YouTube
        yt = YouTube(youtube_url)
        output_file = os.path.join(tmp_download_dir, f"audio.mp4")
        audio_stream = yt.streams.filter(only_audio=True)[-1]
        audio_stream.download(filename=output_file)
        return output_file
    
    def _get_audio_from_local_file(self, local_file, tmp_download_dir):
        from moviepy.editor import VideoFileClip
        video = VideoFileClip(local_file)
        audio = video.audio
        output_file = os.path.join(tmp_download_dir, f"audio.mp3")
        audio.write_audiofile(output_file)
        return output_file
    
    def _transcribe_audio(self, audio_file):
        
        from openai import OpenAI
        client = OpenAI()
        audio = open(audio_file, "rb")
        response = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio, 
            response_format="verbose_json"
            )
        
        audio.close()

        # Initialize summary JSON with mandatory fields
        transcript_document = {}

        # Add additional fields into existing JSON
        transcript_document["text"] = response.text
        transcript_document["metadata"] = {
            "language" : response.language,
            "duration" : response.duration
        }
        transcript_segments = []
        transcript_document["content"] = transcript_segments

        for s in response.segments:
            segment = {}
            segment["metadata"] = {
                "segment_id" : s["id"],
                "start" : s["start"],
                "end" : s["end"]
            }
            segment["text"] = s["text"]
            transcript_segments.append(segment)

        return transcript_document
        