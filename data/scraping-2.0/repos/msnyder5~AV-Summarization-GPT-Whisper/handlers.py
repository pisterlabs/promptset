import whisper
from pytube import YouTube
from transcript import Transcript
from config import *
import openai

openai.api_key = OPENAI_API_KEY

def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

class SummaryHandler:
    def __init__(self,
                 transcript: Transcript,
                 title: str,
                 texttokenlimit: int = TEXT_TOKEN_LIMIT,
                 responsetokenlimit: int = RESPONSE_TOKEN_LIMIT,
                 frontsentencespreserved: int = FRONT_SENTENCES_PRESERVED,
                 backsentencespreserved: int = BACK_SENTENCES_PRESERVED):
        self.transcript: Transcript = transcript
        self.title: str = title
        self.texttokenlimit: int = texttokenlimit
        self.responsetokenlimit: int = responsetokenlimit
        self.frontsentencespreserved: int = frontsentencespreserved
        self.backsentencespreserved: int = backsentencespreserved
    
    @property
    def prompt(self) -> str:
        text, truncated_flag = self.transcript.get_truncated_text(self.texttokenlimit, self.frontsentencespreserved, self.backsentencespreserved)
        if truncated_flag:
            return TRUNCATED_PROMPT.format(text=text, title=self.title, num_front_sentences=self.frontsentencespreserved, num_back_sentences=self.backsentencespreserved)
        else:
            return UNTRUNCATED_PROMPT.format(text=text, title=self.title)
    
    def get_summary(self) -> str:
        completion = openai.Completion.create(
            model="text-davinci-003",
            prompt=self.prompt,
            max_tokens=RESPONSE_TOKEN_LIMIT,
        )
        return completion['choices'][0]['text'].strip()

class YouTubeHandler:
    def __init__(self,
                 youtubeurl: str,
                 youtube_path: str = YOUTUBE_PATH,
                 youtubecaptions: bool = YOUTUBE_CAPTIONS,
                 youtubeautocaptions: bool = YOUTUBE_AUTO_CAPTIONS):
        self.youtubeurl: str = youtubeurl
        self.youtube_path: str = youtube_path
        self.youtubecaptions: bool = youtubecaptions
        self.youtubeautocaptions: bool = youtubeautocaptions
    
    def get_summary_handler(self) -> SummaryHandler:
        video = YouTube(self.youtubeurl)
        title = video.title
        if self.youtubecaptions and 'en' in video.captions.keys():
            transcript = Transcript.from_yt_captions(video.captions['en'], title)
            return SummaryHandler(transcript, title)
        elif self.youtubeautocaptions and 'a.en' in video.captions.keys():
            transcript = Transcript.from_yt_captions(video.captions['a.en'], title)
            return SummaryHandler(transcript, title)
        else:
            download_stream = video.streams.filter(only_audio=True, file_extension='mp4').first()
            filepath = download_stream.download(output_path=self.youtube_path)
            newhandler = AVHandler(filepath, title)
            return newhandler.get_summary_handler()
    
class AVHandler:
    def __init__(self,
                 filepath: str,
                 title: str,
                 whispermodel: str = WHISPER_MODEL):
        self.filepath: str = filepath
        self.whispermodel: str = whispermodel
        self.title: str = title
    
    def get_summary_handler(self) -> SummaryHandler:
        vprint('Loading Whisper Model.')
        model: whisper.Whisper = whisper.load_model(self.whispermodel)
        vprint('Whisper Model Loaded. Transcribing AV file. This may take a minute.')
        result = model.transcribe(self.filepath)
        vprint('Done transcribing.')
        transcript = Transcript.from_whisper_result(result, self.title)
        return SummaryHandler(transcript, self.title)