import io
import os
import time
from typing import List, Dict, Generator
from dotenv import load_dotenv
from threading import Thread
import queue

import openai
import whisper
from elevenlabs import set_api_key, generate, stream, play

from .models import Session
from .chat_utils import chat_stream

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
set_api_key(os.getenv('ELEVENLABS_API_KEY'))


class AudioBuffer:
    def __init__(self):
        self.buffer = queue.Queue(maxsize=4)  # Adjust the buffer size as needed

    def add_audio(self, audio):
        self.buffer.put(audio, block=True)

    def get_audio(self):
        return self.buffer.get(block=True)


class InterviewerVoice:
    def __init__(self, tts_engine):
        self.tts_engine = tts_engine
        self.audio_buffer = AudioBuffer()

    def pre_fetch_audio(self, text_stream: Generator[str, None, None]):
        for text_chunk in text_stream:
            self.tts_engine.pre_fetch(text_chunk, self.audio_buffer)

    def speak_stream(self, text_stream: Generator[str, None, None]):
        start_time = time.time()
        pre_fetch_thread = Thread(target=self.pre_fetch_audio, args=(text_stream,))
        pre_fetch_thread.start()
        
        while pre_fetch_thread.is_alive() or not self.audio_buffer.buffer.empty():
            audio = self.audio_buffer.get_audio()
            play(audio)
        end_time = time.time()
        print(f"Total Time for Speaking: {end_time - start_time:.4f} seconds")

        return text_stream


class ElevenLabsTTS:
    def __init__(self, voice_id: str, model: str = "eleven_monolingual_v1"):
        self.voice_id = voice_id
        self.model = model
        self.partial_sentence = ""

    def pre_fetch(self, text_chunk: str, audio_buffer: AudioBuffer, voice_id: str = None):
        self.partial_sentence += text_chunk
        if self.partial_sentence.endswith('.') or self.partial_sentence.endswith('?') or self.partial_sentence.endswith('!'):
            start_time = time.time()
            audio = generate(
                text=self.partial_sentence.strip(),
                voice=voice_id or self.voice_id,
                model=self.model,
                stream=True,
            )
            audio_buffer.add_audio(audio)
            # SEND SENTENCE TO DATABASE

            self.partial_sentence = ""


def interview_reply(
    role: str, job_description: str, company_name: str,
    resume: str, name: str,
    interview_so_far: List[Dict[str, str]], 
    n: int, 
    model: str,
    tts_engine: ElevenLabsTTS
):
    messages = [
        {"role": "system", "content": f"""You are an interviewer, conducting behavioral interviews to select the most skilled and well-rounded candidates for the role of {role} at {company_name}. You generate interview questions given a job description, the resume of an interviewee to that job, and the interview so far. Read carefully the job description and associated resume, as well as the instructions that follow. However, note that you are the expert of the interview process, so the following should be taken as guidelines, not as strict rules.

        Job Description:
        \"\"\"
        {job_description}
        \"\"\"

        {name}'s Resume:
        \"\"\"
        {resume}
        \"\"\"

        This interview will last ~{n} minutes.

        A few notes to help you find avenues of discussion:
        - You may compare the resume to the job description to see if the interviewee has the necessary skills. If they do not, you may ask them about it, and about what they wish to do to remedy the situation.
        - You should visit these 4 major areas of discussion:
            - experience
            - skills
            - values
            - personality
        """},
    ]
    messages += interview_so_far

    start_time = time.time()
    text_stream = chat_stream(
        messages=messages,
        model=model,
    )
    end_time = time.time()
    print(f"Time for Text Generation: {end_time - start_time:.4f} seconds")
    
    voice = InterviewerVoice(tts_engine)
    voice.speak_stream(text_stream)


def generate_interview_question(session: Session):
    session = Session.objects.get(session_id=session.session_id)
    role = session.interview.job_title
    job_description = session.interview.job_description
    company_name = session.interview.company_name
    resume = session.interview.resume_text
    name = session.interview.user.username
    n = 10
    model = "gpt-4"
    tts_engine = ElevenLabsTTS("21m00Tcm4TlvDq8ikWAM")

    conversation = Session.objects.get(session_id=session.session_id).conversation_set.all()
    # interview_so_far = []

    messages = [
        {"role": "system", "content": f"""You are an interviewer, conducting behavioral interviews to select the most skilled and well-rounded candidates for the role of {role} at {company_name}. You generate interview questions given a job description, the resume of an interviewee to that job, and the interview so far. Read carefully the job description and associated resume, as well as the instructions that follow. However, note that you are the expert of the interview process, so the following should be taken as guidelines, not as strict rules.

        Job Description:
        \"\"\"
        {job_description}
        \"\"\"

        {name}'s Resume:
        \"\"\"
        {resume}
        \"\"\"

        This interview will last ~{n} minutes.

        A few notes to help you find avenues of discussion:
        - You may compare the resume to the job description to see if the interviewee has the necessary skills. If they do not, you may ask them about it, and about what they wish to do to remedy the situation.
        - You should visit these 4 major areas of discussion:
            - experience
            - skills
            - values
            - personality
        
        Conversation so far:
        {" ".join([f"{c.speaker}: {c.text}" for c in conversation])}
        """},
    ]
    # messages += interview_so_far

    text_stream = chat_stream(
        messages=messages,
        model=model,
    )
    voice = InterviewerVoice(tts_engine)
    voice.speak_stream(text_stream)
    return text_stream

def stt_whisper(audio_data: bytes, model_name: str = "base.en") -> str:
    """
    Transcribes the given audio data using OpenAI's Whisper ASR model.

    :param audio_data: Byte array of the audio data.
    :param model_name: Name of the Whisper ASR model to use. Default is "base.en".
    :return: Transcribed text.
    """
    # Load Whisper ASR model
    model = whisper.load_model(model_name)

    # Convert audio bytes to a file-like object
    audio_file = io.BytesIO(audio_data)
    audio_file.name = "audio.wav"

    # Convert audio file to numpy array
    audio = whisper.load_audio(audio_file)

    # Perform transcription
    result = model.transcribe(audio)
    transcription_text = result["text"]

    return transcription_text
    

if __name__ == "__main__":
    import json
    with open("interview.json", "r") as f:
        interview_json = json.load(f)
    
    role = interview_json['role']
    job_description = interview_json['job_description']
    company_name = interview_json['company_name']
    resume = interview_json['resume']
    name = interview_json['name']
    interview_so_far = interview_json['interview_so_far']
    n = interview_json['n']

    tts_engine = ElevenLabsTTS("21m00Tcm4TlvDq8ikWAM")
    interview_reply(
        role=role,
        job_description=job_description,
        company_name=company_name,
        resume=resume,
        name=name,
        interview_so_far=interview_so_far,
        n=n,
        model="gpt-3.5-turbo", # gpt-3.5-turbo, gpt-4
        tts_engine=tts_engine
    )
