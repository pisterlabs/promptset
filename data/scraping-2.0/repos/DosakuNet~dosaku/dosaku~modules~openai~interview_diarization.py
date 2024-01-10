"""OpenAI InterviewDiarization module."""
from math import ceil
import os

from openai import OpenAI
from pydub import AudioSegment

from dosaku import Service
from dosaku.modules import GPT
from dosaku.utils import ifnone


class OpenAIInterviewDiarization(Service):
    """OpenAI interview diarization class.

    Example::

        from dosaku.modules import OpenAIInterviewDiarization

        transcriber = OpenAIInterviewDiarization()
        audio_file = 'tests/resources/fridman_susskind.mp3'
        text = transcriber.transcribe_interview(audio_file, interviewer='Lex Fridman', interviewee='Leonard Susskind')
        print(text)
    """

    whisper_instructions = (
        'INTERVIEWER: So I was just thinking about the Roman Empire, as one does.\n'
        '\n'
        'INTERVIEWEE: Is that whole meme where all guys are thinking about the Roman Empire at least once a day?\n'
    )

    gpt_instructions = (
        'You are an expert editor tasked with editing transcribed text from interviews. You will be given raw '
        'interview text between an interviewer and an interviewee. The raw text will not have '
        'any punctuation or speaker labels. Your task is to add speaker labels, proper punctuation, and remove any '
        'extraneous "ums" or filler words. If you are given the names of the interviewer and interviewee use them in '
        'your transcription; if the names are not given, use "INTERVIEWER" and "INTERVIEWEE".\n'
        '\n'
        'For example, given the text:\n'
        '\n'
        'Lex Fridman interviewing Elon Musk:\n'
        '\n'
        'so um I was just thinking about the roman empirer as one does is that whole meme where all guys are thinking '
        'about the roman empire at least once a day and half the population is confused whether it’s true or not but '
        'more seriously thinking about the wars going on in the world today and um as you know, sometimes, war and '
        'military conquest has been a big parte of roman society and culture, and I think has been a big part of most '
        'empires and dynasties throughout human history yeah they usually came as a result of conquest I mean like '
        'there’s some like the hapsburg empire where there was just a lot of clever marriages\n'
        '\n'
        'You should respond with the following text:\n'
        '\n'
        'LEX FRIDMAN: So I was just thinking about the Roman Empire, as one does.\n'
        '\n'
        'ELON MUSK: Is that whole meme where all guys are thinking about the Roman Empire at least once a day?\n'
        '\n'
        'LEX FRIDMAN: And half the population is confused whether it’s true or not. But more seriously, thinking about '
        'the wars going on in the world today, and as you know, war and military conquest has been a big part of Roman '
        'society and culture, and I think has been a big part of most empires and dynasties throughout human history.\n'
        '\n'
        'ELON MUSK: Yeah, they usually came as a result of conquest. I mean, there’s some like the Hapsburg Empire '
        'where there was just a lot of clever marriages.\n'
    )

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.client = OpenAI(api_key=self.config['API_KEYS']['OPENAI'])

    def save_chunks(self, raw_audio: AudioSegment, chunk_length: int, overwrite: bool = False):
        temp_dir = self.config['DIR_PATHS']['TEMP']
        num_chunks = ceil(len(raw_audio) / chunk_length)
        filenames = []
        for idx in range(num_chunks):
            filename = os.path.join(temp_dir, f'audio_{idx}.mp3')
            if overwrite is True or not os.path.exists(filename):
                start = chunk_length * idx
                end = min(chunk_length * (idx + 1), len(raw_audio) - 1)
                audio_chunk = raw_audio[start:end]
                audio_chunk.export(filename, format='mp3')
                self.logger.debug(f'Exported audio chunk to {filename}')
            filenames.append(filename)

        return filenames

    def transcribe_interview(
            self,
            audio_file: str,
            chunk_length: int = 1200,
            chunk_separater: str = '\n\n***GPT CHUNK BREAK***\n\n',
            overwrite_files: bool = True,
            interviewer: str = 'Interviewer',
            interviewee: str = 'Interviewee'
    ):
        """Transcribe audio file with diarization (speak identification).

        If the input file is too long it will be broken into separate chunks for processing.

        Args:
            audio_file: Path to the given mp3 file.
            chunk_length: Maximum length of each audio chunk for processing, in seconds.
            chunk_separater: Separater text placed between transcribed chunks.
            overwrite_files: Whether to overwrite temp files found with the same name.
            interviewer: The name of the interviewer.
            interviewee: The name of the interviewee.
        """
        chunk_length = chunk_length * 1000  # pydub measures time in ms
        raw_audio = AudioSegment.from_mp3(audio_file)
        files = self.save_chunks(raw_audio, chunk_length, overwrite=overwrite_files)

        final_transcript = ''
        for idx, audio_filename in enumerate(files):
            text_filename = os.path.join(self.config['DIR_PATHS']['TEMP'], f'transcription_{idx}.txt')
            gpt_text_filename = os.path.join(self.config['DIR_PATHS']['TEMP'], f'gpt_transcription_{idx}.txt')

            if overwrite_files or not os.path.exists(text_filename):
                with (
                    open(audio_filename, 'rb') as audio_file,
                    open(text_filename, 'w') as text_file
                ):
                    transcript = self.client.audio.transcriptions.create(
                        model='whisper-1',
                        file=audio_file,
                        prompt=self.whisper_instructions,
                    )
                    text_file.write(transcript.text)
                    self.logger.debug(f'Transcribed audio chunk {audio_filename} to {text_filename}')

            if overwrite_files is True or not os.path.exists(gpt_text_filename):
                with (
                    open(text_filename, 'r') as text_file,
                    open(gpt_text_filename, 'w') as gpt_file
                ):
                    interviewer = ifnone(interviewer, default='Interviewer')
                    interviewee = ifnone(interviewee, default='Interviewee')
                    text = f'{interviewer} interviewing {interviewee}:\n\n'
                    text += text_file.read()
                    gpt = GPT(instructions=self.gpt_instructions)
                    gpt_text = gpt.message(text)
                    gpt_file.write(gpt_text.message)
                    self.logger.debug(f'Corrected {text_filename} with GPT and saved the result to {gpt_text_filename}')

            with open(gpt_text_filename, 'r') as gpt_file:
                if len(final_transcript) > 0:
                    final_transcript += chunk_separater
                final_transcript += gpt_file.read()

        return final_transcript


OpenAIInterviewDiarization.register_action('save_chunks')
OpenAIInterviewDiarization.register_action('audio_to_text')
