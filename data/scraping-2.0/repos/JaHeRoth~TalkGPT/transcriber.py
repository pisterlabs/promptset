import openai
from transformers import pipeline

from chat_bot import ChatBot, Responder


class ASR:
    def __call__(self, fpath, prompt: str):
        raise NotImplementedError


class OpenAIASR(ASR):
    def __init__(self, model_name="whisper-1"):
        self.model_name = model_name

    def __call__(self, fpath, prompt: str):
        with open(fpath, "rb") as file:
            return openai.Audio.transcribe(self.model_name, file, prompt=prompt, language="en")["text"]


class HuggingFaceASR(ASR):
    def __init__(self, model_name="openai/whisper-large-v2"):
        self.model_name = model_name
        self.asr = pipeline("automatic-speech-recognition", model=self.model_name)

    def __call__(self, fpath, prompt: str):
        return self.asr(str(fpath))["text"]


class Transcriber:
    def __init__(self, asr: ASR, responder: Responder | None):
        self.asr = asr
        self.responder = responder

    def transcribe(self, fpath, messages: list[dict]):
        return self.post_process(self.initial_transcribe(fpath, messages), messages)

    def initial_transcribe(self, fpath, messages: list[dict]):
        # This prompt is fed to Whisper. Whisper will then try to match its style. Whisper ignores all but the
        # last 244 tokens, which is why the most important prompts are at the end.
        input_prompt = "\n".join(
            [message["content"] for message in messages[1:-1]]
            # All but the system message and the last assistant message
        ) + (
                messages[0]["content"] + "\n" +  # The system message
                messages[-1]["content"] + "\n" +  # The assistant's last message
                "Hello, welcome to my lecture.\n" +  # To include punctuation
                "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking."  # To include filler words
        )
        return self.asr(fpath, input_prompt)

    def post_process(self, raw_transcription, messages: list[dict]):
        if self.responder is None:
            return raw_transcription
        post_processing_system = "You are a helpful assistant. Your task is to correct any spelling discrepancies in " \
                                 "the transcribed text. Only add necessary punctuation such as periods, commas, and " \
                                 "capitalization, and use only the context provided."
        chat_bot = ChatBot(self.responder, system_message=post_processing_system)
        chat_bot.messages = chat_bot.messages + messages[1:]
        return chat_bot.respond(raw_transcription)
