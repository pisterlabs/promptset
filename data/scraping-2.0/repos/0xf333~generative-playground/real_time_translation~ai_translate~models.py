from abc import abstractmethod, ABC

import openai
from google.cloud import texttospeech


class BaseGenerativeModel(ABC):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError


class WhisperModel(BaseGenerativeModel):
    def __init__(self, model_name: str, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.model = model_name

    def run(self, file_path: str):
        if self.verbose:
            print(f"Transcribing audio file: {file_path}")
        audio_file = open(file_path, "rb")
        transcript = openai.Audio.transcribe(self.model, audio_file)
        if self.verbose:
            print(f"Transcript output: {transcript}")
        return transcript["text"]


class TranslationModel(BaseGenerativeModel):
    SYSTEM_TEMPLATE = (
            "You are an AI assistant whose main goal is to help people in "
            "translate text from one language to another. You must write "
            "the translation from the user input in {language}. "
            "Note that you MUST provide just the translation, do not add any"
            "other text."
        )

    def run(self, user_input, language):
        if self.verbose:
            print(f"User input: {user_input}")
        system_message = {
            "role": "system",
            "content": self.SYSTEM_TEMPLATE.format(language=language),
        }
        user_message = {
            "role": "user",
            "content": user_input,
        }
        messages = [system_message, user_message]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        if self.verbose:
            print(f"OpenAI response: {response}")
        model_response = response["choices"][0]["message"]["content"]
        return model_response


class TextToVoice(BaseGenerativeModel):
    LANGUAGE_CODES = {
        "english": "en-US",
        "spanish": "es-ES",
        "french": "fr-FR",
        "german": "de-DE",
        "italian": "it-IT",
    }

    def run(self, input_text: str, language: str):
        # Instantiates a client
        client = texttospeech.TextToSpeechClient()

        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=input_text)

        # Build the voice request, select the language code ("en-US") and the ssml
        # voice gender ("neutral")
        voice = texttospeech.VoiceSelectionParams(
            language_code=self.LANGUAGE_CODES[language],
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )

        # Select the type of audio file you want returned
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        # Perform the text-to-speech request on the text input with the selected
        # voice parameters and audio file type
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # The response's audio_content is binary.
        output_file = "output.mp3"
        with open(output_file, "wb") as out:
            # Write the response to the output file.
            out.write(response.audio_content)
            print(f'Audio content written to file "{output_file}"')
        return output_file
