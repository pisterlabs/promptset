import os
import uuid

import auditok
import openai
import soundfile as sf

from .openai_decorator import retry_on_openai_errors
from .utils import get_project_root

# Set the OpenAI API key from environment variable or use a default key
openai.api_key = os.environ.get(
    "OPENAI_API_KEY", ""
)


class Whisper:
    """
    This class serves as a wrapper for the OpenAI Whisper API to facilitate chatbot responses.
    """

    def __init__(self, model_name: str = "whisper-1", whisper_sample_rate: int = 16000):
        """
        Initialize the Whisper chatbot instance.

        :param model_name: The name of the OpenAI Whisper model to use.
        :param whisper_sample_rate: The sample rate for audio processing.
        """
        self.model_name = model_name
        self.whisper_sample_rate = whisper_sample_rate

    def vad_audiotok(self, audio_content):
        """
        Perform voice activity detection using the audiotok package.

        :param audio_content: Bytes of audio data.
        :return: Chunks containing speech detected in the audio.
        """
        audio_regions = auditok.split(
            audio_content,
            sr=self.whisper_sample_rate,
            ch=1,
            sw=2,
            min_dur=0.5,
            max_dur=30,
            max_silence=0.3,
            energy_threshold=30
        )
        return audio_regions

    def audio_process(self, wav_path, is_byte=False):
        """
        Process audio data, performing voice activity detection and segmenting the audio.

        :param wav_path: Path to the audio file or audio bytes.
        :param is_byte: Boolean flag indicating if the input is audio bytes.
        :return: Segmented audio chunks containing detected speech.
        """
        if not is_byte:
            with open(wav_path, 'rb') as f:
                wav_bytes = f.read()
        else:
            wav_bytes = wav_path
        wav, sr = sf.read(wav_path)
        audio_regions = self.vad_audiotok(wav_bytes)
        wav_segments = []
        for r in audio_regions:
            start = r.meta.start
            end = r.meta.end
            segment = wav[int(start * sr):int(end * sr)]
            wav_segments.append(segment)
        return wav_segments

    @retry_on_openai_errors(max_retry=7)
    def transcribe(self, audio_file):
        """
        Transcribe the provided audio using the OpenAI API.

        :param audio_file: Path to the audio file or audio bytes.
        :return: Transcription text from the audio.
        """
        # Save audio bytes as a temporary WAV file
        root_path = get_project_root()
        temp_wav_path = f"{root_path}/resources/audios/{str(uuid.uuid4())}.mp3"
        with sf.SoundFile(temp_wav_path, 'wb', samplerate=self.whisper_sample_rate, channels=1) as f:
            f.write(audio_file)

        auf = open(temp_wav_path, 'rb')
        # Transcribe using OpenAI API
        response = openai.Audio.transcribe(
            self.model_name, auf)
        # Clean up temporary file
        os.remove(temp_wav_path)
        return response['text']

    @retry_on_openai_errors(max_retry=7)
    def transcribe_raw(self, audio_file):
        """
        Transcribe the provided audio using the OpenAI API without saving a temporary file.

        :param audio_file: Path to the audio file or audio bytes.
        :return: Transcription text from the audio.
        """
        auf = open(audio_file, 'rb')
        # Transcribe using OpenAI API
        response = openai.Audio.transcribe(
            self.model_name, auf)
        return response['text']


if __name__ == "__main__":
    # Example usage
    wh = Whisper()
    with open("./audios/0_edited.wav", "rb") as f:
        audio_content = f.read()
    print(type(audio_content))
    segments = wh.audio_process("./audios/0_edited.wav")
