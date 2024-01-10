import io
import time
import tempfile

from modal import method, Image, Secret

from .common import stub


MODEL_NAMES = ["base.en"]#, "large-v2"]

transcriber_image = (
    Image.debian_slim(python_version="3.10.8")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "https://github.com/openai/whisper/archive/v20230314.tar.gz",
        "ffmpeg-python",
        "openai",
    )
)


@stub.cls(
    container_idle_timeout=180,
    image=transcriber_image,
    secret=Secret.from_name("my-openai-secret"),
)
class Whisper:
    def __enter__(self):
        import torch
        import whisper

        self.use_gpu = torch.cuda.is_available()
        device = "cuda" if self.use_gpu else "cpu"
        self.models = {
            model_name: whisper.load_model(model_name, device=device) for model_name in MODEL_NAMES
        }

    @method()
    def transcribe_segment(
        self,
        audio_data: bytes,
        model_name: str = None,
        format: str = ".wav",
    ):
        try:
            import openai
            from whisper import load_audio

            t0 = time.time()

            # Create a temporary .wav file from audio bytes
            with tempfile.NamedTemporaryFile(delete=False, suffix=format) as temp_audio_file:
                temp_audio_file.write(audio_data)
                temp_audio_file_path = temp_audio_file.name

            # Convert .wav file to numpy array
            np_array = load_audio(temp_audio_file_path)

            if model_name in self.models:
                result = self.models[model_name].transcribe(np_array, language="en")  # type: ignore
                print(result)
                transcription_text = result["text"]
            elif model_name:
                raise ValueError(f"Unknown model name: {model_name}")
            else:
                # Convert audio bytes to a file-like object
                audio_file = io.BytesIO(audio_data)
                audio_file.name = "audio.wav"

                # Pass the file-like object to the OpenAI API
                result = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio_file,
                    language="en",
                )
                transcription_text = result["text"]


            print(f"Transcribed the following text in {time.time() - t0:.2f}s: {transcription_text}")

            return transcription_text
        except Exception as e:
            print(f"Error during transcription with Whisper: {str(e)}")
            raise
