from .base import ModelRoute
import whisper
from gpu_box.types import JSONType, File


class WhisperBase(ModelRoute):
    """Whisper from OpenAI (base)"""

    name = "whisper-base"
    model_size = "base"

    def load(self):
        return whisper.load_model(self.model_size)

    async def run(self, file: File) -> JSONType:
        print(f"running model on {file.name} (@{file.path})")
        return self.model.transcribe(file.path)


class WhisperLarge(WhisperBase):
    """Whisper from OpenAI (large)"""

    name = "whisper-large"
    model_size = "large"
