import os
import pandas as pd
import io
from pydub import AudioSegment

from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from evadb.utils.generic_utils import try_to_import_openai

_VALID_TTS_MODEL = [
    "tts-1",
    "tts-1-hd"
]

_VALID_VOICE = [
    "alloy",
    "echo",
    "fable",
    "onyx",
    "nova",
    "shimmer"
]

_VALID_RESPONSE_FORMAT = [
    "mp3",
    "opus",
    "aac",
    "flac"
]

_MIN_SPEED = 0.25
_MAX_SPEED = 4.0

class TextToSpeech(AbstractFunction):
    @property
    def name(self) -> str:
        return "TextToSpeech"
    
    @setup(cacheable=True, function_type="text-to-speech", batchable=True)
    def setup(
        self, 
        model="tts-1",
        voice="alloy",
        response_format="mp3",
        speed: float = 1,
        openai_api_key="",
    ) -> None:
        assert model in _VALID_TTS_MODEL, f"Unsupported TTS Model {model}"
        assert voice in _VALID_VOICE, f"Unsupported Voice {voice}"
        assert response_format in _VALID_RESPONSE_FORMAT, f"Unsupported Response Format {response_format}"
        assert speed >= _MIN_SPEED and speed <= _MAX_SPEED, f"Input speed must be between {_MIN_SPEED} and {_MAX_SPEED}"

        self.model = model
        self.voice = voice
        self.response_format = response_format
        self.speed = speed
        self.openai_api_key = openai_api_key

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["prompt"],
                column_types=[
                    NdArrayType.STR,
                ],
                column_shapes=[(None,)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["response"],
                column_types=[NdArrayType.FLOAT32],
                column_shapes=[(None, None, 3)],
            )
        ],
    )

    def forward(self, text_df):
        try_to_import_openai()
        from openai import OpenAI

        api_key = self.openai_api_key
        if len(self.openai_api_key) == 0:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        assert (
            len(api_key) != 0
        ), "Please set your OpenAI API key using SET OPENAI-KEY = 'sk-' or environment variable (OPENAI-KEY)"

        client = OpenAI(api_key=api_key)

        def tts(text_df: PandasDataframe):
            results = []
            queries = text_df[text_df.columns[0]]
            for query in queries:
                response = client.audio.speech.create(
                    model=self.model,
                    input = query,
                    response_format = self.response_format,
                    voice = self.voice,
                    speed = self.speed
                )
                audio_data = AudioSegment.from_file(io.BytesIO(response.content))
                audio_array = audio_data.get_array_of_samples()
                results.append(audio_array)
            return results

        df = pd.DataFrame({"response": tts(text_df=text_df)})
        return df