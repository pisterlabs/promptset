import os
import pandas as pd
import ast, array
from pydub import AudioSegment
import tempfile
import pickle

from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from evadb.utils.generic_utils import try_to_import_openai

_VALID_STT_MODEL = [
    "whisper-1"
]


_VALID_RESPONSE_FORMAT = [
    "json",
    "text",
    "srt",
    "verbose_json",
    "vtt"
]

class SpeechToText(AbstractFunction):
    @property
    def name(self) -> str:
        return "SpeechToText"
    
    @setup(cacheable=True, function_type="speech-to-text", batchable=True)
    def setup(
        self, 
        model="whisper-1",
        response_format="verbose_json",
        temperature: float=0,
        openai_api_key="",
    ) -> None:
        assert model in _VALID_STT_MODEL, f"Unsupported STT Model {model}"
        assert response_format in _VALID_RESPONSE_FORMAT, f"Unsupported Response Format {response_format}"
        self.model = model
        self.response_format = response_format
        self.temperature = temperature
        self.openai_api_key = openai_api_key

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["speech"],
                column_types=[NdArrayType.FLOAT32],
                column_shapes=[(None, None, 3)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["response"],
                column_types=[
                    NdArrayType.STR,
                ],
                column_shapes=[(None,)],
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

        def stt(text_df: PandasDataframe):
            results = []
            audios = text_df[text_df.columns[0]]
            for audio in audios:
                audio_array = array.array('h', ast.literal_eval(audio))
                audio_data = AudioSegment(audio_array.tobytes(), frame_rate=88200, sample_width=2, channels=1)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file_path = temp_file.name
                    audio_data.export(temp_file_path, format="wav")
                    audio_file = open(temp_file_path, "rb")
                    transcript = client.audio.transcriptions.create(
                        model=self.model, 
                        temperature=self.temperature,
                        response_format=self.response_format,
                        file=audio_file
                    )
                    os.remove(temp_file_path)
                    results.append(transcript)
            return results

        df = pd.DataFrame({"response": stt(text_df=text_df)})
        return df