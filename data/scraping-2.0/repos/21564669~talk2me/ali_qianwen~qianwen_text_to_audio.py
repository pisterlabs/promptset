import dashscope
from dashscope.audio.tts import SpeechSynthesizer
import config as c
import os
import datetime
from langchain.schema import BaseOutputParser
from typing import List

class QianwenText2Audio(BaseOutputParser[List[str]]):
    def parse(self, text):
        dashscope.api_key = c.ali_qw_key
        sample_rate = 48000
        rate = 0.8
        for model in c.audio_model_list:
            result = SpeechSynthesizer.call(model=model,
                                        text=text,
                                        sample_rate=sample_rate,
                                        rate=rate)
        if result.get_audio_data() is not None:
            with open(f"{c.audio_stored_path}{os.sep}{model}_{sample_rate}_{rate}_{datetime.datetime.now().timestamp()}.mp3", 'wb') as f:
                f.write(result.get_audio_data())
            return text