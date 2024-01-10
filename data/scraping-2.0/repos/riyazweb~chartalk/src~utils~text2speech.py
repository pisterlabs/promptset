import os
import tempfile
from TTS.api import TTS
import openai
openai.api_key = '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'


class TTSTalker():
    def __init__(self) -> None:
        model_name = TTS.list_models()[0]
        self.tts = TTS(model_name)

    def test(self, text, language='en'):

        tempf  = tempfile.NamedTemporaryFile(
                delete = False,
                suffix = ('.'+'wav'),
            )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"imagine you are karl max and reply to me for this question: {text} in 15 words"}
            ]
        )
        OP = response['choices'][0]['message']['content']
        OP = OP.replace('"', '')
    
        self.tts.tts_to_file(OP, speaker=self.tts.speakers[3], language=language, file_path=tempf.name)

        return tempf.name
