class AudioRecognizer():
    def __init__(self, method):
        self.method = method

    def recognize(self):
        return self.method.recognize()


class Method:
    def recognize(self):
        raise NotImplementedError()


class Whisper(Method):
    def __init__(self, loaded_model):
        # Parameter on whisper
        import whisper
        self.model = whisper.load_model(loaded_model)
        print('Finished loading model')

    def recognize(self):
        return self.model.transcribe("temp.wav", language="ja")


class WhisperAPI(Method):
    def __init__(self):
        # Parameter on whisper
        import openai
        import os
        self.openai = openai
        self.openai.api_key = os.environ["OPENAI_API_KEY"]

    def recognize(self):
        file = open("temp.wav", "rb")
        return self.openai.Audio.transcribe("whisper-1", file)
