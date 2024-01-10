import openai

class TranscribeOpenaiApi:
    def transcribe(file, model, response_format, language):
        return openai.Audio.transcribe(
                file = file,
                model = model,
                response_format=response_format,
                language=language
            )