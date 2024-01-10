import openai

# gotta add more later, limit = 244 tokens
# https://platform.openai.com/tokenizer
ABBREVIATIONS_UK = "НП,ТТН,ЕН,ПІБ,грн,ТОВ,ФОП,ПП,ПАТ,КП,ДП,НДФЛ,ЧП,ГП,СПД, р-н, обл"

async def convert_audio_to_text(local_input_file_path: str):
    transcription = openai.Audio.transcribe("whisper-1", open(local_input_file_path, 'rb'), language="uk",
                                            prompt=ABBREVIATIONS_UK)
    return transcription["text"]
