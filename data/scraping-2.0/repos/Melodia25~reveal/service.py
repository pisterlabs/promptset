import os

from fastapi import UploadFile
from openai import  OpenAI
import tempfile
import shutil
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.getenv("KEY_OPENAI"),
)



async def transcribe(audio: UploadFile):
    if audio.content_type != 'audio/mpeg':
        return {"error": "El archivo debe ser un MP3."}

        # Validar el tamaño del archivo (25 MB = 25 * 1024 * 1024 bytes)
    if audio.size > 25 * 1024 * 1024:
        return {"error": "El archivo no debe superar los 25 MB."}

    if audio.content_type != 'audio/mpeg':
        return {"error": "El archivo debe ser un MP3."}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_file_name = temp_file.name
        await audio.seek(0)  # Rewind to the beginning of the file
        shutil.copyfileobj(audio.file, temp_file)


    # Now pass this byte stream to the OpenAI API
    with open(temp_file_name, "rb") as file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=file,
            response_format="verbose_json"
        )
    os.remove(temp_file_name)
    return transcript