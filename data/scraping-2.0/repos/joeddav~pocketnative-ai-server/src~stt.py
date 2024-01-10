from openai import OpenAI, APIError
from werkzeug.datastructures import FileStorage
import os
import tempfile

DEFAULT_PROMPT = """\
This is a transcript of an English speaker trying to learn Russian. It may have \
ошипки, but only in the русский part. The user may switch between English and \
Russian."""

DEFAULT_USER_EXAMPLES = [
    "Привет! Как ты сегодня делаешь? Я хочу идти в кино сегодня вечером. Ты свободен?",
    "What does 'занимание' mean?",
    (
        "I'm trying to figure out how to say uhm 'the more you fight it, the ",
        "worse it'll feel.' What's the right... the right phrase I'm looking for? Like ",
        "uhm 'Если ты...Если ты будешь больше сопротивляться, то будет больнее' but I ",
        "know there's a better way of saying that.",
    ),
    "Привет! Как ты сегодня делаешь? Я хочу идти в кино сегодня вечером. Ты свободен?"
]



def transcribe(file: FileStorage, temperature=0.2, prompt=None, user_examples=None) -> str:
    if prompt is None:
        prompt = DEFAULT_PROMPT
    if user_examples is None:
        user_examples = DEFAULT_USER_EXAMPLES

    full_prompt = prompt + "\n\n" + "\n".join([f"Example: {ex}" for ex in user_examples])

    client = OpenAI()

    # Save the file temporarily
    _, temp_filepath = tempfile.mkstemp()
    temp_filepath += "-" + file.filename
    file.save(temp_filepath)

    try:
        with open(temp_filepath, 'rb') as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=f,
                temperature=temperature,
                prompt=full_prompt, 
            )
    finally:
        # Remove the temporary file
        os.remove(temp_filepath)

    return transcript.text
