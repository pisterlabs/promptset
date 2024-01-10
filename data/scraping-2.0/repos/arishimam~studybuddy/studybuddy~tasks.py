from celery import shared_task
from .models import AudioFile, Note
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


client = OpenAI()


@shared_task
def generate_notes_from_audio(audio_file_id):
    # Loads audio file
    audio_file = AudioFile.objects.get(id=audio_file_id)

    file_path = audio_file.file.path

    transcript = ""
    with open(file_path, 'rb') as f:
        # Calls Whisper API to transcribe audio file
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=f, response_format="text")

    # Splits transcripts into manageable chunks for gpt api
    def split_transcript(transcript, max_length=4000):
        chunks = []
        current_chunk = ""
        sentence_endings = {'.', '!', '?'}

        for char in transcript:
            current_chunk += char
            if char in sentence_endings and len(current_chunk) >= max_length:
                # Trim leading and trailing whitespaces
                chunks.append(current_chunk.strip())
                current_chunk = ""

        # Add any remaining text as a final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    # Calls split_transcript function and stores broken up
    # text into chunks array
    chunks = split_transcript(transcript)
    notes = ""

    # Loop through chunks and promp gpt api to convert into notes
    for i in chunks:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates notes from a transcription of a lecture in a structured format. Please provide the notes as a bulleted list, using '-' for each new note. Start each note on a new line. I don't want everything converted into a note. Be concise and create the least number of notes possible, while still capturing the general idea and important lecture details."},
                {"role": "user", "content": i}
            ]
        )
        notes += response.choices[0].message.content
        notes += "\n"

    # parsed_notes = notes.strip().split('\n')

    Note.objects.create(audio_file=audio_file, user=audio_file.user,
                        content=notes, title=audio_file.file.name)
