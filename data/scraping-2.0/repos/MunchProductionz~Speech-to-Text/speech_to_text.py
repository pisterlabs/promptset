import whisper
import openai
from languages import languages
from api_keys import keys


def speech_to_text_whisper(audio_filename, filetype="mp3", model="base"):
    
    # Available models:
    # - Tiny    (fastest, but least accurate)
    # - Base    (default)
    # - Small
    # - Medium  (good balance)
    # - Large   (best accuracy, but slow)

    # Set the path to the audio file
    audio_filepath = "audio/" + audio_filename + "." + filetype
    text_filepath = "text/" + audio_filename + "1.txt"

    # Load the model and transcribe the audio file
    model = whisper.load_model(model)
    result = model.transcribe(audio_filepath, fp16=False)

    # print the recognized text
    print(result['text'])
    
    # Write text to a file
    with open(text_filepath, 'w') as text_file:
        text_file.write(result['text'])
    
    return print('Transcription saved as ' + audio_filename + '.txt')


def speech_to_text_openai(audio_filename, filetype="mp3", model="whisper-1", response_format="text", language="English"):
    
    # Set the OpenAI API key
    openai.api_key = keys["OPENAI_API_KEY"]
    
    # Set the path to the audio file and get the language code
    audio_filepath = "audio/" + audio_filename + "." + filetype
    language = languages[language.lower()]
    print(audio_filepath)
    print(language)
    
    # Transcribe the audio file using the OpenAI API
    with open(audio_filepath, "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            file = audio_file,
            model = model,
            response_format=response_format,
            language=language
        )
    print(transcript)
    
    
# Run the function
speech_to_text_whisper("test", filetype="m4a", model="base")
