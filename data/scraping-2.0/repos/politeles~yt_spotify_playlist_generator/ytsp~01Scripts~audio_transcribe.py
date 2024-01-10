import os
from pydub import AudioSegment
import openai
from dotenv import load_dotenv,find_dotenv
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import file_json as fj


def generate_corrected_transcript(temperature, system_prompt, transcript):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": transcript
            }
        ]
    )
    return response['choices'][0]['message']['content']

# segment audio file unsing pydub in 10 minute chunks
def segment_audio_file(audio_file):
    # split audio file into 10 minute chunks
    audio = AudioSegment.from_mp3(audio_file)
    # PyDub handles time in milliseconds
    ten_minutes = 10 * 60 * 1000
    audio_chunks = []
    # split audio file into 10 minute chunks
    for i in range(0, len(audio), ten_minutes):
        audio_chunks.append(audio[i:i+ten_minutes])
    # save the chunks
    for i, chunk in enumerate(audio_chunks):
        out_file = "audio_chunks/chunk{0}.wav".format(i)
        print("exporting", out_file)
        chunk.export(out_file, format="wav")
    return audio_chunks


def main():
    parser = argparse.ArgumentParser(description='Transcribe audio files')
    parser.add_argument('file', help='audio file to transcribe')
    # parser.add_argument('save_dir', help='directory to save the audio file')
    # parser.add_argument('transcript_dir', help='directory to save the transcription')
    args = parser.parse_args()

    file_name = args.file
    # save_dir = args.save_dir
    # transcript_dir = args.transcript_dir
    
    # check if there are chunk files in the folder
    # if not os.path.exists("audio_chunks"):
    #     os.makedirs("audio_chunks")
    # if not os.path.exists("transcript"):

    # if save_dir == None:
    #     save_dir = os.getcwd()
    print("loading environment variables...")
    _ = load_dotenv(find_dotenv())
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
# use the whisper api to transcribe the audio file
    audio_file= open("audio01.mp3", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file,responseFormat="text")
    # print the transcript

    system_prompt = "Eres un estudiante de la asignatura de lengua española. El texto está en español pero pronunciado por un andalúz. Tu tarea es corregir cualquier error de transcripción en el texto que se proporciona. Añade sólo los signos de puntuación, comas y utiliza sólo el contexto proporcionado."
   
    
    # corrected_text = generate_corrected_transcript(0, system_prompt, transcript)
    # print(corrected_text)

    # save transcript to a file in text format
    # with open("transcript"+"/"+"audio1"+".txt", "w") as f:
    #      f.write(corrected_text)
    # f.close()




    

if __name__ == "__main__":
    main()