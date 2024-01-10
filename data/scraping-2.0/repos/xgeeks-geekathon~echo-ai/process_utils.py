
from parsed_file import ParsedFile
import file_utils
import summary
import os
import openai
import gradio as gr

def transcribe(audio_file, response_format="json", language="en"):
    with open(audio_file.name, 'rb') as audio_file:
        return openai.Audio.transcribe(
            file=audio_file, 
            model="whisper-1",
            response_format=response_format, 
            language=language)

def translate(audio_file, response_format="json"):
    with open(audio_file.name, 'rb') as audio_file:
        return openai.Audio.translate(
            file=audio_file, 
            model="whisper-1",
            response_format=response_format)

def process_file(file, is_translation):
    parsedFile = ParsedFile(file.name)
    resultPath = parsedFile.getTranslatePath() if is_translation else parsedFile.getTranscribePath()
    resultPathSrt = parsedFile.getTranslateSrtPath() if is_translation else parsedFile.getTranscribeSrtPath()

    filePath = file.name
    
    if parsedFile.extension == ".mp4":
        file = file_utils.transform_video(file)
    
    if not os.path.exists(resultPath):
        transcript = translate(file) if is_translation else transcribe(file)
        file_utils.save_to_file(transcript, resultPath)
    
    if not os.path.exists(resultPathSrt):
        transcript_srt = translate(audio_file=file, response_format="srt") if is_translation else transcribe(audio_file=file, response_format="srt")
        file_utils.save_to_file(transcript_srt, resultPathSrt)

    data = file_utils.load_file(resultPath)
    resume = summary.resumeTranscript(data)

    print("All done!")

    if is_translation:
        return (data[0].page_content, (filePath, resultPathSrt), gr.Row.update(visible=True))
    else:
        return (resume, data[0].page_content, resultPath, resultPathSrt, (filePath, resultPathSrt), gr.Row.update(visible=True))
