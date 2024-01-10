from pathlib import Path


import openai
from pytube import YouTube


def transcript_audio_to_text(file_paths: list[Path]) -> str:
    transcriptions: list[dict] = []
    
    for audio_path in file_paths:
        audio_file = open(audio_path, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)

        transcriptions.append(transcript["text"])

        audio_file.close()
        
    return transcriptions
    

def write_transcription_in_file(transcriptions: list[str], 
                                file_names: list[str], 
                                path_save: str | Path) -> None:
    print(f"{path_save}/{file_names[0]}")
    for i, transcription in enumerate(transcriptions):
        try:
          with open(path_save/file_names[i], "w") as file:
              file.write(transcription)  
        except FileNotFoundError:
            print("File not found")
        except PermissionError:
            print("Permission denied")
    
