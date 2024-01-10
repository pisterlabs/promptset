import os
import magic
from datetime import datetime
import ffmpeg
import openai
from pydub import AudioSegment

def get_file_type(file_path):
    file_type = magic.from_file(file_path, mime=True)
    return file_type.split('/')[0]

def identify_files(folder_path):
    video_files = []
    audio_files = []

    for root, _, files in os.walk(folder_path):
        if not files:
            os.rmdir(root)  # Delete empty folders
            continue

        for file in files:
            file_path = os.path.join(root, file)
            file_type = get_file_type(file_path)

            if file_type == 'video':
                video_files.append(file_path)
            elif file_type == 'audio':
                audio_files.append(file_path)

    return video_files, audio_files

def main():
    folder_path = '/Users/aaronnganm1/Documents/Zoom'
    video_files, audio_files = identify_files(folder_path)

    if not audio_files:
        for video_file in video_files:
            audio_file = ffmpeg.input(video_file).output('pipe:', format='s16le', acodec='pcm_s16le', ac=1, ar='16k').run_async(pipe_stdout=True)
            audio_file_length = audio_file.stderr.read().decode().split('Duration: ')[1].split(',')[0].split(':')
            audio_file_length_seconds = int(audio_file_length[0]) * 3600 + int(audio_file_length[1]) * 60 + int(audio_file_length[2])

            # Split the audio file into chunks of 25 MB or less
            chunk_duration = 25 * 1024 * 1024 / (16000 * 2 * 2)
            audio_chunks = []
            for i in range(0, audio_file_length_seconds, chunk_duration):
                chunk = AudioSegment.from_file(audio_file.stdout, format="raw", frame_rate=16000, channels=1, sample_width=2).set_frame_rate(16000).set_channels(1).set_sample_width(2)
                audio_chunks.append(chunk)

            # Merge the audio chunks and save the final audio file
            merged_audio = AudioSegment.empty()
            for chunk in audio_chunks:
                merged_audio += chunk

            merged_audio.export(os.path.splitext(video_file)[0] + '.mp3', format='mp3')

if __name__ == '__main__':
    main()