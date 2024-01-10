import openai
import io
from pydub import AudioSegment,silence
import os 
import config
openai.api_key = config.openai_api_key

# Define the function to transcribe an audio file
def transcribe_audio(audio_file_path):
    with io.open(audio_file_path, 'rb') as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file, language="uk")
        return transcript

# Load the input audio file
audio = AudioSegment.from_file("vocals.wav")

# Define the minimum segment duration in milliseconds
min_segment_duration = 1000  # 1 second

# Find non-silent chunks in the audio file
non_silent_chunks = silence.split_on_silence(audio, 
                                             min_silence_len=500, 
                                             silence_thresh=audio.dBFS-14, 
                                             keep_silence=250)

# Initialize the start time for the first segment
start_time = 0

# Process each non-silent chunk
for i, chunk in enumerate(non_silent_chunks):
    # Only process if the chunk duration is longer than the minimum segment duration
    if len(chunk) >= min_segment_duration:
        # Calculate the end time for the chunk
        end_time = start_time + len(chunk)

        # Export the chunk to a WAV file
        segment_file = f"segment_{i}.wav"
        chunk.export(segment_file, format="wav")

        # Transcribe the chunk
        try:
            transcript = transcribe_audio(segment_file)

            # Print the time-coded subtitle
            print(f"{i+1}")
            print(f"{start_time//3600000:02d}:{(start_time//60000)%60:02d}:{(start_time//1000)%60:02d},{start_time%1000:03d} --> "
                  f"{end_time//3600000:02d}:{(end_time//60000)%60:02d}:{(end_time//1000)%60:02d},{end_time%1000:03d}")
            print(transcript.text)
            print()

        except openai.error.InvalidRequestError as e:
            print(f"Error in segment {i+1}: {str(e)}")

        # Update the start time for the next chunk
        start_time = end_time

    # Clean up the segment file
    os.remove(segment_file)
