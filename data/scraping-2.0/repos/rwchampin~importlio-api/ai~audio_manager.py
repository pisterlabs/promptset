import os
import tempfile
import io
from pydub import AudioSegment
from pydub.silence import split_on_silence
from openai import OpenAI

class AudioManager:
    def __init__(self):
        self.client = OpenAI()
        self.MAX_FILE_SIZE_MB = 25

    def get_file_size_mb(self, audio_segment):
        # Calculate file size in megabytes
        return os.path.getsize(io.BytesIO(audio_segment.export(format="wav").read())) / (1024 ** 2)

    def save_audio_to_temp_file(self, audio_bytes):
        # Save the audio content to a temporary file
        temp_file_path = os.path.join(tempfile.gettempdir(), "temp_audio.mp3")
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(audio_bytes)
        return temp_file_path

    def split_audio_on_silence(self, audio_path, output_path, silence_thresh=-40, min_silence_len=1000):
        print(f"Splitting audio from: {audio_path}")
        # Load the audio from the provided path
        audio = AudioSegment.from_file(audio_path)

        # Split on silences
        segments = split_on_silence(audio, silence_thresh=silence_thresh, min_silence_len=min_silence_len)

        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Save each segment to the output directory
        output_files = []
        for i, segment in enumerate(segments):
            output_file = os.path.join(output_path, f"segment_{i + 1}.wav")
            segment.export(output_file, format="wav")
            output_files.append(output_file)

        print(f"Number of segments after splitting: {len(output_files)}")
        print(f"Segment paths after splitting: {output_files}")

        return output_files

    def transcribe_segment(self, segment_path):
        try:
            print(f"Transcribing audio segment from: {segment_path}")
            with open(segment_path, "rb") as audio_file:
                # Use the OpenAI SDK to transcribe the MP3 audio
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            print(f"Transcript: {transcript}")
            return transcript
        except Exception as e:
            # Handle exceptions or errors
            print(f"Error transcribing audio segment: {e}")
            return None

    def process_audio(self, audio_bytes):
        temp_file_path = None
        try:
            # Save audio to a temporary file
            temp_file_path = self.save_audio_to_temp_file(audio_bytes)

            # Split audio into chunks based on silences and save them
            output_directory = "output_directory"  # Change this to your desired directory
            output_segments = self.split_audio_on_silence(temp_file_path, output_directory)

            # Transcribe each segment
            transcribed_segments = []
            for i, segment_path in enumerate(output_segments):
                try:
                    transcript = self.transcribe_segment(segment_path)
                    if transcript:
                        transcribed_segments.append(transcript)
                except Exception as transcription_error:
                    print(f"Error during transcription for segment {i + 1}: {transcription_error}")

            print(f"Transcribed segments: {transcribed_segments}")
            return transcribed_segments
        except Exception as e:
            # Handle exceptions or errors during audio processing
            print(f"Error processing audio: {e}")
            return None
        finally:
            # Always clean up: Delete the temporary file after use
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
