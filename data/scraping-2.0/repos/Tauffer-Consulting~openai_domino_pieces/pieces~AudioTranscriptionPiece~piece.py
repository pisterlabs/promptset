from domino.base_piece import BasePiece
from .models import InputModel, OutputModel, SecretsModel
from openai import OpenAI
from pydub import AudioSegment


class AudioTranscriptionPiece(BasePiece):
    """
    This Piece uses the OpenAI API to extract text transcripts from audio.
    """
    def piece_function(self, input_data: InputModel, secrets_data: SecretsModel):
        openai_api_key = secrets_data.OPENAI_API_KEY
        if openai_api_key is None:
            raise Exception("OPENAI_API_KEY not found in ENV vars. Please add it to the secrets section of the Piece.")
        client = OpenAI(api_key=openai_api_key)

        # Input arguments are retrieved from the Input model object
        file_path = input_data.audio_file_path

        print("Making OpenAI audio transcription request...")
        try:
            full_audio = AudioSegment.from_mp3(file_path)
            total_time = len(full_audio)
            # PyDub handles time in milliseconds
            ten_minutes = 10 * 60 * 1000
            full_transcript = ""
            i = 0
            while True:
                # Split audio into 10 minute chunks, run transcription on each chunk
                print(f"Transcribing audio chunk {i+1}...")
                endpoint = min((i+1)*ten_minutes, total_time-1)
                minutes = full_audio[i*ten_minutes:endpoint]
                minutes.export(f"audio_piece_{i}.mp3", format="mp3")
                audio_file = open(f"audio_piece_{i}.mp3", "rb")
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    temperature=input_data.temperature
                )
                full_transcript += " " + transcript.text
                i += 1
                audio_file.close()
                if endpoint == total_time-1:
                    break
        except Exception as e:
            print(f"\nTrascription task failed: {e}")
            raise Exception(f"Transcription task failed: {e}")

        # Display result in the Domino GUI
        self.format_display_result(input_data=input_data, string_transcription_result=full_transcript)

        if input_data.output_type == "string":
            self.logger.info("Transcription complete successfully. Result returned as string.")
            return OutputModel(
                transcription_result=full_transcript,
                file_path_transcription_result=""
            )

        output_file_path = f"{self.results_path}/audio_transcription_result.txt"
        with open(output_file_path, "w") as f:
            f.write(full_transcript)

        if input_data.output_type == "file":
            self.logger.info(f"Transcription complete successfully. Result returned as file in {output_file_path}")
            return OutputModel(
                transcription_result="",
                file_path_transcription_result=output_file_path
            )

        self.logger.info(f"Transcription complete successfully. Result returned as string and file in {output_file_path}")
        return OutputModel(
            transcription_result=full_transcript,
            file_path_transcription_result=output_file_path
        )

    def format_display_result(self, input_data: InputModel, string_transcription_result: str):
        md_text = f"""
## Generated transcription:  \n
{string_transcription_result}  \n

## Args
**temperature**: {input_data.temperature}
"""
        file_path = f"{self.results_path}/display_result.md"
        with open(file_path, "w") as f:
            f.write(md_text)
        self.display_result = {
            "file_type": "md",
            "file_path": file_path
        }
