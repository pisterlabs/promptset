import openai
import os
import tempfile
import wave

API_KEY = os.environ.get('OPENAI_API_KEY')
if not API_KEY:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
openai.api_key = API_KEY

def save_to_wav(data):
    temp_file_name = tempfile.mktemp(suffix=".wav")
    with wave.open(temp_file_name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(44100)
        wf.writeframes(data)
    return temp_file_name

def get_transcription(data):
    wav_file_path = save_to_wav(data)
    
    try:
        with open(wav_file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file, response_format="vtt")
            transcript_text = extract_transcription_from_vtt(transcript)
        return transcript_text
    finally:
        os.remove(wav_file_path)

# 字幕ファイルから音声認識結果を抽出する
def extract_transcription_from_vtt(vtt_content):
    lines = vtt_content.split("\n")
    # ヘッダーや空行、タイムスタンプを取り除く
    lines = [line.strip() for line in lines if not (line.startswith("WEBVTT") or line.startswith("00:") or line == "")]
    return "\n".join(lines)
