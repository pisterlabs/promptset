from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os
import sys
from pydub import AudioSegment

MAX_LENGTH = 4096


def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def split_text(text, max_length):
    words = text.split()
    split_texts = []
    current_text = ""
    for word in words:
        if len(current_text) + len(word) + 1 > max_length:
            split_texts.append(current_text)
            current_text = word
        else:
            current_text += " " + word
    split_texts.append(current_text)
    return split_texts


def generate_speech(client, text, model_name="tts-1", voice="nova"):
    response = client.audio.speech.create(
        model=model_name, voice=voice, input=text.strip()
    )
    return response


def main(text_file_path):
    # 環境変数のロード
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)

    print("loaded environment variables")

    # テキストファイルの読み込み
    text = load_text(text_file_path)
    print("loaded text file")

    # テキストを分割
    split_texts = split_text(text, MAX_LENGTH)
    print("split text")

    # 音声の生成と一時保存
    combined = AudioSegment.empty()
    temp_files = []
    for i, part_text in enumerate(split_texts, start=1):
        speech = generate_speech(client, part_text)
        temp_file_path = Path(text_file_path).with_name(
            Path(text_file_path).stem + f"_temp_{i}.mp3"
        )
        with open(temp_file_path, "wb") as file:
            file.write(speech.content)
        print(f"generated speech part: {temp_file_path}")

        # 部分音声を読み込み、結合
        part = AudioSegment.from_mp3(temp_file_path)
        combined += part

        # 一時ファイルのパスを保存
        temp_files.append(temp_file_path)

    # 結合された音声ファイルを保存
    combined_file_path = Path(text_file_path).with_name(
        Path(text_file_path).stem + ".mp3"
    )
    combined.export(combined_file_path, format="mp3")
    print(f"Combined speech saved as {combined_file_path}")

    # 一時ファイルの削除
    for temp_file in temp_files:
        os.remove(temp_file)
        print(f"Removed temporary file: {temp_file}")


# スクリプトの実行
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tts-openai.py <text_file_path>")
        sys.exit(1)

    text_file_path = sys.argv[1]
    main(text_file_path)
