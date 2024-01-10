import traceback
from openai import OpenAI
import os


class STT:
    def __init__(self) -> None:
        self.output_directory_name = os.environ.get(
            "OUTPUT_DIRECTORY_NAME", "output")
        # ディレクトリがなければ作る
        if not os.path.exists(self.output_directory_name):
            os.mkdir(self.output_directory_name)

    def create_text(self, audio_path):
        """音声ファイルからテキストを作る"""
        try:
            client = OpenAI()
            with open(audio_path, "rb") as f:
                # 音声ファイルを読み込んで変換
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="text"
                )

                # 変換結果からテキストを抽出して書き出す
                basename, _ = os.path.splitext(audio_path)
                text_file_root = basename.split("/")[-1]
                text_file_path = f"{self.output_directory_name}/{text_file_root}.txt"
                with open(text_file_path, mode="w", encoding="utf-8") as f0:
                    f0.write(transcript)
                return transcript

        except Exception as e:
            print(e)
            traceback.print_exc()
            exit()


if __name__ == "__main__":
    sample_audio_path = "audio/" + os.listdir("audio")[0]
    print(f"対象音声ファイル: {sample_audio_path}")

    stt = STT()
    ret_text = stt.create_text(sample_audio_path)
    print(f"作成されたテキスト: {ret_text}")
