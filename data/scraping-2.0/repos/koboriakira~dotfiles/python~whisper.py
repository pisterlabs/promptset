"""
実行コマンド
python ~/.dotfiles/python/whisper.py {音声ファイルの絶対パス}
"""
import os
import openai
import sys

if len(sys.argv) != 2:
    print("2つ以上の引数が指定されています")
    sys.exit()


file_path = sys.argv[1] # 絶対パス
dir_name = os.path.dirname(file_path) # ディレクトリ名
file_name = os.path.basename(file_path) # ファイル名
name, ext = os.path.splitext(file_name) # 名前と拡張子

txt = ""
with open(file_path, "rb") as audio_file:
    transcript = openai.Audio.transcribe(
        "whisper-1", audio_file, verbose=True, language="ja", task="translate")
    txt = transcript['text']

print(txt)
with open(f"{dir_name}/{name}.txt", "w") as f:
    f.write(txt)
