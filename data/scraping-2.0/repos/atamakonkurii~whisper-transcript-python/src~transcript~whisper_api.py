import os
import openai
import glob
import shutil
import time

def transcript(input_audio_file, output_file_name, api_key, prompt):
  # apikeyを渡す
  openai.api_key = api_key

  # audioファイル読み込み
  audio_file= open(input_audio_file, "rb")

  # whisper-1モデルで音声認識⇨Japanese
  transcriptJapanese = openai.Audio.transcribe(model="whisper-1", 
                                              file=audio_file,
                                              language="ja",
                                              temperature=0.1,
                                              response_format="srt",
                                              prompt=prompt,
                                              )

  with open(output_file_name, 'w') as f:
    print(transcriptJapanese, file=f)

# プロンプト
prompt = "日本の家族が台湾に旅行に来てくれました" 

api_key = os.environ["OPENAI_API_KEY"]
input_audio_dir = "./docker_share/output/mp3/split"  # MP3ファイルがあるディレクトリ
output_srt_dir = "./docker_share/output/srt/japanese"  # SRTファイルを出力するディレクトリ

# output_dir内のすべてのMP3ファイルを取得
input_audio_files = glob.glob(os.path.join(input_audio_dir, '*.mp3'))

# ファイルの作成時刻でソート
sorted_audio_files = sorted(input_audio_files, key=os.path.getctime)

# 出力ディレクトリが存在する場合は削除してから再作成
if os.path.exists(output_srt_dir):
    shutil.rmtree(output_srt_dir)
os.makedirs(output_srt_dir)

# 作成順にファイル名を表示
for i, audio_file in enumerate(sorted_audio_files):
    output_file_name = os.path.join(output_srt_dir, f"chunk_{i+1}.srt") # 生成後のSRTファイルの名前
    transcript(audio_file, output_file_name, api_key, prompt)
    print(f"2秒待機します。")
    time.sleep(2)
    print(f"2秒待機しました。")
    
