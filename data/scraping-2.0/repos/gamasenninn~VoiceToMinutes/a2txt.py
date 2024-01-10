import openai
import sys
import os
from dotenv import load_dotenv
import glob


# OpenAI APIキーを設定
load_dotenv()
openai.api_key = os.environ["OPEN_API_KEY"]

# 文字起こしファイル関連処理
def check_transcription_file_exists(file_path):
    # 元のファイル名から拡張子を除外し、.txtファイルのパスを作成
    base_name = os.path.splitext(file_path)[0]
    txt_file_path = f"{base_name}.txt"

    # ファイルが存在するかどうかをチェック
    return os.path.exists(txt_file_path)

def read_transcription_from_file(file_path):
    # .txtファイルのパスを生成
    base_name = os.path.splitext(file_path)[0]
    txt_file_path = f"{base_name}.txt"

    # ファイルから文字起こしテキストを読み込む
    with open(txt_file_path, "r", encoding="utf-8") as file:
        return file.read()

# 要約ファイル関連処理
def save_transcription_to_file(transcription, original_file_path):
    # 元のファイル名から拡張子を除外し、新しいファイル名を作成
    base_name = os.path.splitext(original_file_path)[0]
    new_file_path = f"{base_name}.txt"

    # 文字起こし結果をファイルに保存
    with open(new_file_path, "w", encoding="utf-8") as text_file:
        text_file.write(transcription)

def check_summrize_file_exists(file_path):
    # 元のファイル名から拡張子を除外し、.txtファイルのパスを作成
    base_name = os.path.splitext(file_path)[0]
    txt_file_path = f"{base_name}.json"

    # ファイルが存在するかどうかをチェック
    return os.path.exists(txt_file_path)

def save_summrize_to_file(sum_text, original_file_path):
    # 元のファイル名から拡張子を除外し、新しいファイル名を作成
    base_name = os.path.splitext(original_file_path)[0]
    new_file_path = f"{base_name}.json"

    # 文字起こし結果をファイルに保存
    with open(new_file_path, "w", encoding="utf-8") as text_file:
        text_file.write(sum_text)

# 要約処理（openai API）
def transcribe_audio(filename):
   
    with open(filename, "rb") as file:
        params ={
            "response_format" : "vtt",
            "temperature" : 0, 
            "language" : "ja" ,
            "prompt": 
                "句読点や読点を付与してください。"
        }
        transcription = openai.Audio.transcribe("whisper-1", file, **params)
        print("trans:",transcription)
        #return transcription.text
        return transcription

def summarize_text(text):
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            #model="gpt-4-1106-preview",
            response_format={ "type": "json_object" },
            messages=[
                {
                    "role": "system",
                    "content": "議事録の下書きを作成してください。"
                                "議事録ですのでタイトルと内容をセットにしてそれごとに複数書いてください"
                                "出力は純粋な配列のJSON形式でお願いします。" 
                                "{minutes:["
                                    "{"
                                        "title:タイトル（20文字以内で内容に最も適したタイトル）,\n" 
                                        "content:内容(要点や決定事項を箇条書きにしてください),\n" 
                                        "times:この議題の開始時間00:00:00-終了時間00:00:00\n" 

                                    "},"
                                "]}"
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
        )

    # 要約されたテキストを取得
    summary = response.choices[0].message.content.strip()
    return summary

# 文字起こし->要約の一連の流れ（主処理）
def v2txt_sum(filename):

    transcription_text = ""
    if check_transcription_file_exists(filename):
        if check_summrize_file_exists(filename):
            return    

    # 音声ファイルを文字起こし
    if not check_transcription_file_exists(filename):   #文字起こしファイルがなかったら

        print("文字起こしを開始します")
        transcription_text = transcribe_audio(filename)

        # 文字起こしテキストを標準出力に表示
        print("whisper AI 文字起こし:")
        print(transcription_text)
        # 文字起こしテキストを保存
        save_transcription_to_file(transcription_text, filename)
    else:
        # 文字起こしファイルが存在する場合、ファイルから読み込む
        transcription_text = read_transcription_from_file(filename)        

    # 要約処理
    #if not check_summrize_file_exists(filename): #文字起こしファイルがなかったら
    print("議事要約を開始します")
    sum_txt = summarize_text(transcription_text)
    print(sum_txt)
    save_summrize_to_file(sum_txt, filename)


if __name__ == "__main__":
    # コマンドライン引数または標準入力からファイル名を取得
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = input("ファイル名を入力してください:").strip()

    #print("文字起こしを開始します・・・・・\n")
    for file in glob.glob(filename):
        print(file)
        v2txt_sum(file)

    #input("\nなにかキーを押してください終了します:")
