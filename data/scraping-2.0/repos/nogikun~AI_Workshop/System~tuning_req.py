try:
    from openai import OpenAI
except:
    print("ライブラリがインポートできませんでした。プログラムを終了します。")
    exit()

# Setting Client
client = OpenAI(
    # APIキーを設定
    api_key=input('OpenAIのAPIキーを入力してください:')
)


#client.api_key = input('OpenAIのAPIキーを入力してください:')
file_path = input('ファイルパスを教えてください（ファイル名を除く）：')

# ハイパーパラメータを設定
params = {
    "n_epochs":3 # n_epochs = 3 の場合、学習時間は30分程度
    }

# チューニングデータをアップロード
file_response = client.files.create(
  file=open(f"{file_path}/TuningData.jsonl", "rb"),
  purpose='fine-tune'
)
# アップロードしたファイルのIDを保存
file_id = file_response.id

# 学習実行
fine_tuning_response = client.fine_tuning.jobs.create(
  training_file=file_id,
  model="gpt-3.5-turbo",
  hyperparameters=params
)
# ジョブIDを保存
job_id = fine_tuning_response.id

# ジョブIDを出力
print(f"Job ID: {job_id}")
