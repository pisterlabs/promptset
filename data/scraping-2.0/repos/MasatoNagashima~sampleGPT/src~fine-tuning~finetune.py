from openai import OpenAI
from datetime import datetime

client = OpenAI()

# 学習用データのファイルパス
filepath = "resources/sample.jsonl"

# アップロード
train_file = client.files.create(
    file=open(filepath, "rb"),
    purpose='fine-tune'
    )

# ファインチューニング実行
client.fine_tuning.jobs.create(
    training_file = train_file.id, 
    model = 'gpt-3.5-turbo'
    )

# ファインチューニングの実行状況を確認
for job in client.fine_tuning.jobs.list(limit=10):
    print(job)
