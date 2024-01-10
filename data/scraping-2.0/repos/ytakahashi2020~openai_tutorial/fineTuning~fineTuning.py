from openai import OpenAI
import time
client = OpenAI()

upload_response = client.files.create(
    file=open("./projectA.jsonl", "rb"),
    purpose="fine-tune"
)

print(upload_response.id)

job = client.fine_tuning.jobs.create(
    training_file=upload_response.id,  # ここでファイルIDを使用
    model="gpt-3.5-turbo"
)

job_id = job.id  # ファインチューニングジョブのID

while True:
    job_status = client.fine_tuning.jobs.retrieve(job_id)
    print(f"現在のステータス: {job_status.status}")

    # ステータスが 'succeeded' または 'failed' の場合、ループを終了
    if job_status.status in ['succeeded', 'failed']:
        print("ジョブが完了しました。")
        break

    # 1分待機
    time.sleep(60)


completed_job = client.fine_tuning.jobs.retrieve(job_id)
model_name = completed_job.fine_tuned_model  # 生成されたモデルの名前

print(job_id)
print(completed_job)
print(model_name)

response = client.chat.completions.create(
    model=model_name,  # ここでファインチューニングされたモデルを使用
    messages=[
        {"role": "system", "content": "このチャットボットはProjectAの内容に対して回答します。"},
        {"role": "user", "content": "Project Aについて教えてください"}
    ]
)
print(response.choices[0].message.content)
