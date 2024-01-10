from openai import OpenAI
client = OpenAI(
    api_key=" ")

# # 생성된 데이터셋 파일을 openai 에 등록합니다.
# client.files.create(
#   file=open("data.jsonl", "rb"),
#   purpose="fine-tune"
# )
#
# print(client.files.list())

# client.fine_tuning.jobs.create(
#   training_file="file-5eLgGW054146tq9theG0dJHy",
#   model="gpt-3.5-turbo"
# )

client.fine_tuning.jobs.create(
  training_file="file-5eLgGW0541",
  model="gpt-3.5-turbo"
)
