# 导入 openai 库
import openai

# 导入 os 库，用于获取环境变量
import os

# 从环境变量中获取 OpenAI 的 API 密钥，并设置为 openai 的 API 密钥
openai.api_key = os.getenv("OPENAI_API_KEY")

# 创建一个文件，用于微调模型
openai.File.create(
  file=open("mydata.jsonl", "rb"),  # 打开名为 mydata.jsonl 的文件，以二进制读取模式
  purpose='fine-tune'  # 文件的目的是微调
)

# 使用指定的训练文件和模型进行微调
openai.FineTuningJob.create(training_file="file-abc123", model="gpt-3.5-turbo")

# 列出最近的 10 个微调工作
openai.FineTuningJob.list(limit=10)

# 获取某个微调任务的状态
openai.FineTuningJob.retrieve("ft-abc123")

# 取消一个微调任务
openai.FineTuningJob.cancel("ft-abc123")

# 列出某个微调任务的最多 10 个事件
openai.FineTuningJob.list_events(id="ft-abc123", limit=10)

# 删除一个微调过的模型（必须是创建模型的组织的所有者）
openai.Model.delete("ft-abc123")


completion = openai.ChatCompletion.create(
  model="ft:gpt-3.5-turbo:my-org:custom_suffix:id",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)