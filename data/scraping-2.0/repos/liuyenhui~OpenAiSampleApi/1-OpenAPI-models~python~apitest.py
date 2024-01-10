from openai import OpenAI
client = OpenAI()

# 获取模块
models = client.models.list()
# 打印支持的模块
for model in models:
    print(model.id)



