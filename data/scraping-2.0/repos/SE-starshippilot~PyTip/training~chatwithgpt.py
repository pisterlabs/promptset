import openai

# 设置您的 OpenAI API 密钥
openai.api_key = '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'



# 定义要生成问题的主题
topic = "generate common questions on using Matplot lib, only present questions"

# 生成1000个问题
response = openai.Completion.create(
    engine="text-davinci-002",  # 选择适当的引擎
    prompt=topic,
    max_tokens= 2000,
    n= 100
   # 生成1000个问题
)

# 提取生成的问题
questions = [item['text'].strip() for item in response.choices]

# print(response.choices)
# 将问题保存到一个文档中
with open('matplotlib_questions.txt', 'w') as file:
    file.write('\n'.join(questions))

print(f"Generated and saved {len(questions)} questions to 'matplotlib_questions.txt'.")
