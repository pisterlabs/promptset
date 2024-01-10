import openai

# 设置您的 OpenAI API 密钥
openai.api_key = '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'

# 从输入文件中读取问题
input_file_path = 'commonquest.txt'
with open(input_file_path, 'r') as file:
    questions = file.readlines()

# 初始化空列表来存储问题和回答
question_answer_pairs = []

# 向 ChatGPT 提问并将问题和回答存储到列表中
for question in questions:
    question = question.strip()
    response = openai.Completion.create(
        engine="text-davinci-002",  # 选择适当的引擎
        prompt=question+", if any function is involved, you will give me the function namethat most related to my query. For example, the guery is \
            'how to draw a histogram' and your answer will be 'plt.hist()' and nothing else.",
        max_tokens=4000
    )
    answer = response.choices[0].text.strip()
    question_answer_pairs.append(f"Q: {question}\nA: {answer}\n")

# 将问题和回答写入输出文件
output_file_path = 'Q&A_v2.txt'
with open(output_file_path, 'w') as file:
    file.writelines(question_answer_pairs)

print(f"Questions and answers saved to: {output_file_path}")
