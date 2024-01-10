import openai

# 替换为你的OpenAI API密钥
api_key = "YOUR_API_KEY_HERE"

# 问题和答案的字典
questions_and_answers = {
    "tmux的介绍": "",
    "tmux的使用方法": "",
    "vim的介绍": "",
    "vim的使用方法": "",
    "nano的介绍": "",
    "nano的使用方法": "",
}

# 循环遍历问题，发送请求并将答案保存到字典中
for question in questions_and_answers:
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=question + "\n",
        max_tokens=100,  # 调整根据答案长度需要的标记数
        api_key=api_key,
    )
    answer = response.choices[0].text.strip()
    questions_and_answers[question] = answer

# 将答案保存到Markdown文件
with open("answers.md", "w", encoding="utf-8") as file:
    for question, answer in questions_and_answers.items():
        file.write(f"{question}\n")
        file.write(f"    {answer}\n\n")