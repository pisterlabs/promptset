import openai

# 配置您的 OpenAI API 密钥
openai.api_key = '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'  # jinli


# openai.api_key = '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'
def get_gpt4_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "You are a robot designed to write Feasibility Study report"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=8000
    )
    return response['choices'][0]['message']['content'].strip()

def main():
    while True:
        user_input = input("您：")
        if user_input.lower() == 'exit':
            print("退出程序.")
            break
        else:
            prompta="Write a Feasibility Study on Establishing an Internet Bank in Cambodia at maximum 8000 words based on"
            promptb=
            promptc=
            response = get_gpt4_response(prompt)
            print(f'GPT-4：{response}')

if __name__ == "__main__":
    main()
# cd C:\Users\frank\Desktop\jd5374
