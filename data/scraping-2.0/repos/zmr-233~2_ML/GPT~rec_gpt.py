import openai
import os

# 设置API基础地址和API密钥
api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com")  # 默认为OpenAI官方API地址
api_key = os.getenv("OPENAI_API_KEY")

# 初始化对话历史
dialog_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

def chat_with_gpt(user_input):
    # 将用户的新消息添加到对话历史中
    dialog_history.append({"role": "user", "content": user_input})

    try:
        # 发送聊天请求，包括整个对话历史
        response = openai.ChatCompletion.create(
            model="gpt-4",  # 假设gpt-4是模型的标识符
            messages=dialog_history
        )
        # 提取并返回GPT的回应
        gpt_response = response['choices'][0]['message']['content']
        # 将助手的回应添加到对话历史中
        dialog_history.append({"role": "assistant", "content": gpt_response})
        return gpt_response
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        gpt_response = chat_with_gpt(user_input)
        print(f"GPT: {gpt_response}")
