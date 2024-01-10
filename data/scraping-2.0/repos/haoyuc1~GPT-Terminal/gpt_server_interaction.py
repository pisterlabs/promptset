import openai
import json
import subprocess
import os

def validate_api_key(api_key):
    try:
        openai.api_key = api_key
        openai.Engine.list()
        return True
    except Exception as e:
        print(e)
        return False

def load_config(file_path="config.json"):
    try:
        with open(file_path, "r") as f:
            config = json.load(f)
            if "api_key" not in config or not config["api_key"] or not validate_api_key(config["api_key"]):
                while True:
                    print("API密钥未在配置文件中找到或无效，请输入您的API密钥：")
                    api_key = input()
                    if validate_api_key(api_key):
                        config["api_key"] = api_key
                        with open(file_path, "w") as f:
                            json.dump(config, f)
                        break
                    else:
                        print("无效的API密钥，请重新输入。")
            return config
    except FileNotFoundError:
        while True:
            print("配置文件未找到，请输入您的API密钥：")
            api_key = input()
            if validate_api_key(api_key):
                config = {"api_key": api_key}
                with open(file_path, "w") as f:
                    json.dump(config, f)
                break
            else:
                print("无效的API密钥，请重新输入。")
        return config

def interact_with_gpt(prompt):
    conversation_history = [
        {"role": "system", "content": "You are interacting with a server, GPT-4 will generate server commands and chat content based on your input. Please output commands and chat content in JSON format as follows:\n{\"command\": \"your_command\", \"chat\": \"your_chat_content\"}"},
        {"role": "user", "content": prompt}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation_history,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )

    return response.choices[0].message["content"].strip()

def execute_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout.decode('utf-8'), stderr.decode('utf-8')

def main():

    # 加载配置文件
    config = load_config()
    # 设置API密钥
    openai.api_key = config["api_key"]
 #"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" #config["api_key"]

    while True:
        # 获取用户输入
        user_input = input("请输入您的问题或命令（输入'退出'以结束）：")

        # 如果用户输入“退出”，则跳出循环
        if user_input.lower() == "退出":
            break

        # 将带有上下文的用户输入发送给GPT-4
        gpt_response = interact_with_gpt(user_input)
        print(f"GPT-4生成的JSON: {gpt_response}")  # 打印GPT-4生成的JSON

        try:
            gpt_response_json = json.loads(gpt_response)
            if "command" in gpt_response_json:
                gpt_command = gpt_response_json["command"]
                print(f"GPT-4生成的命令：{gpt_command}")

            if "chat" in gpt_response_json:
                gpt_chat = gpt_response_json["chat"]
                print(f"GPT-4回应：{gpt_chat}")

        except (json.JSONDecodeError, KeyError):
            print("GPT-4没有生成有效的JSON格式命令或聊天内容。请尝试其他问题或命令。")
            continue

        execute_choice = input(f"您要执行这个命令吗（输入'no'不执行，输入其他任何内容则执行）：{gpt_command}\n")

        if execute_choice.lower() != "no":
            stdout, stderr = execute_command(gpt_command)
            print(f"服务器输出：\n{stdout}\n{stderr}")
            prompt = f"我刚刚执行了以下命令：\n{gpt_command}\n服务器的输出如下：\n{stdout}\n{stderr}\n请给出回应。"
        else:
            print("您选择了不执行该命令。")
            prompt = f"我没有执行以下命令：\n{gpt_command}\n请给出回应。"

        gpt_response = interact_with_gpt(prompt)

        # 打印GPT-4的回应
        print(f"GPT-4回应：{gpt_response}")

if __name__ == "__main__":
    main()
