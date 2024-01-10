from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
import os

# 環境変数の準備
os.environ["OPENAI_API_KEY"] = "XXX"
os.environ["GOOGLE_CSE_ID"] = "XXX"
os.environ["GOOGLE_API_KEY"] = "XXX"

# LLMの設定
llm = OpenAI(model_name="gpt-3.5-turbo")

# 使用するツールをロード
tools = load_tools(["google-search"], llm=llm)

# エージェントを初期化
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# 会話履歴を格納するための変数
conversation_history = ""

def read_files_recursively(path, type):
    files_dict = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(type):
                with open(os.path.join(root, file), 'r') as f:
                    contents = f.read()
                    files_dict[file] = contents
    return files_dict

def send_gpt(prompt):
    try:
        response = agent.run(prompt)
    except ValueError as e:
        # エラーが "Could not parse LLM output: `" で始まる場合、エラーメッセージを整形して response に格納
        response = str(e)
        if not response.startswith("Could not parse LLM output: `"):
            raise e
        response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
    return response

if __name__ == "__main__":
    # Specify the directory path where your program files are located
    directory_path = input("プログラムファイルのパスを入力してください：")
    type = input("プログラムの種類を入力してください（ex：.kt）：")
    
    files_dict = read_files_recursively(directory_path, type)
    for filename, contents in files_dict.items():
        conversation_history += f"File: {filename}\nContents:\n{contents}\n---\n"
        response = send_gpt(f"ユーザー: 下記のソースコードがあります。\n{conversation_history}\n")
        print("回答:", response)
        conversation_history += f"ChatGPT: {response}\n"

    while True:
        # ユーザーからの入力を受け付ける
        user_input = input("質問を入力してください (終了するには 'exit' と入力してください)：")
        
        # 入力が 'exit' の場合、ループを終了
        if user_input.lower() == "exit":
            break
        
        # 会話履歴にユーザーの入力を追加
        conversation_history += f"ユーザー: {user_input}\n"
        
        # エージェントに会話履歴を与えて回答を生成
        response = send_gpt(conversation_history)

        # 回答を表示
        print("回答:", response)
        
        # 会話履歴にエージェントの回答を追加
        conversation_history += f"ChatGPT: {response}\n"
