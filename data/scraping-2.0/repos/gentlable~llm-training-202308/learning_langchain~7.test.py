# Import things that are needed generically
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import load_tools, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool

llm = ChatOpenAI(temperature=0)
tools = load_tools(["google-search"], llm=llm)

# エージェントを初期化
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# 会話履歴を格納するための変数
conversation_history = ""

if __name__ == "__main__":
    while True:
        # ユーザーからの入力を受け付ける
        user_input = input("質問を入力してください (終了するには 'exit' と入力してください)：")
        
        # 入力が 'exit' の場合、ループを終了
        if user_input.lower() == "exit":
            break
        
        # 会話履歴にユーザーの入力を追加
        conversation_history += f"ユーザー: {user_input}\n"
        
        # エージェントに会話履歴を与えて回答を生成
        try:
            response = agent.run(conversation_history)
        except ValueError as e:
            # エラーが "Could not parse LLM output: `" で始まる場合、エラーメッセージを整形して response に格納
            response = str(e)
            if not response.startswith("Could not parse LLM output: `"):
                raise e
            response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
        
        # 回答を表示
        print("回答:", response)
        
        # 会話履歴にエージェントの回答を追加
        conversation_history += f"ChatGPT: {response}\n"
