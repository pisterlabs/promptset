import os
import argparse

from langchain.llms import OpenAI

from langchain.agents import Tool
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_api_key', type=str, default="dummuy")
    parser.add_argument('--model_name', type=str, default="text-davinci-003")
    parser.add_argument('--serp_api_key', type=str, default="dummuy")
    args = parser.parse_args()

    os.environ["SERPAPI_API_KEY"] = args.serp_api_key

    # ---------------------------------
    # モデル定義
    # ---------------------------------
    llm = OpenAI(
        openai_api_key=args.openai_api_key,
        model_name=args.model_name,
        temperature=0.9,    # 大きい値では出現確率が均一化され、より多様な文章が生成される傾向がある。低い値では出現確率の高い単語が優先され、より一定の傾向を持った文章が生成される傾向がある。
    )
    print("llm: ", llm)

    # ---------------------------------
    # LangChain Agents の Tools 定義
    # Tools : Agent が外部とやり取りをするために呼び出す外部関数や外部ツール
    # ---------------------------------
    tools = load_tools(
        [
            "serpapi",      # serpapi : Google 検索結果を取得する外部 API ツール
            "llm-math",     # llm-math : 算術計算をする LangChain ツール
        ],
        llm=llm
    )

    # Tool(name='Search', description='A search engine. Useful for when you need to answer questions about current events. Input should be a search query.' ... )
    # Tool(name='Calculator', description='Useful for when you need to answer questions about math.' ... )
    print("tools: ", tools)

    # ---------------------------------
    # Agent オブジェクト作成
    # ---------------------------------
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,    # AgentType.ZERO_SHOT_REACT_DESCRIPTION : Tools オブジェクトの `description` フィールドなどから、どのツールを用いるかを決める Agent
        verbose=True
    )

    # ---------------------------------
    # Agent 実行（LLM 推論で最適な外部ツール実行）
    # ---------------------------------
    try:
        response = agent.run("""
        今日の広島市の最高気温を教えて。
        そして、最高気温を2乗した結果を教えて。
        """)
        print(f"response: {response}")
    except Exception as e:
        print(f"Excception was occurred | {e}")
        exit(1)

    exit(0)
