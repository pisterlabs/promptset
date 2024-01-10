import argparse
from langchain.llms import OpenAI

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_api_key', type=str, default="dummuy")
    parser.add_argument('--model_name', type=str, default="text-davinci-003")
    parser.add_argument('--prompt', type=str, default="今日は天気が良いですね")
    args = parser.parse_args()

    # モデル定義
    llm = OpenAI(
        openai_api_key=args.openai_api_key,
        model_name=args.model_name,
        temperature=0.9,    # 大きい値では出現確率が均一化され、より多様な文章が生成される傾向がある。低い値では出現確率の高い単語が優先され、より一定の傾向を持った文章が生成される傾向がある。
    )
    print("llm: ", llm)

    # LLM推論実行
    try:
        response = llm(prompt=args.prompt)
        print(f"response: {response}")
    except Exception as e:
        print(f"Excception was occurred | {e}")
        exit(1)

    exit(0)
