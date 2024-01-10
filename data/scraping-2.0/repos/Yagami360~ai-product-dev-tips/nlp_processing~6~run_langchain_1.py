import argparse
from langchain import PromptTemplate
from langchain.llms import OpenAI

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_api_key', type=str, default="dummuy")
    parser.add_argument('--model_name', type=str, default="text-davinci-003")
    parser.add_argument('--template', type=str, default="{keyword1}と{keyword2}について教えてください")
    parser.add_argument('--keyword1', type=str, default="LLM")
    parser.add_argument('--keyword2', type=str, default="ChatGPT")
    parser.add_argument('--save_propmt_filepath', type=str, default="out_data/prompt-template-1.json")
    args = parser.parse_args()

    # PromptTemplate オブジェクト作成（バリデーションなし）
    # prompt = PromptTemplate.from_template(
    #     template=args.template      # 例 : "{keyword1}と{keyword2}について教えてください"
    # )

    # PromptTemplate オブジェクト作成（バリデーションあり）
    prompt = PromptTemplate(
        template=args.template,      # 例 : "{keyword1}と{keyword2}について教えてください"
        input_variables=["keyword1", "keyword2"],
    )

    # プロンプトテンプレートの json データをローカル環境に保存
    prompt.save(args.save_propmt_filepath)

    # プロンプト文生成
    # 例 : "{keyword1}と{keyword2}について教えてください" -> "{LLM}と{ChatGPT}について教えてください"
    prompt_text = prompt.format(keyword1=args.keyword1, keyword2=args.keyword2)
    print("prompt_text: ", prompt_text)

    # モデル定義
    llm = OpenAI(
        openai_api_key=args.openai_api_key,
        model_name=args.model_name,
        temperature=0.9,    # 大きい値では出現確率が均一化され、より多様な文章が生成される傾向がある。低い値では出現確率の高い単語が優先され、より一定の傾向を持った文章が生成される傾向がある。
    )
    print("llm: ", llm)

    # LLM推論実行
    try:
        response = llm(prompt=prompt_text)
        print(f"response: {response}")
    except Exception as e:
        print(f"Excception was occurred | {e}")
        exit(1)

    exit(0)
