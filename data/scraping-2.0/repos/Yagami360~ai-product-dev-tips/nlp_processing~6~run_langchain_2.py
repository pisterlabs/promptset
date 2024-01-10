import argparse
from langchain import PromptTemplate
from langchain import FewShotPromptTemplate
from langchain.llms import OpenAI

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_api_key', type=str, default="dummuy")
    parser.add_argument('--model_name', type=str, default="text-davinci-003")
    parser.add_argument('--save_propmt_filepath', type=str, default="out_data/prompt-template-2.json")
    args = parser.parse_args()

    # ---------------------------------
    # PromptTemplate オブジェクト作成
    # ---------------------------------
    template = """
    英語: {keyword1}
    日本語: {keyword2}\n
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["keyword1", "keyword2"],
    )

    # ---------------------------------
    # FewShotPromptTemplate オブジェクト作成
    # ---------------------------------
    # Few-shot learning（いくつかの正解例を与えた後に、質問文を投げる形式）の正解例
    # PromptTemplate の {...} 部分の値を定義することで正解例を与える
    fewshot_examples = [
        {
            "keyword1": "cat",
            "keyword2": "猫",
        },
        {
            "keyword1": "dog",
            "keyword2": "犬",
        },
    ]

    fewshot_prompt = FewShotPromptTemplate(
        examples=fewshot_examples,              # Few-shot learning での正解例
        example_prompt=prompt,                  # PromptTemplate オブジェクト
        prefix="英語を日本語に翻訳してください",     # プロンプト（質問文）
        suffix="英語 : {input}\n",               # Few-shot learning での正解例における接頭語（この例では "英語 : "） 
        input_variables=["input"],              # suffix の {...} の変数名
        example_separator="\n\n",
    )

    # プロンプトテンプレートの json データをローカル環境に保存
    fewshot_prompt.save(args.save_propmt_filepath)

    # ---------------------------------
    # FewShotPromptTemplate オブジェクトからプロンプト文生成
    # ---------------------------------
    prompt_text = fewshot_prompt.format(input="cheese")
    print("prompt_text: ", prompt_text)

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
    # LLM推論実行
    # ---------------------------------
    try:
        response = llm(prompt=prompt_text)
        print(f"response: {response}")
    except Exception as e:
        print(f"Excception was occurred | {e}")
        exit(1)

    exit(0)
