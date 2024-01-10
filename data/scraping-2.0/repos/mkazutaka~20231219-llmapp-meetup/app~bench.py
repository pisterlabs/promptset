import openai
import json
from tqdm import tqdm

from app.agents import OriginAssistant, OpenAIAssistant
from app.appcsv import write_score, write_any


QUESTIONS = [
    "インボイス制度について教えてください",
    "インボイス制度において発行者のデメリットはなんですか？"
    "インボイスはいつまでに登録申請をする必要がありますか？",
    "インボイス制度が始まると、課税事業者はどうなりますか？",
    "簡易課税について教えて",
    "簡易課税のメリット・デメリットを教えて",
    "インボイス発行事業者になるためには、どうすればよいでしょうか？",
    "インボイスの記載事項について教えて",
    "仕入税額控除をするための要件を教えて",
    "「適格請求書等保存方式」の概要を教えてください",
    "適格請求書発行事業者の登録は、どのような手続で行うのですか。",
    "適格請求書等保存方式の下での仕入税額控除の要件を教えてください。",
    "適格請求書の様式は、法令又は通達等で定められていますか",
    "登録番号は、どのような構成ですか",
    "３万円未満の自動販売機や自動サービス機による商品の販売等は、適格請求書の交付義務 が免除されるそうですが、具体的にはどのようなものが該当しますか。",
]

SYSTEM_PROMPT = """
あなたは、チャットボットの返答が適切かを評価する優秀なアシスタントです。
ユーザの入力する質問と回答を以下の基準に従って評価してください

1. 適切性: 回答がきちんとした文章や丁寧な文章になっているか（1点は不適切、5点は非常に適切）
2. 正確性: 情報の正確さ（1点は誤情報、5点は完全に正確）
3. 補足情報: 回答に有用な補足情報が含まれているか（1点は全くなし、5点は非常に有用）

総合評価: これらの要素を踏まえた上で、回答に対する総合的な評価を1-5点で付けてください。

JSONで結果を出力します。
キー値には、appropriateness, accuracy, additional, score, commentを含めてください。総合評価のキー値は`score`にしてください。
"""

USER_PROMPT = """
質問: {question}
回答: {answer}
"""

target_files = [
    "invoice"
]
benches = [
    OpenAIAssistant(),
    OriginAssistant(top_k=2),
    OriginAssistant(top_k=3),
]


def evaluate(question: str, answer: str):
    user_prompt = USER_PROMPT.replace("{question}", question).replace("{answer}", answer)
    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        response_format={ "type": "json_object" }
    )
    return json.loads(response.choices[0].message.content)


def main():
    # evaluate(question, answer)
    for file in target_files:
        queries = QUESTIONS

        pbar = tqdm(total=len(queries), desc="Processing", leave=False)
        doc_path = f"data/{file}.pdf"
        for bench in benches:
            scores = []
            bench.build(doc_path)
            for query in queries:
                answer = bench.chat(query)
                result = evaluate(query, answer)
                scores.append(result["score"])

                print(result)
                print(scores)

                result['query'] = query
                result['answer'] = answer
                headers = [
                    "appropriateness",
                    "accuracy",
                    "additional",
                    "score",
                    "comment",
                    "query",
                    "answer",
                ]
                write_any('result/result_detail.csv', bench.name(), headers, result)
                pbar.update()
            write_score('result/result.csv', bench.name(), scores)
            pbar.close()


if __name__ == '__main__':
    main()
