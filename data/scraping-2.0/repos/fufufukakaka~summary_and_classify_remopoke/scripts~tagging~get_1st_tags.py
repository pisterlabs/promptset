import ast
import collections
import os

import openai
import pandas as pd
from tqdm.auto import tqdm

openai.api_key = os.environ["OPENAI_API_KEY"]


def get_1st_tag(title: str, summary: str) -> str:
    response = openai.ChatCompletion.create(
        temperature=0,
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": """
以下の文章はポケットモンスターに関するプレゼンテーション『「キミはせかいのはじまりをしっているか」～ポケモン本編の悪の組織とは～』の要約です。

このプレゼンテーションでは、ポケモンの世界における悪の組織について話されています。悪の組織はゲーム内で主人公と敵対し、ストーリーの展開上で重要な役割を果たしています。具体的には、ロケット団やプラズマ団などが取り上げられ、彼らがポケモンを利用して世界を支配しようとしていることが紹介されています。また、悪の組織の目的や思想の違い、ポケモンと人間の関係についても考えさせられる内容です。さらに、地方の中枢に影響力を持つ組織や共同体から外れた存在としての悪の組織についても触れられています。このプレゼンテーションは、ポケモンの世界観や悪の組織についての理解を深めるためのものであり、子供たちにとっても重要な要素となっています。

これを説明する簡潔なタグを考えてください。イメージはツイッターのハッシュタグのような感じです。出力は10個のタグを配列で返してください。なお、リモポケ学会、というタグは除外してください。
""",
            },
            {
                "role": "assistant",
                "content": '["#ポケモン悪の組織","#ロケット団の野望","#プラズマ団の思想","#ポケモン世界観","#ポケモンと人間の関係","#組織の目的と差異","#ストーリーの裏側","#子供たちの冒険","#世界支配の野心","#組織の影響力"]',
            },
            {
                "role": "user",
                "content": f"以下の文章はポケットモンスターに関するプレゼンテーション『{ title }』の要約です。",
            },
            {
                "role": "user",
                "content": summary,
            },
            {
                "role": "user",
                "content": "これを説明する簡潔なタグを考えてください。イメージはツイッターのハッシュタグのような感じです。出力は10個のタグを配列で返してください。なお、リモポケ学会、というタグは除外してください。",
            },
        ],
    )
    res = response["choices"][0]["message"]["content"]
    return res


def main():
    """
    まず各要約文章からタグを生成します(1st stage)

    1st stage で得られたタグを元に、分類に使えそうなタグとしてまとめあげたものを生成します(2nd stage)

    タグ付与は、オフラインで手動処理後、次のスクリプトで行います。
    """

    df = pd.read_csv("output/remopoke_videos_summary_langchain_35turbo.csv")
    tags = []
    for title, summary in zip(tqdm(df["title"].values), df["summary"].values):
        _tags = get_1st_tag(title, summary)
        _tags = ast.literal_eval(_tags)
        tags.append(_tags)
    # df["1st_tag"] = tags
    # df.to_csv(
    #     "output/remopoke_videos_summary_langchain_35turbo_1st_tags.csv", index=False
    # )

    # resume
    # df = pd.read_csv("output/remopoke_videos_summary_langchain_35turbo_1st_tags.csv")
    # tags = df["1st_tag"].values

    # 2nd stage
    all_tags = []
    for tag in tags:
        all_tags.extend(ast.literal_eval(tag))
    all_tags = list(set(all_tags))

    counters = collections.Counter(all_tags)
    tags2 = [v[0] for v in counters.most_common() if v[1] > 1]
    print(tags2)


if __name__ == "__main__":
    main()
