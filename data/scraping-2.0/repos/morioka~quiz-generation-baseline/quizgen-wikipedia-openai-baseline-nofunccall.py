# # AI王 クイズ生成 ベースライン
# 
# ## 概要
# - 与えられたテーマをもとにして早押しクイズを生成する
# - 入力：テーマ
# - 出力：クイズ(質問、正解, 出典)

import argparse
import json
import os
import random
import time

import numpy as np
import openai
import pandas as pd

import wikipedia

wikipedia.set_lang("ja")


def get_wikipedia_content(theme :str = None, content_len :int = 6000):
  # Wikipediaの見出し語であるか
  titles = wikipedia.search(theme)
  if theme not in titles:
    return None

  # Wikipediaページを取得
  wp = wikipedia.page(theme)
  if wp.title != theme:
    return None

  content = wp.content[:content_len]

  return content


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
#    torch.manual_seed(seed)
#    torch.cuda.manual_seed(seed)
#    torch.backends.cudnn.deterministic = True
#    torch.backends.cudnn.benchmark = False

openai.api_key = os.getenv("OPENAI_API_KEY")


QG_SYSTEM_PROMPT = """あなたはプロのクイズ作家です。早押しクイズを作成して下さい。
以下の制約条件をもとに、早押しクイズを作ってください。

# 制約条件：
・与えられたテーマの単語または文章に基づいて、問題文と正解からなるクイズを1つ作ってください。
・クイズの「問題文」は、前半の「前振り」と後半の「後限定」をつないで作ってください。自然な文章にしてください。
・「正解」は、テーマに基づいた問題文に対する答えです。テーマと異なる単語を選んでください。
・「前振り」は、正解を説明する修飾表現です。できるだけ知って役立つ情報を盛り込んでください。
・「後限定」は、疑問文です。必ず「でしょう？」で終わってください。皆が知っているような、正解を確実に導き出せる確実な情報を盛り込んでください。
・問題文は与えられたテーマの単語または文章の表現を変更して用います。
・問題文の長さは必ず60文字から80文字以内です。
・正解の長さは単語1個分です。
・以下の項目で日本語で出力してください。

```
問題文:クイズの問題文
正解：クイズの正解
```

"""

QG_USER_PROMPT = """テーマ:{theme}
"""

QG_REVIEW_SYSTEM_PROMPT = """あなたはプロのクイズ作家です。
以下の制約条件をもとに、与えられた早押しクイズをよりよいものに修正してください。

# 制約条件：
・クイズの問題文と正解のペアが与えられます。一般的なクイズとして適切ならばそのまま出力してください。適切でないならば、適切なものになるように問題文または正解を修正してください。
・クイズの問題文は、前半の前振りと後半の後限定をつないで作られます。
・「前振り」は、正解を説明する修飾表現です。できるだけ知って役立つ情報を含みます。
・「後限定」は、疑問文です。必ず「でしょう？」で終わります。皆が知っているような、正解を確実に導き出せる確実な情報を含みます。
・テーマの単語または文章とクイズの問題文が与えられます。問題文がテーマの記述に沿っているならばそのまま出力してください。そうでないならば、テーマの記述のみに含まれるよう問題文または正解を修正してください。
・以下の項目で日本語で出力してください。

```
問題文:修正後のクイズの問題文
正解：修正後のクイズの正解
```
"""

QG_REVIEW_USER_PROMPT = """テーマ:{theme}
問題文:{question}
正解:{answer}
"""

def generate_quiz(theme, 
                  retry_max=0, 
                  interval=1, 
                  model="gpt-3.5-turbo", 
                  debug=False,
                  temperature=0.7):

    def generate(theme):
        try:
            messages=[
                {"role": "system", "content": QG_SYSTEM_PROMPT},
                {"role": "user", "content": QG_USER_PROMPT.format(theme=theme)}
            ]
            
            completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )

            if debug:
                print(completion)
            message = completion["choices"][0]["message"]

            content = message['content']
            dict = {}    
            for b in content.split('\n'):
                if debug:
                    print("****", b)
                if b.startswith("問題文"):
                    dict['question'] = b[4:].lstrip()
                if b.startswith("正解"):
                    dict['answer'] = b[3:].lstrip()
            if debug:
                print("Dict by ChatGPT", dict)
            return dict

        except:
            pass

        return None

    # サービス応答次第でリトライ
    res = None
    for _ in range(retry_max + 1):
        res = generate(theme)
        if res is not None:
            break
        time.sleep(interval)

    return res


def review_quiz(quiz, 
                retry_max=0, 
                interval=1, 
                model="gpt-3.5-turbo", 
                  debug=False,
                  temperature=0.7):

    def review(quiz):
        try:
            material = quiz['theme']
            if quiz['reference'] is not None:
                material = quiz['reference']

            messages=[
                {"role": "system", "content": QG_REVIEW_SYSTEM_PROMPT},
#                {"role": "user", "content": QG_REVIEW_USER_PROMPT.format(theme=quiz['theme'],
                {"role": "user", "content": QG_REVIEW_USER_PROMPT.format(theme=material,
                                                                            question=quiz['question'],
                                                                            answer=quiz['answer'])}
            ]

            completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )

            if debug:
                print(completion)
            message = completion["choices"][0]["message"]

            content = message['content']
            dict = {}    
            for b in content.split('\n'):
                if debug:
                    print("****", b)
                if b.startswith("問題文"):
                    dict['question'] = b[4:].lstrip()
                if b.startswith("正解"):
                    dict['answer'] = b[3:].lstrip()
            if debug:
                print("Dict by ChatGPT", dict)
            return dict

        except:
            pass

        return None


    # サービス応答次第でリトライ
    res = None
    for _ in range(retry_max + 1):
        res = review(quiz)
        if res is not None:
            break
        time.sleep(interval)

    return res

QG_MATERIAL_SYSTEM_PROMPT="""あなたはプロのクイズ作家です。

以下の制約条件をもとに、早押しクイズの素材を選んでください。

# 制約条件：
・与えられたテーマの単語または文章に基づいて、早押しクイズの問題文の前半「前振り」としてふさわしい区間と、問題文の後半の「後限定」としてふさわしい区間をそれぞれ抽出してください。
・クイズの難易度は、一般的な大人向けです。
・「前振り」は、答えを説明する修飾表現です。できる限り、聞いてためになる情報を盛り込んでください。200文字程度抜き出してください。
・「後限定」は、皆が知っているような、答えを確実に導き出せる確実な情報を含んでください。200文字程度抜き出してください。
・前振りと後限定の内容は重複しません。
・前振りと後限定をそれぞれ1箇所、合計2か所を、リストにして出力してください。
・以下の項目で日本語で出力してください。

```
前振り:クイズの前振り素材
後限定：クイズの後限定
```

"""

QG_MATERIAL_USER_PROMPT = """テーマ: {content}
"""


def pickup_quiz_material(theme, 
                         retry_max=0, 
                         interval=1, 
                         model="gpt-3.5-turbo", 
                         max_tokens=1500, 
                         debug=False,
                         temperature=0.7):

    def pickup(content):
        try:
            messages=[
                {"role": "system", "content": QG_MATERIAL_SYSTEM_PROMPT},
                {"role": "user", "content": QG_MATERIAL_USER_PROMPT.format(content=content)}
            ]
            
            completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )

            if debug:
                print(completion)
            message = completion["choices"][0]["message"]

            content = message['content']
            dict = {}    
            for b in content.split('\n'):
                if debug:
                    print("****", b)
                if b.startswith("前振り"):
                    dict['maefuri'] = b[4:].lstrip()
                if b.startswith("後限定"):
                    dict['atogentei'] = b[4:].lstrip()
            if debug:
                print("Dict by ChatGPT", dict)
            return dict

        except:
            pass

        return None

    # サービス応答次第でリトライ
    res = None
    for _ in range(retry_max + 1):
        res = pickup(theme)
        if res is not None:
            break
        time.sleep(interval)

    return res


def main(args):

    set_seed(args.seed)

    input_data = pd.read_json(args.input_file, lines=True)
    if args.sample > 0:
        input_data = input_data.sample(n=args.sample)

    data = [
        {
            "theme": theme,     # テーマ
            "question": None,   # 問題文
            "answer": None,     # 正解
            "reference": None   # 出典
        } for theme in list(input_data["theme"])
    ]

    for d in data:
        time.sleep(args.interval)

        if args.verbose:
            print("theme:    ", d['theme'])

        # クイズ素材部分の抽出
        if args.from_wikipedia_content:
            content = get_wikipedia_content(d['theme'], content_len=6000)

            res = pickup_quiz_material(content,
                                       retry_max=args.retry_max, 
                                       interval=args.interval,
                                       model=args.material_model,
                                       debug=args.debug,
                                       temperature=args.material_temperature)
            if res is None:
                if args.verbose:
                    print('failed to pickup material')
                continue

#            material = res['content']
            material = res['maefuri'] + res['atogentei']
            material = material.replace("前振り:", '').replace("【前振り】", '').replace("後限定:", "").replace("【後限定】", "") # 姑息的

            d['reference'] = material
        else:
            material = d['theme']

        # クイズ生成
        res = generate_quiz(material,
                            retry_max=args.retry_max, 
                            interval=args.interval,
                            model=args.generation_model,
                            debug=args.debug,
                            temperature=args.generation_temperature)

        if res is None:
            if args.verbose:
                print('failed to generate quiz')
            continue
        
        d['question'] = res['question']
        d['answer'] = res['answer']
        if args.verbose:
            print("question:  ", d['question'])
            print("answer:    ", d['answer'])

        if args.review_quiz:
            # 評価＋修正
            res = review_quiz(d,
                            retry_max=args.retry_max,
                            interval=args.interval,
                            model=args.review_model,
                            debug=args.debug,
                            temperature=args.review_temperature)
            if res is None:
                if args.verbose:
                    print('failed to review quiz')
                continue

            d['question'] = res['question']
            d['answer'] = res['answer']
            if args.verbose:
                print("reviewd_question:  ", d['question'])
                print("reviewd_answer:    ", d['answer'])

    pd.DataFrame(data).to_json(args.output_file, orient='records', force_ascii=False, lines=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""
    OpenAI APIを用いたクイズ生成のサンプルコード。
    """)
    parser.add_argument("input_file",
                        type=str,
                        help="json lines形式で1行1テーマで書かれている評価データセット。"
                        )
    parser.add_argument("--output_file",
                        type=str,
                        default="output_data.jsonl",
                        help="OpenAIモデルの出力結果を格納するファイル。")
    parser.add_argument("--sample",
                        default=-1,
                        type=int,
                        help="モデルに与えるテーマ数。指定がない場合は全テーマに対して推論を行う。")
    parser.add_argument("--interval",
                        default=3,
                        type=int,
                        help="APIの最小呼び出し間隔。")
    parser.add_argument('--retry_max',
                        default=0,
                        type=int,
                        help="API呼び出しの再試行上限")
    parser.add_argument('--seed',
                        default=42,
                        type=int,
                        help="乱数シード")
    parser.add_argument('--verbose',
                       action='store_true',
                       help="途中経過の出力")
    parser.add_argument('--from_wikipedia_content',
                       action='store_true',
                       help="テーマに基づくWikipedia記事のみ利用")
    parser.add_argument('--review_quiz',
                       action='store_true',
                       help="生成したクイズを修正する")
    parser.add_argument("--generation_model",
                        default="gpt-3.5-turbo",
                        type=str,
                        help="クイズ生成のモデル"
                        )
    parser.add_argument("--review_model",
                        default="gpt-3.5-turbo",
                        type=str,
                        help="クイズ修正のモデル"
                        )
    parser.add_argument("--material_model",
                        default="gpt-3.5-turbo-16k",
                        type=str,
                        help="Wikipedia記事から素材抽出のモデル"
                        )
    parser.add_argument("--generation_temperature",
                        default=0.7,
                        type=float,
                        help="クイズ生成モデルの温度"
                        )
    parser.add_argument("--review_temperature",
                        default=0.7,
                        type=float,
                        help="クイズ修正モデルの温度"
                        )
    parser.add_argument("--material_temperature",
                        default=0.7,
                        type=float,
                        help="素材抽出モデルの温度"
                        )
    parser.add_argument('--debug',
                       action='store_true',
                       help="デバグ向け情報を出力する")
        
    args = parser.parse_args()

    main(args)