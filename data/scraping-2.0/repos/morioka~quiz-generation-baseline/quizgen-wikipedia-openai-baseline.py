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
以下のルールを守ってください。

・以下に示すテーマに基づいて、問題文と正解からなる早押しクイズを作ってください。
・早押しクイズですので、問題文の前半の「前振り」、問題文の後半の「後限定」、そして「正解」に分けて作ってください
・「前振り」は、正解を説明する修飾師です。できる限り、聞いてためになる情報を盛り込んでください。
・「後限定」は、文末は「でしょう？」で終わるようにしてください。ただし、「誰でしょう？」「何でしょう？」だけでなく、皆が知っているような、正解を確実に導き出せる確実な情報を入れて下さい。
・例を示します。
前振り：小説『白鯨』に登場する捕鯨船の航海士に因んで名付けられた、
後限定：シアトルに本拠地を置く世界的なコーヒーチェーンは何でしょう？
・「前振り」と「後限定」をつないで問題文を作ってください。自然な文章になるようにして下さい。
・正解とテーマが同じになることは避けてください。
・作った問題文と正解を関数 generate_quiz に与えてください。
"""

QG_USER_PROMPT = """テーマ:{theme}
"""

QG_REVIEW_SYSTEM_PROMPT = """あなたはプロのクイズ作家です。早押しクイズをよりよいものに修正できます。
以下のルールを守ってください。

・以下に示す、問題文、正解の組は、一般的なクイズとして適切であるならばそのまま出力してください。適切でないならば、適切なものになるよう。問題文または正解を修正してください。
・以下に示す、問題文、正解の組は、テーマの記述に沿っているならばそのまま出力してください。テーマの記述に含まれない内容ならば、テーマの記述のみに含まれるよう問題文または正解を修正してください。
・修正した問題文と正解を関数 review_quiz に与えてください。
"""
# ・「問題の答え」は、正解の他に、外れ選択肢を３つ作ってください。正解がどれかも示して下さい。

QG_REVIEW_USER_PROMPT = """テーマ:{theme}
問題文:{question}
正解:{answer}
"""

def generate_quiz(theme, 
                  retry_max=0, 
                  interval=1, 
                  model="gpt-3.5-turbo", 
                  debug=False):

    def generate(theme):
        try:
            messages=[
                {"role": "system", "content": QG_SYSTEM_PROMPT},
                {"role": "user", "content": QG_USER_PROMPT.format(theme=theme)}
            ]

            functions=[
                {
                    "name": "generate_quiz",
                    "description": "クイズを生成してjson形式で返す",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string", "description": "問題文"
                            },
                            "answer": {
                                "type": "string", "description": "正解"
                            },
#                            "distractors": {
#                                "type": "array", "description": "不正解のリスト",
#                                "items": {
#                                    "type": "string", "description": "不正解"
#                                }
#                            }
#                        },
#                        "required": ["question","answer","distractors"],
                        },
                        "required": ["question","answer"],
                    },
                }
            ]

            if debug:
                print(messages)
            
            completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                functions=functions,
                function_call="auto",
            )

            if debug:
                print(completion)
            message = completion["choices"][0]["message"]
            try:
                return json.loads(message['function_call']['arguments'])
            except KeyError:
                # 前振り、後限定が個別に返されてしまう場合の対処
                # {
                #  "role": "assistant",
                #  "content": {"question": { "前振り": ..., "後限定": ....}, "answer": ...}
                # }
                try:
                    if message['content'] is not None:
                        res = json.loads(message['content'])
                        res['question'] = "".join(res['question'].values())
                        return res
                except:
                    pass
        except: #  ServiceUnavailableError:
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
                debug=False):

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

            functions=[
                {
                    "name": "review_quiz",
                    "description": "クイズを評価・修正してjson形式で返す",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "theme": {
                                "type": "string", "description": "テーマ"
                            },
                            "question": {
                                "type": "string", "description": "問題文"
                            },
                            "answer": {
                                "type": "string", "description": "正解"
                            },
                        },
                        "required": ["theme", "question","answer"],
                    },
                }
            ]

            if debug:
                print(messages)

            completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                functions=functions,
                function_call="auto",
            )

            if debug:
                print(completion)

            message = completion["choices"][0]["message"]
            try:
                return json.loads(message['function_call']['arguments'])
            except KeyError:
                # 前振り、後限定が個別に返されてしまう場合の対処
                # {
                #  "role": "assistant",
                #  "content": {"question": { "前振り": ..., "後限定": ....}, "answer": ...}
                # }
                try:
                    if message['content'] is not None:
                        res = json.loads(message['content'])
                        res['question'] = "".join(res['question'].values())
                        return res
                except:
                    pass
        except: #  ServiceUnavailableError:
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
・以下の文章から、早押しクイズの問題文の前半「前振り」としてふさわしい区間と、問題文の後半の「後限定」としてふさわしい区間をそれぞれ抽出してください。
・「前振り」は、答えを説明する修飾師です。できる限り、聞いてためになる情報を盛り込んでください。
・「後限定」は、皆が知っているような、答えを確実に導き出せる確実な情報を含んでください。
・1か所200文字程度で抜き出してください。
・それぞれ1箇所、合計2か所を、リストにして出力してください。

"""

QG_MATERIAL_USER_PROMPT = """文章: {content}
"""


def pickup_quiz_material(theme, 
                         retry_max=0, 
                         interval=1, 
                         model="gpt-3.5-turbo", 
                         max_tokens=1500, 
                         debug=False):

    def pickup(content):

        try:
            messages=[
                {"role": "system", "content": QG_MATERIAL_SYSTEM_PROMPT},
                {"role": "user", "content": QG_MATERIAL_USER_PROMPT.format(content=content)}
            ]

            functions=[
                {
                    "name": "review_quiz",
                    "description": "クイズを評価・修正してjson形式で返す",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "maefuri": {
                                "type": "string", 
                                "description": "前振り"
                            },
                            "atogentei": {
                                "type": "string", 
                                "description": "後限定"
                            }
                        },
                        "required": ["maefuri", "atogentei"],
                    },
                }
            ]

            if debug:
                print(messages)

            completion = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                functions=functions,
                max_tokens=max_tokens,
                function_call="auto",
            )

            if debug:
                print(completion)
            message = completion["choices"][0]["message"]
            try:
                return json.loads(message['function_call']['arguments'])
            except:
                res = None
                try:
                    res = eval(message['function_call']['arguments'])   # セキュリティ上の懸念?
                except:
                    pass

                if res is None:
                    res =  {
                        "maefuri": message['content'],
                        "atogentei": ""
                    }
                return res

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
                                       debug=args.debug)
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
                            debug=args.debug)
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
                            debug=args.debug)
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
    parser.add_argument("--genreration_temperature",
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