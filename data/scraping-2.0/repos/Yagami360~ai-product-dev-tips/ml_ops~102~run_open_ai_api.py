import argparse
import openai

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_api_key', type=str, default="dummuy")
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--content', type=str, default="今日は天気が良いですね")
    args = parser.parse_args()

    # APIキーの設定
    openai.api_key = args.openai_api_key

    # 使用可能なモデルリスト
    print("available models: ", openai.Model.list())

    # OpenAI API 呼び出し
    try:
        response = openai.ChatCompletion.create(
            model=args.model,
            messages=[
                {"role": "user", "content": args.content},
            ],
            temperature=0.7,        # number or null Optional Defaults to 1 / 大きい値では出現確率が均一化され、より多様な文章が生成される傾向がある。低い値では出現確率の高い単語が優先され、より一定の傾向を持った文章が生成される傾向がある。
            top_p=1,                # number or null Optional Defaults to 1 / An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or temperature but not both.
            n=1,                    # integer Optional Defaults to 1 / 回答の数。3を指定すれば3つの回答を得られる。
            stream=False,           # boolean or null Optional Defaults to falseIf set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only server-sent events as they become available, with the stream terminated by a data: [DONE] message. Example Python code.
            stop=None,              # string / array / null Optional Defaults to null / トークンの生成を停止する文字列
            max_tokens=100,         # integer Optional Defaults to inf / 生成されるレスポンスのトークンの最大数
            presence_penalty=0,     #
            frequency_penalty=0,    # 2.0 から 2.0 の間の数値を指定。値が低い場合、生成された文章に既に含まれている単語やフレーズが強調されすぎて、文章の多様性が低下する可能性がある。値が髙い場合、生成された文章が同じ単語やフレーズを繰り返すことが少なくなり、より多様な文章を生成することができる。
            # logit_bias={96096:20},  # {トークンID: value} で指定 / トークンの生成確率を調整するために、各トークンに対してlogit_biasを設定することができる。正の値を持つトークンは出現確率が上がり、負の値を持つトークンは出現確率が下がる。
        )
        print(f"response: {response}")
        print(f"content: {response['choices'][0]['message']['content'].strip()}")
    except Exception as e:
        print(f"Excception was occurred | {e}")
        exit(1)

    exit(0)
