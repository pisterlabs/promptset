import openai

# 生成した画像をOpenCVで表示するかどうか
# 依存ライブラリ: opencv-python
# 画像を表示する場合はTrueにする
SHOW_IMAGE = True
if SHOW_IMAGE:
    import cv2
    import numpy as np
    import base64

# OrganizationとAPI keyを.envから読み込む関数
# 依存ライブラリ: python-dotenv
# .envファイルの中身:
# OPENAI_ORGANIZATION = org-xxxxxxxxxxxxxxxxxxxxxxxx
# OPENAI_API_KEY = sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def get_openai_vars():
    # .envファイルから環境変数を読み込む
    import os
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path)
    # 環境変数から値を取得する
    organization = os.environ.get("OPENAI_ORGANIZATION_ID")
    api_key = os.environ.get("OPENAI_API_KEY")
    return organization, api_key

# 初期化
def init_openai():
    # OrganizationとAPI keyを.envから読み込む
    organization, api_key = get_openai_vars()
    openai.organization = organization
    openai.api_key = api_key

# 質問に対する回答を画像で返すために、画像生成APIに渡すプロンプトを生成する
def generate_prompt(question):
    # 関数名を出力して横線で区切る
    print('generate_prompt ------------------------------')
    # 日本語で回答を得て、次にそれを画像生成AIのプロンプトにするために英語に翻訳する
    messages = [
        {"role": "system", "content": "あなたの回答は必ずもっとも有力だと思われる1つの候補のみに絞ってください。回答は1文で完結させてください。"},
        {"role": "user", "content": question}
    ]
    # チャットAPIを呼び出して回答を生成する
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    # 回答を取得する
    answer_jp = response["choices"][0]["message"]["content"].strip()
    print('answer_jp: ' + answer_jp)
    # 回答を画像生成AIのプロンプトにする
    # 強調する指示を追加する
    strengthen_prompt = "。回答は英語でお願いします。英語です。英語英語英語英語マジで英語！絶対英語！"
    messages = [
        {"role": "system", "content": "次の入力を画像生成AIで描くための英語のプロンプトを作成してください。明るいタッチで写実的に描くことが望ましいです。"},
        {"role": "user", "content": answer_jp + strengthen_prompt}
    ]
    # チャットAPIを呼び出してプロンプトを生成する
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    # プロンプトを取得する
    generate_prompt = response["choices"][0]["message"]["content"].strip()
    print('generate_prompt: ' + generate_prompt)
    return generate_prompt

# 画像生成する
def imagegen(prompt):
    # 関数名を出力して横線で区切る
    print('imagegen ------------------------------')
    # 入力されたプロンプトを出力する
    print('prompt: ' + prompt)
    # 画像生成API(openai.Image.create)を呼び出す。枚数はn、画像サイズはsizeで指定する
    if not SHOW_IMAGE:
        response = openai.Image.create(prompt=prompt, n=1, size='512x512')
        # デバッグ用にすべての情報を出力する
        print('response: {}'.format(response))
    else:
        # 画像を表示する場合はb64_json(Base64エンコード)で画像を取得する
        response = openai.Image.create(prompt=prompt, n=3, size='512x512', response_format='b64_json')
        # 各画像を順番に処理する
        for idx, data in enumerate(response['data']):
          # デコードして画像を表示する
          decoded_data = base64.b64decode(data['b64_json'])
          buffer = np.frombuffer(decoded_data, np.uint8)
          img = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
          # 画像を表示する。タイムアウトは10,000ms(10秒)
          print('image{}: {}...{}'.format(idx+1, data['b64_json'][:40], data['b64_json'][-40:]))
          cv2.imshow(f'image{idx+1}', img)
          cv2.waitKey(10000)

# メイン関数
def main():
    # 初期化
    init_openai()
    # 質問文
    question = '深夜に食べると最高に美味しい食べ物は何ですか？'
    # 質問に対する回答を画像で返すために、画像生成APIに渡すプロンプトを生成する
    prompt = generate_prompt(question)
    # 画像生成する
    imagegen(prompt)

# メイン関数を実行する
if __name__ == '__main__':
    main()
