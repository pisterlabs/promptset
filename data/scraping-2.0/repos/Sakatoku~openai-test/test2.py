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
    organization = os.environ.get('OPENAI_ORGANIZATION_ID')
    api_key = os.environ.get('OPENAI_API_KEY')
    return organization, api_key

# 初期化
def init_openai():
    # OrganizationとAPI keyを.envから読み込む
    organization, api_key = get_openai_vars()
    openai.organization = organization
    openai.api_key = api_key

# 画像生成API(openai.Image.create)をテストする
def test_imagegen():
    # 関数名を出力して横線で区切る
    print('test_imagegen -------------------------')
    # プロンプトをハードコーディングで定義する。なんとなく英語の方がうまくいく気がする？
    prompt_en = 'A portrait of a girl and a cat, by japanese traditional ukiyoe style in age Edo (1750s)'
    # prompt_jp = '少女と猫が描かれたポートレート。江戸時代中期の浮世絵風で'
    selected_prompt = prompt_en
    # 画像生成API(openai.Image.create)を呼び出す。枚数はn、画像サイズはsizeで指定する
    if not SHOW_IMAGE:
        response = openai.Image.create(prompt=selected_prompt, n=1, size='512x512')
        # デバッグ用にすべての情報を出力する
        print('response: {}'.format(response))
    else:
        # 画像を表示する場合はb64_json(Base64エンコード)で画像を取得する
        response = openai.Image.create(prompt=selected_prompt, n=3, size='512x512', response_format='b64_json')
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
    # 画像生成API(openai.Image.create)をテストする
    test_imagegen()

# メイン関数を実行する
if __name__ == '__main__':
    main()