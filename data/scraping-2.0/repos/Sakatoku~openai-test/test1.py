import openai

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

# Modelをすべて出力
# アカウントの問題でgpt-4はまだ有効になっていない(2023/8/17時点)
def list_models():
    # 関数名を出力して横線で区切る
    print('list_models ----------------------------')
    # Modelを取得する。organizationの設定が必要
    l = openai.Model.list()
    # ModelのIDのみを出力する
    for idx, item in enumerate(l['data']):
            print('{}: {}'.format(idx, item['id'])) 

# チャットAPI(openai.ChatCompletion.create)をテストする
def test_chat():
    # 関数名を出力して横線で区切る
    print('test_chat ------------------------------')
    # 会話履歴を手動で定義する
    test_messages = [
            {'role': 'system', 'content': 'あなたは平成一桁年代のJ-POPのマニアとしてふるまってください。'},
            {'role': 'user', 'content': '1990年のヒット曲はなんですか？'},
            {'role': 'assistant', 'content': '米米CLUBの「浪漫飛行」です。'},
            {'role': 'user', 'content': '同じ時期の米米CLUBの他の代表曲を3つ挙げてください。'} # この行が最新の質問
        ]
    # チャットAPI(openai.ChatCompletion.create)を呼び出す
    response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=test_messages)
    # 回答本文だけ抜粋して出力する
    content = response['choices'][0]['message']['content'].strip()
    print('回答: {}'.format(content))
    # デバッグ用にすべての情報を出力する
    print('response: {}'.format(response))

# メイン関数
def main():
    # 初期化
    init_openai()
    # Modelをすべて出力
    list_models()
    # 改行
    print('')
    # チャットAPI(openai.ChatCompletion.create)をテストする
    test_chat()

# メイン関数を実行する
if __name__ == '__main__':
    main()