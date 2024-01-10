import os
import sys
import logging
import openai
from pathlib import Path

def main():
    ENDPOINT_URL = os.environ.get("ENDPOINT_URL")
    API_KEY      = os.environ.get("API_KEY")

    path   = sys.argv[1]
    code = Path(sys.argv[2]).read_text()

    try:
        openai.api_type    = "azure"
        openai.api_version = "2023-05-15"
        openai.api_base    = ENDPOINT_URL
        openai.api_key     = API_KEY

        aiRole = '''
あなたはコードレビューを行うAIアシスタントです。

入力は次の形式で与えられます。
- "# ファイルパス"の後に続くコードブロックにコードのファイルパスが与えられます。
- "# 入力"の後に続くコードブロックに変更前のコードが与えられます。

入力されたコードを調べ、要約してください。
その際、以下のルールに従ってください。
- "# フォーマット"の後に続くコードブロック内のフォーマットに従いMarkdown形式で出力してください。
- "ファイルパス"には入力で受け取ったコードのファイルパスを記載してください。
- 必ず若者言葉でフレンドリーな口調で返答してください。

# フォーマット
```
## AIコードレビュー
ファイルパス
### 概要
どのようなコードなのかをリスト形式で具体的に記載してください。
### 改善案
コードを改良する案がある場合はリスト形式で具体的に記載してください。
改良案がない場合はリスト形式で褒めてください。
```
'''

        aiPrompt = f'''
# ファイルパス
{path}
# 入力
```
{code}
```
'''

        response = openai.ChatCompletion.create(
            deployment_id="github-app-test",
            messages=[
                {"role": "system", "content": aiRole},
                {"role": "user", "content": aiPrompt}
            ]
        )

        print(response.choices[0]["message"]["content"].strip())

    except Exception as e:
        logging.error(e)

if __name__ == "__main__":
    main()