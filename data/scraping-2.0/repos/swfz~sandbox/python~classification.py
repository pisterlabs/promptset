import openai
import sys

markdown_file = sys.argv[1]

def read_content(file):
    with open(file, 'r', encoding="utf-8") as f:
        return f.read()

def classify(markdown_file):
    mddata = read_content(markdown_file)

    system_content = """
与えられた記事を見て、どのような内容の記事なのか教えてください

また下記リストの内記事の内容が近いのはどの項目ですか？
- 書評
- ライブラリ、パッケージ作りました
- Tips
- 開発組織のあれこれ
- ポエム
- release note

また、扱っている内容の難易度はどのくらいですか？
"""
    res = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": mddata},
            ]
        )

    print(res.choices[0]["message"]["content"].strip())

classify(markdown_file)
