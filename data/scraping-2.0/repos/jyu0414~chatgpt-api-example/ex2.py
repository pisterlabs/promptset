import openai

# API Keyを設定
openai.api_key = ""

chat_completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "入力された文章のタイトルをつける"},
        {"role": "user", "content": "ChatGPTを自然言語処理エンジンとして見ると，自由記述解答など従来コンピュータで処理が難しかったデータを簡単に扱える可能性が拡がる．しかし，通常のチャット形式での利用時には，応答が毎回異なるため一貫性を持たせたプログラミングが困難である．そこで，本講演ではプログラムからChatGPTを利用するAPIと，それをデータ分析やソフトウェアのバックグラウンドシステムにどのように組み込むかについて具体的な実装例とともに紹介する．"}
        ]
    )

print(chat_completion["choices"][0]["message"]["content"])