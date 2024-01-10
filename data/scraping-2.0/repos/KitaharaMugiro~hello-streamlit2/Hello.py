import openai
import streamlit as st

openai.api_base = "https://oai.langcore.org/v1"

def main():
    st.title('ChatGPT キャッチコピー作成アプリ')
    user_input = st.text_input('キャッチコピーを作成するためのキーワードを入力してください: ')
    
    if st.button('キャッチコピーを生成'):
        catchphrase = call_chatgpt_api(user_input)
        st.write('生成されたキャッチコピー: ', catchphrase)

def call_chatgpt_api(input_text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "キャッチコピーを考えてください"},
                      {"role": "user", "content": input_text}]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f'エラー: キャッチコピーを生成できませんでした。{str(e)}'

if __name__ == '__main__':
    main()
