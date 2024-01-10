import streamlit as st
import streamlit_ace as st_ace
from openai import OpenAI
import json


TEST = False


class Code_chatbot():
    def __init__(self, code_language, openai_api_key):
        self.code_language = code_language
        if not TEST:
            self.client = OpenAI(
                api_key=openai_api_key
            )
        self.code = ''
        self.overview = {
            'explanation': '',
            'improvement': ''}

    def submit_code(self, code):
        prompt = f'''我將會給你一段{self.code_language}程式碼，請你解釋這段程式在做什麼，以及可以怎麼改善它。
【以下為程式碼】
{code}
【以上為程式碼】

條件限制：
【程式碼說明】: 解釋程式的流程和用途，使用繁體中文 
【程式改進建議】:  說明程式寫不好的地方，提供相關改進建議，使用繁體中文

回傳格式：
以json格式回傳，如{{"explanation":程式碼說明,"improvement":程式改進建議}}'''

        if TEST:
            return

        completion = self.client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'user', 'content': prompt}
            ]
        )
        print(completion) 
        message = json.loads(completion.choices[0].message.content)  
        print(message)

        self.code = code
        self.overview['explanation'] = message['explanation']
        self.overview['improvement'] = message['improvement']

    def get_explanation(self):
        if TEST:
            return '這段程式碼是用來計算斐波那契數列的第 n 個數字，其中 n 由使用者輸入。程式初始化兩個變數 a 和 b 為 1，然後使用迴圈計算並更新這兩個變數，最終輸出斐波那契數列的第 n 個數字。'
        return self.overview['explanation']

    def get_improvement(self):
        if TEST:
            return '程式碼整體看起來已經相當簡潔，但還是有一些改進的空間。首先，可以加入一些輸入驗證，確保使用者輸入的是正整數。另外，可以考慮將程式包裝成一個函式，以便更容易重複使用。最後，可以加上註釋來解釋程式的目的和運作方式，提高代碼的可讀性。'
        return self.overview['improvement']


def main():
    st.title('Code Chatbot')

    code_languages = st_ace.LANGUAGES
    code_language = st.selectbox(
        'language',
        code_languages,
        index=code_languages.index('python'))

    code = st_ace.st_ace(
        placeholder='type your code here',
        language=code_language)

    if code == '':
        return

    with open('api_key') as f:
        code_chatbot = Code_chatbot(code_language, f.read().strip())
    code_chatbot.submit_code(code)

    st.header('Code Explaination')
    st.write(code_chatbot.get_explanation())

    st.header('How to Improve')
    st.write(code_chatbot.get_improvement())

if __name__ == '__main__':
    main()

