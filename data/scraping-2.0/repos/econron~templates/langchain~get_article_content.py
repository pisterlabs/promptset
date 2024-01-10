from bs4 import BeautifulSoup

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from langchain.prompts import ChatPromptTemplate

# Send HTTP request and get the page content
# dnaはクローリング非許可
with open('./dna.html', 'r') as file:
    # Read the content of the file
    html_content = file.read()
soup = BeautifulSoup(html_content, 'html.parser')

word_list = soup.find('ol', {'class': 'unlocking_word_meanings_list'})

# 単語・発音・意味の取得

for li in word_list.find_all('li'):
    em_text = li.find('em').text
    b_text = li.find('b').text
    p_text = li.find('p').text
    print(f'Em Text: {em_text}, B Text: {b_text}, P Text: {p_text}')

# 取得した単語データをベースに問題を作成する
# テーブルに単語データを突っ込む
# 単語データを取得してchat/completionに載せる
# TODO：プロンプトを改良する
# レスポンスデータをテーブルに入れる

prompt = PromptTemplate(
    input_variables=["word1", "word2", "word3"],
    template="You are a helpful AI bot. You create 3 sentences by given words and their meamings. Use these words. {word1}, {word2}, {word3}",
)

llm = ChatOpenAI(temperature=0.0)

chain = LLMChain(llm=llm, prompt=prompt)

result = chain.run({
    "word1": "halt", 
    "word2": "something", 
    "word3": "Gaussian"
})

print(result)


# 取得した文章をテーブルに入れる
# 文ごとにchat/completionに投げて翻訳し、対訳をテーブルに格納する

prompt = PromptTemplate(
    input_variables=["sentence"],
    template="You are a helpful AI bot. Translate the following sentences into Japanese. {sentence}",
)

chain = LLMChain(llm=llm, prompt=prompt)

# テキストの取得
content = soup.find('div', {'class': 'article'})

for p in content.find_all('p'):
    print(f'{p.text}')
    result = chain.run({
        "sentence": f'{p.text}'
    })
    print(result)