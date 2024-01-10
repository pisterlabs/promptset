from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate

# --------------------------------------------------
# Prompt templatesの使い方
# --------------------------------------------------

# テンプレートを作成
prompt = PromptTemplate.from_template("Where is the capital of {country}?")

# 値をバインド
formated_prompt = prompt.format(country="America")

print(formated_prompt)

# --------------------------------------------------
# Chat Prompt templatesの使い方
# --------------------------------------------------

# テンプレート作成用の文字列を初期化
template = "You are a helpful assistant that translates {input_language} to {output_language}."
human_template = "{text}"

# チャットテンプレートを作成
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

# 値をバインド
caht_formated_prompt = chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")

print(caht_formated_prompt)

"""
出力は以下のようにインスタンスオブジェクトの配列になる
[
    SystemMessage(content='You are a helpful assistant that translates English to French.'),
    HumanMessage(content='I love programming.')
]
"""