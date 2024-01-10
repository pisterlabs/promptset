from langchain.prompts import ChatPromptTemplate
from langchain.llms import GPT4All

# --------------------------------------------------
# LangChain Expression Language (LCEL)
# 基礎的な使い方
# --------------------------------------------------

# GPT4All モデルのパスを指定
model_path = "/mnt/c/Users/46ki7/AppData/Local/nomic.ai/GPT4All/ggml-model-gpt4all-falcon-q4_0.bin"

# GPT4All インスタンスを作成
llm = GPT4All(model=model_path)

# プロンプト
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

# チェーンを作成
chain = prompt | llm

# チェーンの実行
print(
    chain.invoke({"topic": "bears"})
)

# チェーンの実行 (バッチ処理)
# max_concurrency: 同時処理数
print(
    chain.batch(
        [{"topic": "bears"}, {"topic": "cats"}],
        config={"max_concurrency": 5}
    )
)

"""
ストリーム (GPT4ALL は対応していないので、通常と変わらない。)
for s in chain.stream({"topic": "bears"}):
    print(s, end="", flush=True)
"""