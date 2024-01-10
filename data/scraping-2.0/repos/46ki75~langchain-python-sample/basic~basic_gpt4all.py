from langchain.llms import GPT4All

# --------------------------------------------------
# GPT4ALL LLM を使用した簡単な生成
# --------------------------------------------------

# GPT4All モデルのパスを指定
model_path = "/mnt/c/Users/46ki7/AppData/Local/nomic.ai/GPT4All/ggml-model-gpt4all-falcon-q4_0.bin"

# GPT4All インスタンスを作成
llm = GPT4All(model=model_path)

print(
    llm.predict("Where is the capital of Japan?")
)
