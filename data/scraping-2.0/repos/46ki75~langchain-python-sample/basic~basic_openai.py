from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

# .envファイルから環境変数をロードする
load_dotenv()

# --------------------------------------------------
# OpenAI ChatGPT を使用した簡単な生成
# --------------------------------------------------

# OpenAI インスタンスを作成
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = OpenAI(api_key=OPENAI_API_KEY)

print(
    llm.predict("Where is the capital of Japan?")
)
