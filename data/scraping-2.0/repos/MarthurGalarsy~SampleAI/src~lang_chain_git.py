import os

from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import GitLoader

load_dotenv()

clone_url = input("GitHubのURLを入力してください：")
type = input("プログラムの種類を入力してください（ex：.kt）：")
repo_path = "./temp"
branch = input("ブランチを入力してください：")

if os.path.exists(repo_path):
    clone_url = None

loader = GitLoader(
    clone_url=clone_url,
    branch=branch,
    repo_path=repo_path,
    file_filter=lambda file_path: file_path.endswith(type),
)

index = VectorstoreIndexCreator(
    vectorstore_cls=Chroma, # default
    embedding=OpenAIEmbeddings(disallowed_special=()), #default
).from_loaders([loader])

while True:
    # ユーザーからの入力を受け付ける
    user_input = input("質問を入力してください (終了するには 'exit' と入力してください)：")
    
    # 入力が 'exit' の場合、ループを終了
    if user_input.lower() == "exit":
        break
    if user_input == "":
        break
   
    response = index.query(user_input)
    print("回答：", response)
