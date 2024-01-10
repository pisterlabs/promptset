import os
import argparse

# LangChain Data connection
from langchain.document_loaders import TextLoader           # LangChain Data connection の Document Loaders
from langchain.text_splitter import CharacterTextSplitter   # LangChain Data connection の Text Splitters
from langchain.embeddings.openai import OpenAIEmbeddings    # LangChain Data connection の Text embedding models
from langchain.vectorstores import Chroma                   # LangChain Data connection の VectorStores
from langchain.chains import RetrievalQA                    # LangChain Data connection の Retrievers
# LangChain Model I/O
from langchain.llms import OpenAI                           # LangChain Model I/O の Language models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_api_key', type=str, default="dummuy")
    parser.add_argument('--text_path', type=str, default="in_data/kenran.txt")
    parser.add_argument('--emb_model_name', type=str, default="text-embedding-ada-002")
    parser.add_argument('--model_name', type=str, default="text-davinci-003")
    parser.add_argument('--prompt', type=str, default="ヤガミについて教えてください")
    args = parser.parse_args()

    # OpenAI の API キー設定
    os.environ["OPENAI_API_KEY"] = args.openai_api_key

    # LangChain Data connection の Document Loaders を使用して、テキストデータ読み込み
    document_loader = TextLoader(args.text_path)
    documents = document_loader.load()

    # LangChain Data connection の Text Splitters を使用して、テキストを分割
    text_splitter = CharacterTextSplitter(
        # separator = "\n",   # セパレータ
        chunk_size=700,     # チャンクの文字数
        chunk_overlap=0
    )
    split_documents = text_splitter.split_documents(documents)
    print(f'split_documents={split_documents}')

    # LangChain Data connection の Text embedding models を使用して、埋め込みモデル定義
    # デフォルトでは、OpenAI 推奨の "text-embedding-ada-002" モデルが使用される
    emb_model = OpenAIEmbeddings(
        model=args.emb_model_name,
    )
    print(f'emb_model={emb_model}')

    emb_result_0 = emb_model.embed_query(split_documents[0].page_content)
    print(f'emb_result_0[0:10]={emb_result_0[0:10]}')

    # 埋め込みモデルで分割テキストを埋め込み、埋め込みベクトルを作成し、特徴量データベース（VectorDB）に保存
    # LangChain Data connection の VectorStores を使用
    feature_db = Chroma.from_documents(split_documents, emb_model)
    print(f'feature_db={feature_db}')

    # 特徴量データベース（VectorDB）から retriever（ユーザーからの入力文に対して、外部テキストデータの分割した各文章から類似度の高い文章を検索＆取得をするための機能）作成
    retriever = feature_db.as_retriever(
        search_kwargs={"k": 1},     # k=1 個の分割文章を検索＆取得
    )
    print(f'retriever={retriever}')

    # retriever で、ユーザーからの入力文に対して、外部テキストデータの分割した各文章から類似度の高い情報を検索＆取得
    print(f"retriever.get_relevant_documents(query=args.prompt)={retriever.get_relevant_documents(query=args.prompt)}")

    # LLM モデル定義
    llm = OpenAI(
        model_name=args.model_name,
        temperature=0.9,                # 大きい値では出現確率が均一化され、より多様な文章が生成される傾向がある。低い値では出現確率の高い単語が優先され、より一定の傾向を持った文章が生成される傾向がある。
    )
    print(f'llm={llm}')

    # LangChain Data connection の Retrievers を使用して、RetrievalQA Chain（質問応答 QA の取得に使用する Chain）を生成
    # Chain : 複数のプロンプト入力を実行するための機能
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever
    )
    print(f'qa_chain={qa_chain}')

    # LLM 推論実行（QA:質問応答）
    try:
        answer = qa_chain.run(args.prompt)
        print(f'answer={answer}')
    except Exception as e:
        print(f"Excception was occurred | {e}")
        exit(1)

    exit(0)
