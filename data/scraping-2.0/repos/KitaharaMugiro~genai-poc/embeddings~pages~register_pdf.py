import streamlit as st
import requests
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter


def text_split_from_page(page):
    text_splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=300,
        chunk_overlap=50,
        add_start_index=True,
    )
    texts = text_splitter.create_documents([page.page_content])
    return texts


def embed_text_with_openai(api_key, text, groupName="default"):
    url = "http://langcore.org/api/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"input": text, "groupName": groupName}

    response = requests.post(url, json=data, headers=headers)

    if response.status_code != 200:
        st.error(f"Error {response.status_code}: {response.text}")
        return None

    return response.json()


st.title("Langcore PDF登録画面")

# APIキーの入力
api_key = st.text_input("Enter your OpenAI API Key:", type="password")

# グループ名の入力 (オプショナル)
group_name = st.text_input("Enter a group name:")

# PDFファイルのアップロード
pdf_file = st.file_uploader("Upload a csv file", type="pdf")

# 一時的に保存
pdf_file_path = "temp.pdf"
if pdf_file:
    with open(pdf_file_path, "wb") as f:
        f.write(pdf_file.read())

# ボタンを押したらEmbeddings処理を行う
if st.button("Register Embeddings"):
    if api_key and pdf_file:
        loader = PyPDFLoader(pdf_file_path)
        pages = loader.load_and_split()
        lines = []
        for page in pages:
            lines += text_split_from_page(page)

        # テキスト部分だけ抽出
        lines = [line.page_content for line in lines if line.page_content]

        # 30文字以上の行だけ抽出
        lines = [line for line in lines if len(line) > 30]

        embedded_lines = []
        with st.spinner("Embedding lines..."):
            progress_bar = st.progress(0)
            for index, line in enumerate(lines, 1):
                # Embeddingの処理
                embedded_line = embed_text_with_openai(api_key, line, group_name)
                if embedded_line is not None:
                    embedded_lines.append(embedded_line)

                # 進行度の表示
                progress_bar.progress(index / len(lines))

        st.write("Embeddings completed!")
    else:
        st.warning("Please input API key and text.")
