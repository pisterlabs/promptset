from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter

class DocumentPreprocessor:
    """
    HTMLを受け取り、LangChainへ引き渡すドキュメントに変換します
    """
    def __init__(self) -> None:
        pass

    def preprocess(self, html,  url, chunk_size=1000):
        text = self.html_to_text(html)
        docs = self.text_to_documents(text=text, url=url, chunk_size=chunk_size)
        return docs

    def html_to_text(self, html):
        """
        HTMLをテキストに変換します
        """
        soup=BeautifulSoup(html,"html.parser")
        text=soup.get_text('\n')
        lines= [line.strip() for line in text.splitlines()]
        return "\n".join(lines)

    def text_to_documents(self, text, url, chunk_size):
        """
        ドキュメントをいくつかのチャンクに分割します。
        """
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=0)
        docs = text_splitter.create_documents([text], metadatas=[{"source": url}])
        return docs