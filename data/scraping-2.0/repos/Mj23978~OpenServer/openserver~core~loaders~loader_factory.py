from pathlib import Path
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders import (AZLyricsLoader, BSHTMLLoader,
                                        ChatGPTLoader, CSVLoader,
                                        DirectoryLoader, GitbookLoader,
                                        GitLoader, HuggingFaceDatasetLoader,
                                        ImageCaptionLoader, IMSDbLoader,
                                        JSONLoader, ObsidianLoader,
                                        OnlinePDFLoader, PlaywrightURLLoader,
                                        PyPDFLoader, SitemapLoader, SRTLoader,
                                        TextLoader, UnstructuredEmailLoader,
                                        UnstructuredImageLoader,
                                        UnstructuredMarkdownLoader,
                                        UnstructuredWordDocumentLoader,
                                        WebBaseLoader, YoutubeLoader)
from langchain.document_loaders.figma import FigmaFileLoader
from langchain.text_splitter import CharacterTextSplitter


class LoadersFactory:
    @staticmethod
    def load_file(path: str) -> List[Document]:
        loader = TextLoader(path, encoding="utf-8")
        documents = loader.load()
        return documents

    @staticmethod
    def csv(path: str) -> List[Document]:
        loader = CSVLoader(file_path=path)
        documents = loader.load()
        return documents

    @staticmethod
    def directory(path: str, glob: str) -> List[Document]:
        text_loader_kwargs = {'autodetect_encoding': True}
        loader = DirectoryLoader(path, glob, loader_kwargs=text_loader_kwargs)
        documents = loader.load()
        return documents

    @staticmethod
    def html_bs4(path: str, glob: str) -> List[Document]:
        loader = BSHTMLLoader(path)
        documents = loader.load()
        return documents

    @staticmethod
    def json(path: str, schema: str) -> List[Document]:
        loader = JSONLoader(Path(path).read_text(), schema)
        documents = loader.load()
        return documents

    @staticmethod
    def markdown(path: str) -> List[Document]:
        loader = UnstructuredMarkdownLoader(path)
        documents = loader.load()
        return documents

    @staticmethod
    def image(path: str) -> List[Document]:
        loader = UnstructuredImageLoader(path)
        documents = loader.load()
        return documents

    @staticmethod
    def pdf(path: str) -> List[Document]:
        loader = PyPDFLoader(path)
        documents = loader.load_and_split()
        return documents

    @staticmethod
    def online_pdf(url: str) -> List[Document]:
        loader = OnlinePDFLoader(url)
        documents = loader.load()
        return documents

    @staticmethod
    def sitemap(url: str) -> List[Document]:
        loader = SitemapLoader(url)
        documents = loader.load()
        return documents

    @staticmethod
    def subtitle(file_path: str) -> List[Document]:
        loader = SRTLoader(file_path)
        documents = loader.load()
        return documents

    @staticmethod
    def email(file_path: str) -> List[Document]:
        loader = UnstructuredEmailLoader(file_path)
        documents = loader.load()
        return documents

    @staticmethod
    def word(file_path: str) -> List[Document]:
        loader = UnstructuredWordDocumentLoader(file_path)
        documents = loader.load()
        return documents

    @staticmethod
    def youtube(url: str) -> List[Document]:
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        documents = loader.load()
        return documents

    @staticmethod
    def playwrite(urls: List[str]) -> List[Document]:
        loader = PlaywrightURLLoader(urls=urls)
        documents = loader.load()
        return documents

    @staticmethod
    def web_base(urls: List[str]) -> List[Document]:
        loader = WebBaseLoader(urls)
        documents = loader.load()
        return documents

    @staticmethod
    def azlyrics(urls: List[str]) -> List[Document]:
        loader = AZLyricsLoader(urls)
        documents = loader.load()
        return documents

    @staticmethod
    def hugging_face(dataset_name: str = "imdb", page_content_column: str = "text") -> List[Document]:
        loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
        documents = loader.load()
        return documents

    @staticmethod
    def imsdb(path: str) -> List[Document]:
        loader = IMSDbLoader(path)
        documents = loader.load()
        return documents

    @staticmethod
    def chat_gpt(path: str) -> List[Document]:
        loader = ChatGPTLoader(path)
        documents = loader.load()
        return documents

    @staticmethod
    def figma(access_token: str, node_id: str, file_key: str) -> List[Document]:
        loader = FigmaFileLoader(access_token, node_id, file_key)
        documents = loader.load()
        return documents

    @staticmethod
    def gitbook(url: str) -> List[Document]:
        loader = GitbookLoader(url, load_all_paths=True)
        documents = loader.load()
        return documents

    @staticmethod
    def obsidian(url: str) -> List[Document]:
        loader = ObsidianLoader(url)
        documents = loader.load()
        return documents

    @staticmethod
    def git(clone_url: str, repo_path: str, branch: str = "master") -> List[Document]:
        loader = GitLoader(
            clone_url=clone_url,
            repo_path=repo_path,
            branch=branch
        )
        documents = loader.load()
        return documents

    @staticmethod
    def blip(image_urls: List[str]) -> List[Document]:
        loader = ImageCaptionLoader(image_urls)
        documents = loader.load()
        return documents

    @staticmethod
    def split_docs(documents: List[Document], **kwargs) -> List[Document]:
        text_splitter = CharacterTextSplitter(**kwargs)
        docs = text_splitter.split_documents(documents)
        return docs
