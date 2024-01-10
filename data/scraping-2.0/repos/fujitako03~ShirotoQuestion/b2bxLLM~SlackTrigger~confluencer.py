from urllib.parse import urlparse

from atlassian import Confluence
from langchain.document_loaders import ConfluenceLoader

class Confluencer:
    """
    ConfluenceのURLから必要な情報を取得するクラス

    Attributes:
        url (str): ConfluenceのURL
    """
    def __init__(self, url):
        self.url = url

    @property
    def base_url(self):
        """
        ConfluenceのURLからベースURLを取得する"""
        parsed_url = urlparse(self.url)
        return parsed_url.scheme + "://" + parsed_url.netloc

    @property
    def space_id(self):
        """
        ConfluenceのURLからスペースIDを取得する"""
        return self.url.split('/')[5]
    
    @property
    def page_id(self):
        """
        ConfluenceのURLからページIDを取得する"""
        return self.url.split('/')[7]
    
    def get_documents(self, user_name, api_key):
        """
        ConfluenceのURLからページIDを取得する
        
        Args:
            user_name (str): Confluenceのユーザー名
            api_key (str): ConfluenceのAPIキー
        """
        loader = ConfluenceLoader(
            url=self.base_url,
            username=user_name,
            api_key=api_key
        )
        documents = loader.load(
            page_ids=[self.page_id],
            include_attachments=False,
            limit=10)
        return documents

    def get_html(self, user_name, api_key):
        """
        ConfluenceのURLからページIDを取得する
        
        Args:
            user_name (str): Confluenceのユーザー名
            api_key (str): ConfluenceのAPIキー
        """
        confluence = Confluence(
            url=self.base_url,
            username=user_name,
            password=api_key
        )
        page_info = confluence.get_page_by_id(page_id=self.page_id, expand='body.storage')
        
        return page_info["body"]["storage"]["value"]
