import nest_asyncio
from langchain.docstore.document import Document
from langchain.document_loaders import CSVLoader  # UnstructuredPDFLoader,
from langchain.document_loaders import (DirectoryLoader, GitbookLoader,
                                        PyPDFLoader, ReadTheDocsLoader,
                                        TextLoader, UnstructuredHTMLLoader,
                                        UnstructuredMarkdownLoader,
                                        UnstructuredURLLoader, WebBaseLoader,
                                        YoutubeLoader)
from langchain.document_loaders.sitemap import SitemapLoader

from src.services.logging_service import logger
from src.utils import get_links

nest_asyncio.apply()


class DocumentLoaderFactory():
    @staticmethod
    def create_loader(loader_type, body):
        if loader_type == 'gitbook':
            urls = body.get('urls', [])
            logger.info(f'[DocumentLoaderFactory.create_loader] Gitbook Link: ' + urls[0])
            loader = GitbookLoader(urls[0], load_all_paths=True)
            loader.default_parser = "xml"
            return loader
        
        elif loader_type == 'web_base':
            urls = body.get('urls', [])
            logger.info(f'[DocumentLoaderFactory.create_loader] Web Base: ' + str(urls))
            loader = WebBaseLoader(urls)
            loader.default_parser = "xml"
            return loader
        
        elif loader_type == 'yt':
            raise Exception("Non Implemented Yet, having issues uploading a directory")
            yt_id = body.get('ytId')
            logger.info(f'[DocumentLoaderFactory.create_loader] Youtube: https://youtube.com/watch?v=' + yt_id)
            return YoutubeLoader(
                yt_id, 
                # add_video_info=True
            )
            
        elif loader_type == 'sitemap':
            urls = body.get('urls', [])
            logger.info(f'[DocumentLoaderFactory.create_loader] Sitemap: ' + str(urls))
            return SitemapLoader(web_path=urls[0])
        
        elif loader_type == 'website':
            urls = body.get('urls', [])
            unique_links = get_links(urls[0])
            logger.info(f'[DocumentLoaderFactory.create_loader] Website: ' + str(unique_links))
            return UnstructuredURLLoader(urls=unique_links)
        
        elif loader_type == 'urls':
            urls = body.get('urls', [])
            logger.info(f'[DocumentLoaderFactory.create_loader] URLs: ' + str(urls))
            return UnstructuredURLLoader(urls=urls)
        
        elif loader_type == 'copy':
            raise Exception("Non Implemented Yet, having issues uploading a directory")
            logger.info(f'[DocumentLoaderFactory.create_loader] Copy: ')
            # metadata = body['metadata'] if body['metadata'] else None
            return Document(page_content=body.get('text'))
        
        elif loader_type == 'txt':
            logger.info(f'[DocumentLoaderFactory.create_loader] Text: ' + body.get('file_path'))
            return TextLoader(body.get('file_path'))
        
        elif loader_type == 'html':
            logger.info(f'[DocumentLoaderFactory.create_loader] HTML: ' + body.get('file_path'))
            return UnstructuredHTMLLoader(body.get('file_path'))
        
        elif loader_type == 'md':
            logger.info(f'[DocumentLoaderFactory.create_loader] Markdown: ' + body.get('file_path'))
            loader = UnstructuredMarkdownLoader(
                body.get('file_path'), 
                # mode="elements"
            )
            logger.info(loader)
            return loader
        
        elif loader_type == 'directory':
            raise Exception("Non Implemented Yet, having issues uploading a directory")
            logger.info(f'[DocumentLoaderFactory.create_loader] Directory: ' + body.get('file_path'))
            return DirectoryLoader(body.get('file_path'), glob="**/*")
        
        elif loader_type == 'csv':
            logger.info(f'[DocumentLoaderFactory.create_loader] CSV: ' + body.get('file_path'))
            loader = CSVLoader(body.get('file_path'))
            return loader
        
        elif loader_type == 'pdf':
            logger.info(f'[DocumentLoaderFactory.create_loader] PDF: ' + body.get('file_path'))
            loader = PyPDFLoader(body.get('file_path'))
            return loader
        
        else:
            raise ValueError('Unsupported document loader type: ' + loader_type)