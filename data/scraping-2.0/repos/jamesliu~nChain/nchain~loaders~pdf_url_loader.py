import os
import shutil
import hashlib
from typing import Optional
import sqlite_utils
import datetime as dt
from nchain import user_dir

try:
    from langchain.document_loaders import OnlinePDFLoader, PyMuPDFLoader
except ImportError:
    raise ImportError(
        'PDF File requires extra dependencies. Install with `pip install --upgrade "embedchain[dataloaders]"`'
    ) from None
from .base_loader import BaseLoader
from nchain.utils.helpers import clean_string

class PdfURLLoader(BaseLoader):
    def __init__(self, db_path: str):
        """
        Initialize the ArxivLoader with a path to the SQLite database.

        :param db_path: Path to the SQLite database file.
        """
        self.db = sqlite_utils.Database(db_path)
        # Ensure the 'papers' table exists
        self.db["pdfs"].create({
            "doc_id": str,
            "total_pages": int,
            "title": str,
            "authors": str,
            "summary": str,
            "pdf_path": str,
            "source": str,
            "url": str,
            "updated": str
        }, pk="doc_id", if_not_exists=True)

    def load_data(self, url, download_dir: Optional[str] = None):
        """Load data from a PDF file."""
        loader = PyMuPDFLoader(url)
        #loader = OnlinePDFLoader(url)
        file_path = loader.file_path
        web_path = loader.web_path
        source = loader.source
        #pages = loader.load_and_split()
        pages = loader.load()
        if not download_dir:
            download_dir = str(user_dir() / 'Documents' / 'pdf')
        os.makedirs(download_dir, exist_ok=True)
        pdf_filename = f"{web_path.replace('http://', '').replace('https://', '').replace('/', '__')}"
        pdf_path = os.path.join(download_dir, pdf_filename)
        shutil.copy(file_path, pdf_path)
        updated = dt.datetime.now()

        if not len(pages):
            raise ValueError("No data found")
        data = []
        all_content = []
        for page in pages:
            content = page.page_content
            content = clean_string(content)
            meta_data = page.metadata
            meta_data["url"] = url
            title = meta_data["title"]
            data.append(
                {
                    "content": content,
                    "meta_data": meta_data,
                }
            )
            all_content.append(content)

        doc_id = hashlib.sha256((" ".join(all_content) + url).encode()).hexdigest()
        row = {
        "doc_id": doc_id,
        "total_pages": len(pages) + 1,
        "title": title,
        "pdf_path": str(pdf_path), 
        "source": source,
        "url": url,
        "updated": updated.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save metadata to the database
        self.db["pdfs"].upsert(row, pk="doc_id")

        return {
            "doc_id": doc_id,
            "data": data,
        }

if __name__ == "__main__":
    loader = PdfURLLoader()
    url = "https://arxiv.org/pdf/2102.07947.pdf"
    res = loader.load_data(url)
    print('doc_id:', res['doc_id'])
    print('data:', type(res['data']), len(res['data']), type(res['data'][0]))
    print('data 0', res['data'][0].keys())
    print(res['data'][0]['meta_data'])

