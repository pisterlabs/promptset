from typing import List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

class TextTrimmedLoader(BaseLoader):
    """Trim all lines but is still loaded as a single document
    """

    def __init__(self, file_path: str, encoding: Optional[str] = None):
        self.file_path = file_path
        self.encoding = encoding
    
    def load(self) -> 'List[Document]':
        try:
            with open(self.file_path, encoding=self.encoding) as f:
                metadata = {"source": self.file_path}
                lines = []
                for line in f.readlines():
                    line = line.strip()
                    if line != '' or line != '\n' or line != '\r\n':
                        lines.append(line)
                return [Document(page_content="\n".join(lines), metadata=metadata)]
        except Exception as e:
            print("[!] Cannot read file '%s'. " % self.file_path)
        return []