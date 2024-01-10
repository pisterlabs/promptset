from pydantic import BaseModel, validator
from typing import List,Any,Optional,Dict,Tuple,Union
from pathlib import Path
import pickle
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai
import os
import matplotlib 
openai.api_key = "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"
DocKey = Any

class Doc(BaseModel):
    docname: str
    citation: str
    dockey: DocKey


class Text(BaseModel):
    text: str
    name: str
    doc: Doc
    embeddings: Optional[List[float]] = None
    similarity: Optional[float] = None

    #@validator("embeddings")
    #def check_embeddings(cls, v):
    #    if v is None:

    def calculate_embeddings(self, model="text-embedding-ada-002"):
        if self.embeddings is None:
            self.embeddings = get_embedding(self.text, model)
        return self.embeddings

def save_texts(texts: List[Text], filename):
    with open(filename, 'wb') as f:
        pickle.dump(texts, f)


def parse_pdf2(path: Path, doc: Doc, chunk_chars: int, overlap: int) -> List[Text]:
    import pypdf

    pdfFileObj = open(path, "rb")
    pdfReader = pypdf.PdfReader(pdfFileObj)
    split = ""
    pages: List[str] = []
    texts: List[Text] = []
    for i, page in enumerate(pdfReader.pages):
        split += page.extract_text()
        pages.append(str(i + 1))
        while len(split) > chunk_chars:
            pg = "-".join([pages[0], pages[-1]])
            t = Text(
                    text=split[:chunk_chars], name=f"{doc.docname} pages {pg}", doc=doc
                )
            t.calculate_embeddings()
            texts.append(t)
            split = split[chunk_chars - overlap :]
            pages = [str(i + 1)]
    if len(split) > overlap:
        pg = "-".join([pages[0], pages[-1]])
        t = Text(text=split[:chunk_chars], name=f"{doc.docname} pages {pg}", doc=doc)
        t.calculate_embeddings()
        texts.append(t)
    pdfFileObj.close()
    return texts
doc = Doc(docname='GEO', citation='Geo', dockey='Xu2023')

file_path = Path('pdfresizer.com-pdf-resize.pdf')
texts = parse_pdf2(file_path, doc, chunk_chars=2000, overlap=400)

save_texts(texts, 'geoinfo3.pkl')