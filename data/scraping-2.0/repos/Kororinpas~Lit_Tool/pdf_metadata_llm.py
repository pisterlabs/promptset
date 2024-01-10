from doi import get_doi
from document_util import get_split_documents

def get_pdf_metadata_using_llm(doc):
    
  import re 

  doc[0].page_content = re.sub('\n+',' ',doc[0].page_content.strip())

#   from langchain.text_splitter import RecursiveCharacterTextSplitter
#   text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap = 50)
  split_docs = get_split_documents(doc, 1500, 50)
  abstract = split_docs[0]
  doi = get_doi(abstract)

  if doi != 'None':
    import habanero
    import time
    citation = habanero.cn.content_negotiation(ids = doi,format='bibentry')
    time.sleep(5)
    import bibtexparser
    citation = bibtexparser.loads(citation)
    citation = citation.entries[0]
    metadata = {'author':citation['author'],
            'year':citation['year'],
            'title':citation['title'],
            'journal':citation['journal'],
            }
    return metadata
  else:
    metadata = 'None'
    return metadata