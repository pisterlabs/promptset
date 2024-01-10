"""
*** PCnewsGPT Hilfsprogramm: PDF-Importervergleich ***
"""

#zu test-importierende Datei
file_path = "source_documents/n178.pdf"

# zu testende Loader
from langchain.document_loaders import (
    PyPDFLoader,
    UnstructuredPDFLoader,
    PDFMinerLoader,
    PyPDFium2Loader,
    PyMuPDFLoader,
)
loaders = [
#    (UnstructuredPDFLoader, {'mode':'elements'}),
#    (PyPDFLoader,{}),
#    (PDFMinerLoader,{}),
    (PyPDFium2Loader,{}),
#    (PyMuPDFLoader,{}),
    ]

"""
Initial banner Message
"""
print("\nPCnewsGPT PDF-Importer im Vergleich V0.1\n")

"""
Load and convert file_path into a langchain document
"""
from langchain.docstore.document import Document as langchain_Document
from re import sub as regex_sub
def load_file(file_path: str, loader_class, loader_args) -> langchain_Document:
    # load via langchain
    loader = loader_class(file_path, **loader_args)
    lc_doc = loader.load()
    for i in range(len(lc_doc)):
        # Tidy up PDFs
        doc = lc_doc[i].page_content
        source = lc_doc[i].metadata['source']
        page_num = lc_doc[i].metadata.get('page', None)
        creationdate = lc_doc[i].metadata.get('creationdate', None)
        # remove line-break hyphenations
        doc = regex_sub(r'-\n+ *', '',doc)
        # remove training spaces in lines
        doc =doc.replace(' \n', '\n')
        # remove excess spaces
        doc = doc.replace('  ', ' ')
        # substitute known ligatures & strange characters
        doc = doc.replace('(cid:297)', 'fb')
        doc = doc.replace('(cid:322)', 'fj')
        doc = doc.replace('(cid:325)', 'fk')
        doc = doc.replace('(cid:332)', 'ft')
        doc = doc.replace('(cid:414)', 'tf')
        doc = doc.replace('(cid:415)', 'ti')
        doc = doc.replace('(cid:425)', 'tt')
        doc = doc.replace('(cid:426)', 'ttf')
        doc = doc.replace('(cid:427)', 'tti')
        doc = doc.replace('\uf0b7', '*')
        doc = doc.replace('•', '*')
        doc = doc.replace('\uf031\uf02e', '1.')
        doc = doc.replace('\uf032\uf02e', '2.')
        doc = doc.replace('\uf033\uf02e', '3.')
        doc = doc.replace('\uf034\uf02e', '4.')
        doc = doc.replace('\uf035\uf02e', '5.')
        doc = doc.replace('\uf036\uf02e', '6.')
        doc = doc.replace('\uf037\uf02e', '7.')
        doc = doc.replace('\uf038\uf02e', '8.')
        doc = doc.replace('\uf039\uf02e', '9.')
        doc = doc.replace('\uf0d8', '.nicht.')
        doc = doc.replace('\uf0d9', '.und.')
        doc = doc.replace('\uf0da', '.oder.')
        doc = doc.replace('→', '.impliziert. (Mathematisch)')
        doc = doc.replace('\uf0de', '.impliziert.')
        doc = doc.replace('↔', '.äquivalent. (Mathematisch)')
        doc = doc.replace('\uf0db', '.äquivalent.')
        doc = doc.replace('≈','.annähernd.')
        doc = doc.replace('\uf061', 'Alpha')
        doc = doc.replace('β', 'Beta')
        doc = doc.replace('\uf067', 'Gamma')
        # substiture other strange characters
        doc = doc.replace('€', 'Euro')
        doc = doc.replace("„", '"')             # Anführungszeichen
        doc = doc.replace("—", '"')             # m-dash
        doc = doc.replace("'", '"')             # replace single with double quotes
        doc = doc.replace("\t", " ")            # replace tabs with a space
        doc = doc.replace("\r", "")             # delete carriage returns
        doc = doc.replace("\v", "")             # delete vertical tabs
        
        # change single \n in content to " ", but not multiple \n
        doc = regex_sub(r'(?<!\n)\n(?!\n)', ' ',doc)
        # change multiple consecutive \n in content to just one \n
        doc = regex_sub(r'\n{2,}', '\n',doc)
        # remove strange single-characters with optional leading and trailing spaces in lines
        doc = regex_sub(r'\n *(\w|\*) *\n', '\n',doc)
        # remove strange single-characters with spaces inbetween texts
        doc = regex_sub(r'((\w|/|:) +){3,}(\w|/|:)', '',doc)
        # remove multiple blanks
        doc = regex_sub(r'  +', ' ',doc)
        
        # split doc-content into pages and remove any trailing empty pages
        pages =doc.split('\x0c')
        while (len(pages) > 1) and (pages[-1] == ''):
            pages.pop()
        # if there are no remaining pages, empty the text in lc_doc
        if len(pages) == 0:
            lc_doc[i].page_content = ""
        # if its just one page, update lc_doc with processed content
        elif len(pages) == 1:
            lc_doc[i].page_content = pages[0]
            lc_doc[i].metadata = {"source": f"{source}"}
            if page_num is not None:
                lc_doc[i].metadata.update({"page": f"{page_num+1}"})
            if creationdate is not None:
                lc_doc[i].metadata.update({"creationdate": f"'{regex_sub(r'[^0-9]','',creationdate)[:8]}'" })
        # if there are muliple pages, create new lc_doc from non-empty pages
        else:
            del lc_doc[i]    # we need a new name + content as multiple pages
            pg_num=0
            for page in pages:
                if page != '':    # only add non-empty pages, keep page numbering
                    metadata = {"source": f"{source}", "page": f"{pg_num+1}"}
                    if creationdate is not None:
                        metadata.update({"creationdate": f"'{regex_sub(r'[^0-9]','',creationdate)[:8]}'" })
                    lc_doc.insert(i, langchain_Document(
                        page_content = page,
                        metadata=metadata
                    ))
                    i+=1
                pg_num += 1

    # remove empty documents
    docs=[]
    for i in range(len(lc_doc)):
        if lc_doc[i].page_content != '':
            docs.append(lc_doc[i])
    # return the loaded doc               
    return docs


"""
process file with all loaders
"""
for idx,loader in enumerate(loaders):
    loader_class, loader_args = loader
    print(f"*** File {file_path}, processed with {loader_class.__name__} ***:")
    documents=load_file(file_path, loader_class, loader_args)
    print(documents,end="\n\n")
