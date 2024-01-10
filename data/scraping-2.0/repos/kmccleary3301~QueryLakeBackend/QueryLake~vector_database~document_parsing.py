from langchain.docstore.document import Document
from typing import List
from langchain.document_loaders import PyPDFium2Loader
from langchain.document_loaders.blob_loaders import Blob
# import PyPDF2
from PyPDF2 import PdfReader
from io import BytesIO
from urllib import request
from markdownify import markdownify as md
import re
# pdf_loader = PyPDFium2Loader("/home/user/python_projects/3035/QueryLakeBackend/user_db/pdf_ium_tmp")



# PyPDF2.PdfFileReader()

def parse_PDFs(bytes_in, return_all_text_as_string : bool = False) -> List[Document]:
    if type(bytes_in) is bytes:
        bytes_in = BytesIO(bytes_in)
    reader = PdfReader(bytes_in)
    # blob_get = Blob(bytes_in)
    # print(reader)

    all_text = []
    pages_text = []
    # reader = PdfReader("GeoBase_NHNC1_Data_Model_UML_EN.pdf")
    for i, page in enumerate(reader.pages):
        # print("Page #%03d" % (i))
        # print(page.mediabox)
        # page = reader.pages[3]
        parts = []

        def visitor_body(text, cm, tm, fontDict, fontSize):
            # try:
            # print(text, list(page.mediabox))
            zoom_ratio = 266*list(page.mediabox)[3]*9/(list(page.mediabox)[2]*16)
            url = "#page=%d&zoom=%d,%.1f,%.1f" % (i+1, zoom_ratio, 0, min(list(page.mediabox)[3], tm[5]+(fontSize*len(text.split("\n")))+5)) # IDK
            url_2 = "#page=%d&zoom=%d,%.1f,%.1f" % (i+1, zoom_ratio, 0, max(0, float(list(page.mediabox)[3])-list(tm)[5]-5))
            parts.append(text)
            all_text.append((text, {
                "location_link_firefox": url,
                "location_link_chrome": url_2,
                "page": i+1
            }))
            # y = float(list(tm)[5])
            # if y > 50 and y < 720:
            #     print("1")
            #     print("2")
            #     print("3")
                # print(text, cm, tm, fontDict, fontSize)
                # print(text, url)
            # except Exception as e:
            #     print(e)

    
        page.extract_text(visitor_text=visitor_body)
        # text_body = " ".join(parts)
        # pages_text.append(text_body)
        
        # print(text_body)
    # print(all_text)
    # print(pages_text)
    # return pages_text
    if return_all_text_as_string:
        return " ".join([pair[0] for pair in all_text])
    return all_text
    # print(pdf_loader.lazy_load(blob_get))


def parse_url(url_in : str) -> Document:
    try:
        resource = request.urlopen(request.Request(url=url_in, headers={'User-Agent': 'Mozilla/5.0'}))
        content =  resource.read().decode(resource.headers.get_content_charset())
        webpage = content
        find_script = webpage.find("<script>")
        while find_script != -1:
            find_end = find_script+webpage[find_script:].find("</script>")
            webpage = webpage[:find_script]+webpage[find_end+len("</script>"):]
            find_script = webpage.find("<script>")
        return re.sub(r"[\n]+", "\n", md(webpage))
    except:
        return None

# def parse_URLS(urls: List[str]) -> List[Document]:
#     loader = UnstructuredURLLoader(urls=urls)
#     return loader.load()

# def parse_text(text_segments: List[str]) -> List[Document]:

