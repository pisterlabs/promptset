import PyPDF2
from typing import Callable, List, Dict
import io, json
from langchain.schema import Document
import pandas as pd

def process_txt(file_contents: bytes, meta_data: dict) -> List[Document]:
    all_text = file_contents.decode("utf-8", errors="ignore")
    return [Document(page_content=all_text, metadata=meta_data)]

def process_pdf(file_contents: bytes, meta_data: dict) -> List[Document]:
    pdfReader = PyPDF2.PdfReader(io.BytesIO(file_contents))

    # Extract text from each page
    doc_list = []
    for page in range(len(pdfReader.pages)):
        pageObj = pdfReader.pages[page]
        text = pageObj.extract_text()
        metadata = meta_data.copy()
        metadata["page"] = page
        curr_doc = Document(page_content=text, metadata=metadata)
        doc_list.append(curr_doc)
    return doc_list

def process_csv(file_contents: bytes, meta_data: dict) -> List[Document]:
    # build up the dictionary, similar to how langchain handles the csv files
    doc_list = []
    df = pd.read_csv(io.BytesIO(file_contents))
    for i, row in df.iterrows():
        # Convert row to dictionary
        content = json.dumps(row.to_dict(), ensure_ascii=False)
        metadata = meta_data.copy()
        metadata["row"] = i
        doc = Document(page_content=content, metadata=metadata)
        doc_list.append(doc)
    return doc_list

def process_excel(file_contents: bytes, meta_data: dict) -> List[Document]:
    # build up the dictionary, similar to how langchain handles the csv files
    doc_list = []
    # get all possible sheets 
    excel_file = pd.ExcelFile(io.BytesIO(file_contents))
    sheet_names = excel_file.sheet_names
    dfs: Dict[str, pd.DataFrame] = {}
    for sheet_name in sheet_names:
        dfs[sheet_name] = pd.read_excel(io.BytesIO(file_contents), sheet_name=sheet_name)
        for column in dfs[sheet_name].columns:
            # Check if column is JSON-serializable
            try:
                json.dumps(dfs[sheet_name][column].tolist())
            except TypeError:
                # Column is not JSON-serializable, convert to string
                dfs[sheet_name][column] = dfs[sheet_name][column].astype("str")
    
    # iterate through all sheets
    for sheet_name in sheet_names:
        for i, row in dfs[sheet_name].iterrows():
            # Convert row to dictionary
            content = json.dumps(row.to_dict(), ensure_ascii=False)
            metadata = meta_data.copy()
            metadata["sheet_name"] = sheet_name
            metadata["row"] = i
            doc = Document(page_content=content, metadata=metadata)
            doc_list.append(doc)
    return doc_list

ALL_PROCESSORS = {
    "application/pdf": process_pdf,
    "application/json": process_txt,
    "application/javascript": process_txt,
    "text/css": process_txt,
    "text/csv": process_csv,
    "text/javascript": process_txt,
    "text/plain": process_txt,
    "application/vnd.ms-excel": process_excel,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": process_excel,
    "application/vnd.oasis.opendocument.spreadsheet": process_excel,
}

# HTML content types:
# Applications:
# application/java-archive
# application/EDI-X12   
# application/EDIFACT   
# application/javascript   
# application/octet-stream   
# application/ogg   
# application/pdf  
# application/xhtml+xml   
# application/x-shockwave-flash    
# application/json  
# application/ld+json  
# application/xml   
# application/zip  
# application/x-www-form-urlencoded  
# Audios: 
# audio/mpeg   
# audio/x-ms-wma   
# audio/vnd.rn-realaudio   
# audio/x-wav   
# Images: 
# image/gif   
# image/jpeg   
# image/png   
# image/tiff    
# image/vnd.microsoft.icon    
# image/x-icon   
# image/vnd.djvu   
# image/svg+xml    
# Multipart:
# multipart/mixed    
# multipart/alternative   
# multipart/related (using by MHTML (HTML mail).)  
# multipart/form-data  
# Text:
# text/css    
# text/csv    
# text/html    
# text/javascript (obsolete)    
# text/plain    
# text/xml    
# Video:
# video/mpeg    
# video/mp4    
# video/quicktime    
# video/x-ms-wmv    
# video/x-msvideo    
# video/x-flv   
# video/webm   
# VND: 
# application/vnd.android.package-archive
# application/vnd.oasis.opendocument.text    
# application/vnd.oasis.opendocument.spreadsheet  
# application/vnd.oasis.opendocument.presentation   
# application/vnd.oasis.opendocument.graphics   
# application/vnd.ms-excel    
# application/vnd.openxmlformats-officedocument.spreadsheetml.sheet   
# application/vnd.ms-powerpoint    
# application/vnd.openxmlformats-officedocument.presentationml.presentation    
# application/msword   
# application/vnd.openxmlformats-officedocument.wordprocessingml.document   
# application/vnd.mozilla.xul+xml   