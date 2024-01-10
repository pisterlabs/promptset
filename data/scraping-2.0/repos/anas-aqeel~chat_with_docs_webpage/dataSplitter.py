from langchain.text_splitter import MarkdownHeaderTextSplitter


headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

def split_file_data(file):   
    with open(file, 'r', encoding='utf-8') as f:
        markdown_document = f.read()
        md_header_splits = markdown_splitter.split_text(markdown_document)
 
    return md_header_splits    

