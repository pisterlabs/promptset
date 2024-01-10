from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

# Given a file location, read line by line and return the first 100 lines
def read_file(file_name):
    s = ""
    num_lines = 0
    for line in open(file_name):
        s += line
        num_lines += 1
        if num_lines > 100:
            break
    return s

# TODO: Return HTML headers to split on
# Format should be a list of tuples, where the first 
# element is the HTML tag, and the second element is 
# a label for the header
# ex: [("p", "Paragraph"), ("h6", "Header 6"))]
def get_html_headers():
    return []

# TODO: Return Markdown headers to split on
# Format should be a list of tuples, where the first 
# element is the Markdown tag, and the second element is 
# a label for the header
# ex: [("##", "Header2"), ...]
def get_md_headers():
    return []

# Given file name, return the HTML split into chunks
# DO NOT modify this code but please read to understand
# how the text splitter works
def transform_html(file_name):
    doc = read_file(file_name)
    headers_to_split_on = get_html_headers()
    splitter = HTMLHeaderTextSplitter(headers_to_split_on = headers_to_split_on)
    splits = splitter.split_text(doc)
    return splits

# Given file name, return the Markdown split into chunks
# DO NOT modify this code but please read to understand
# how the text splitter works
def transform_md(file_name):
    doc = read_file(file_name)
    headers_to_split_on = get_md_headers()
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(doc)
    return md_header_splits

# TODO: Modify the following code so that it actually
# splits the text into meaningful chunks:
def transform_txt(file_name):
    doc = read_file(file_name)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1,
        chunk_overlap = 0,
        length_function = len,
        is_separator_regex = False,
    )

    split_text = splitter.create_documents([doc])
    
    return split_text

