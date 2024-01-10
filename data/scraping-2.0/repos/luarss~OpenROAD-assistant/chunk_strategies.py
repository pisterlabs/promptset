from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter, MarkdownTextSplitter
import re

def remove_text(text):
    # remove code blocks ```
    text = re.sub(r"```.*?```", '', text, flags=re.DOTALL)

    # remove urls
    text = re.sub(r"https?://\S+|www\.\S+", '', text)
    return text
    
def md_splitter(MD_header_split = True, chunk_size = 200, chunk_overlap = 15):
    # returns a MD splitter based on the parameters 
    if MD_header_split: 
        headers_to_split_on = [("#"*i, f"Header {i}") for i in range(1, 8)]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    else:
        markdown_splitter = MarkdownTextSplitter(chunk_size=200, chunk_overlap=15)
    return markdown_splitter

def text_splitter(data, chunk_size = 250, chunk_overlap = 20):
    # Get MD Splitter (fixed to MD_header_split = True for now)
    markdown_splitter = md_splitter()

    # Recursive splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Use MD Headers metadata to remove useless info. 
    remove_headers_1 = ["Contributor Covenant Code of Conduct"]
    remove_headers_2 = ["Authors", "Commands"]

    final = []
    for d in data:
        content, metadata = d.page_content, d.metadata
        content = remove_text(content)
        docs = markdown_splitter.split_text(content)
        for d in docs:
            if 'Header 1' in d.metadata: 
                if any(substring in d.metadata['Header 1'] for substring in remove_headers_1):
                    continue
            if 'Header 2' in d.metadata: 
                if any(substring in d.metadata['Header 2'] for substring in remove_headers_2):
                    continue
            d.metadata.update(metadata)
            final.append(d)

    final = text_splitter.split_documents(final)
    return final