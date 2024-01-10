from langchain.text_splitter import MarkdownHeaderTextSplitter
import pickle as pickle
import os

def read_md(path):
    # read the file named md_anderson_chemo.md and return the string.
    with open(path, "r") as f:
        text = f.read()
    return text

def split_md(text):
    # split the text into a list of strings
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(text)
    return md_header_splits

def save_to_pkl(docs):
    pickled_str = pickle.dumps(docs)
    with open("./documents/pickled_documents.pkl", "wb") as f:
        f.write(pickled_str)


def main():
    # read the file named md_anderson_chemo.md and return the string.
    text = read_md("./documents/md_anderson_chemo.md")
    # split the text into a list of strings
    md_header_splits = split_md(text)
    #save the list of strings to a pickle file
    save_to_pkl(md_header_splits)
    # print the first 5 splits

if __name__ == "__main__":
    main()
