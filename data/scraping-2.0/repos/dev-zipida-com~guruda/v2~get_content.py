from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from github import Github
import os

def get_github_contents(addr, branch_name):

    contents_list = []
    access_token = os.getenv("GITHUB_API_KEY")
    g = Github(access_token)
    repo = g.get_repo(addr)

    contents = repo.get_contents('', ref=branch_name)
    contents_list = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function = len,
    )
    # GET github repository contents using Github API
    while contents:
        file_content = contents.pop(0)
        extensions = (".go", ".py", ".js", ".ts", ".tsx", ".html", ".css", ".md", ".java", ".c", ".cpp")

        if file_content.type == 'dir':
            contents.extend(repo.get_contents(file_content.path, ref=branch_name))
        else:
            file_extension = os.path.splitext(file_content.path)[1]
            if file_extension not in extensions:
                continue
            contents_list.extend(text_splitter.create_documents([file_content.decoded_content.decode("utf-8")]))

    return contents_list    
