# MD加载
def loadMD(path):
    from langchain.document_loaders import UnstructuredMarkdownLoader
    import re
    
    loader = UnstructuredMarkdownLoader(path)
    document = loader.load()
    con = document[0].page_content.replace("\n\n","\n")
    pattern = r'\\u[0-9]{4}'
    result = re.sub(pattern, " ", con)
    return result