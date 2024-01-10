from langchain.document_loaders import ArxivLoader 

def doc_loader(search_query):
    '''The purpose of this function is to load the documents from Arxiv'''
    '''We are using the Arxiv Loader under Langchain library, which intern invokes the Arxiv API'''

    try :
        docs = ArxivLoader(query=search_query, load_max_docs=1).load()
        return docs[0].metadata, docs[0].page_content
    except Exception as e:
        print(e)