from langchain.document_loaders import UnstructuredFileLoader

def extract_text_with_langchain_pdf(pdf_file):
    
    loader = UnstructuredFileLoader(pdf_file)
    documents = loader.load()
    pdf_pages_content = '\n'.join(doc.page_content for doc in documents)
    
    return pdf_pages_content



if __name__ == '__main__':
    text = extract_text_with_langchain_pdf('./data/pdfs/Notion Mastery Sales Page.pdf')
    import ipdb; ipdb.set_trace()