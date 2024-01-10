from langchain.document_loaders import PyPDFLoader

pdf_path = "./textsplitter/data/S19-15665.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()
batches = []
for page in pages:
    #print("=" * 50)
    page_len = 0
    batch = ""
    for i in page.page_content.split("\n"):
        if (page_len + len(i)) > 500:
            batches.append(batch)
            batch = ""
            page_len = 0
        page_len += len(i)
        batch += i + " "

print(batches)    

#print(pages[0].page_content)
