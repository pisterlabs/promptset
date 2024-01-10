from langchain.document_loaders import UnstructuredHTMLLoader

# requires `pip install unstructured`
loader = UnstructuredHTMLLoader("sample.html")
data = loader.load()
print(data)
"""
[Document(page_content="Welcome to My Web Page\n\nThis is a simple HTML page. It's a great starting point for learning HTML.\n\nClick here to visit Example.com", metadata={'source': 'sample.html'})]
"""

from langchain.document_loaders import BSHTMLLoader

# requires `pip install beautifulsoup4 lxml`
loader = BSHTMLLoader("sample.html")
data = loader.load()
print(data)
"""
[Document(page_content="\n\nMy Simple Web Page\n\n\nWelcome to My Web Page\nThis is a simple HTML page. It's a great starting point for learning HTML.\nClick here to visit Example.com\n\n\n", metadata={'source': 'sample.html', 'title': 'My Simple Web Page'})]
"""

print(data[0].page_content)
"""


My Simple Web Page


Welcome to My Web Page
This is a simple HTML page. It's a great starting point for learning HTML.
Click here to visit Example.com



"""