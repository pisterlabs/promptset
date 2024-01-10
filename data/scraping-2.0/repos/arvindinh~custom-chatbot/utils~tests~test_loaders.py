import sys
sys.path.append('..')
from langchain.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
)
from utils.loaders.loader_mapper import LoaderMapper
import pytest
import re


@pytest.mark.parametrize("doc, expected", [
    ('tests/docs/dummy_doc_twinkle.pdf', PyMuPDFLoader('tests/docs/dummy_doc_twinkle.pdf')),
    ('tests/docs/example.csv', CSVLoader('tests/docs/example.csv')),
    ('tests/docs/dummy.txt', TextLoader(file_path='tests/docs/dummy.txt', encoding="utf8")),
    ('tests/docs/dummy.html', UnstructuredHTMLLoader('tests/docs/dummy.html')),
    ('tests/docs/dummy.md', UnstructuredMarkdownLoader('tests/docs/dummy.md')),
    ('tests/docs/dummy.docx', UnstructuredWordDocumentLoader('tests/docs/dummy.docx')),
    ('tests/docs/dummy.pptx', UnstructuredPowerPointLoader('tests/docs/dummy.pptx')),
    ('tests/docs/dummy.xlsx', UnstructuredExcelLoader('tests/docs/dummy.xlsx')),
])
def test_return_loader(doc, expected):
    mapper = LoaderMapper()
    loader = mapper.find_loader(doc)
    assert type(loader) == type(expected)

@pytest.mark.parametrize("doc, expected", [
    ('tests/docs/dummy_doc_twinkle.pdf',
    """Twinkle, twinkle, little star,\nHow I wonder what you are!\nUp above the world so high,\nLike a diamond in the sky.\nTwinkle, twinkle, little star,\nHow I wonder what you are!"""),
    ('tests/docs/example.csv', 
     """Name: John
        Age: 25
        Country: USA"""),
    ('tests/docs/dummy.txt',
    """Blah blah blah. Sample text. Blah Blah
    Blah Blah Blah. This is so fun. Blah Blah.
    Abcdefghijklmnopqrstuvwxyz.""" ),
    ('tests/docs/dummy.html', 
    """This is a dummy HTML file.
    It serves as an example."""),
    ('tests/docs/dummy.md', 
    """Dummy Markdown File
    This is a dummy Markdown file.
    It serves as an example. Item 1 Item 2 Item 3"""),
    ('tests/docs/dummy.docx',
    """Dummy Document
    This is a dummy Word document."""),
    ('tests/docs/dummy.pptx',
    """Dummy Presentation
    This is a dummy PowerPoint presentation."""),
    ('tests/docs/dummy.xlsx',
    """This is a dummy Excel spreadsheet."""),
])
def test_load_doc(doc, expected):
    mapper = LoaderMapper()
    loader = mapper.find_loader(doc)
    loaded_doc = loader.load()
    text = loaded_doc[0].page_content
    actual_normalized = re.sub(r'\s+', ' ', text.strip())
    expected_normalized = re.sub(r'\s+', ' ', expected.strip())
    assert actual_normalized == expected_normalized
    
    
