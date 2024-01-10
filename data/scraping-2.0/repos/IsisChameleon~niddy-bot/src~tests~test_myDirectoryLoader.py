import unittest
from unittest.mock import patch, Mock
from pathlib import Path
from modules.loaders import MyDirectoryLoader
from langchain.schema import Document

class TestMyDirectoryLoader(unittest.TestCase):

    @patch('modules.loaders.CSVLoader')
    @patch('modules.loaders.PDFPlumberLoader')
    def test_load(self, MockPDFPlumberLoader, MockCSVLoader):
        # Arrange
        dir_path = Path('some/directory')
        csv_loader = Mock()
        pdf_loader = Mock()
        MockCSVLoader.return_value = csv_loader
        MockPDFPlumberLoader.return_value = pdf_loader
        csv_loader.load.return_value = [Document(page_content='csv_doc1'), Document(page_content='csv_doc2')]
        pdf_loader.load.return_value = [Document(page_content='pdf_doc1'), Document(page_content='pdf_doc2')]

        # Act
        my_directory_loader = MyDirectoryLoader(dir_path)
        with patch('modules.loaders.Path.rglob', return_value=[
            Path('file1.csv'),
            Path('file2.pdf'),
            Path('file3.txt')
        ]), patch('modules.loaders.Path.is_file', return_value=True): 
            docs = my_directory_loader.load()

        # Assert
        self.assertEqual(docs, [Document(page_content='csv_doc1'), Document(page_content='csv_doc2'),Document(page_content='pdf_doc1'), Document(page_content='pdf_doc2')])
        MockCSVLoader.assert_called_once_with('file1.csv')
        MockPDFPlumberLoader.assert_called_once_with('file2.pdf')
        csv_loader.load.assert_called_once()
        pdf_loader.load.assert_called_once()

if __name__ == '__main__':
    unittest.main()
