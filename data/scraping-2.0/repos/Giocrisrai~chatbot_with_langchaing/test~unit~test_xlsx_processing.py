import pytest
import os
from unittest.mock import Mock, patch
from langchain.schema.document import Document
from api.app.services.xlsx_processing import process_xlsx
import tempfile


@patch('api.app.services.xlsx_processing.UnstructuredExcelLoader')
def test_process_xlsx_success(mock_loader: Mock) -> None:
    """
    Test to ensure that the process_xlsx function correctly processes a simulated XLSX.

    This test simulates the behavior of UnstructuredExcelLoader returning mock documents,
    and then checks whether the process_xlsx function processes these documents as expected.

    Args:
        mock_loader (Mock): A mock object of UnstructuredExcelLoader.
    """
    # Set up simulated behavior
    mock_docs: List[Document] = [
        Document(page_content='\n  \n    \n      Team\n      Location\n      Stanley Cups\n    \n    \n      Blues\n      STL\n      1\n    \n    \n      Flyers\n      PHI\n      2\n    \n    \n      Maple Leafs\n      TOR\n      13\n    \n  \n',
                 metadata={'source': 'example_data/stanley-cups.xlsx', 'filename': 'stanley-cups.xlsx',
                           'file_directory': 'example_data', 'filetype': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                           'page_number': 1, 'page_name': 'Stanley Cups', 'text_as_html': '<table border="1" class="dataframe">\n  <tbody>\n    <tr>\n      <td>Team</td>\n      <td>Location</td>\n      <td>Stanley Cups</td>\n    </tr>\n    <tr>\n      <td>Blues</td>\n      <td>STL</td>\n      <td>1</td>\n    </tr>\n    <tr>\n      <td>Flyers</td>\n      <td>PHI</td>\n      <td>2</td>\n    </tr>\n    <tr>\n      <td>Maple Leafs</td>\n      <td>TOR</td>\n      <td>13</td>\n    </tr>\n  </tbody>\n</table>',
                           'category': 'Table'})
    ]
    mock_loader.return_value.load.return_value = mock_docs

    # Create a temporary fake XLSX file
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # Execute the function with the temporary file path
        documents = process_xlsx(temp_path)

        # Perform assertions
        assert len(documents) == 1
        assert documents[0].page_content == '\n  \n    \n      Team\n      Location\n      Stanley Cups\n    \n    \n      Blues\n      STL\n      1\n    \n    \n      Flyers\n      PHI\n      2\n    \n    \n      Maple Leafs\n      TOR\n      13\n    \n  \n'
        assert documents[0].metadata['source'] == 'example_data/stanley-cups.xlsx'
    finally:
        # Clean up by removing the temporary file
        os.remove(temp_path)


@patch('api.app.services.xlsx_processing.UnstructuredExcelLoader')
def test_process_xlsx_error(mock_loader: Mock) -> None:
    """
    Test the error handling in the process_xlsx function.

    This test simulates a situation where UnstructuredExcelLoader throws an exception,
    and then checks whether the process_xlsx function correctly propagates this exception.

    Args:
        mock_loader (Mock): A mock object of UnstructuredExcelLoader.
    """
    # Set up the mock to throw an exception
    mock_loader.return_value.load.side_effect = Exception("Error de carga")

    # Create a temporary fake XLSX file
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # Verify that an exception is thrown as expected
        with pytest.raises(Exception) as exc_info:
            process_xlsx(temp_path)
        assert "Error de carga" in str(exc_info.value)
    finally:
        # Clean up by removing the temporary file
        os.remove(temp_path)
