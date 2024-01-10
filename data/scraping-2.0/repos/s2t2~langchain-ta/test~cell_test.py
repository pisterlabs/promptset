#from langchain.docstore.document import Document

from app.cell import Cell


def test_cell_type():

    # a cell is a wrapper for a document, which knows its cell type and has shortcut properties into the metadata:
    code_cell_content = "'code' cell: '['import numpy as np']'"
    text_cell_content =  "'markdown' cell: '['If you now insert your cursor after `np` and press **Period**(`.`), you will see the list of available completions within the `np` module. Completions can be opened again by using **Ctrl+Space**.']'",

    # metadata and cell type:

    cell = Cell(page_content=code_cell_content)
    assert cell.metadata == {"cell_type":"CODE", "is_empty": False}
    assert cell.cell_type == "CODE"

    cell = Cell(page_content=code_cell_content, metadata={"number": 99})
    assert cell.metadata == {"number": 99, "cell_type":"CODE", "is_empty": False}
    assert cell.cell_type == "CODE"

    cell = Cell(page_content=text_cell_content, metadata={"number": 99})
    assert cell.metadata == {"number": 99, "cell_type":"TEXT", "is_empty": False}
    assert cell.cell_type == "TEXT"

    # empty:

    assert Cell(page_content="'code' cell: '[]'").is_empty
    assert Cell(page_content="'markdown' cell: '[]'").is_empty
