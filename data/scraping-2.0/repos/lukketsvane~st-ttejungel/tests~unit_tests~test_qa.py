from langchain.docstore.document import Document
from knowledge_gpt.core.qa import get_sources
from knowledge_gpt.core.embedding import FolderIndex

from typing import List
from .fake_file import FakeFile
from knowledge_gpt.core.parsing import File

from knowledge_gpt.core.debug import FakeVectorStore


def test_getting_sources_from_answer():
    """Test that we can get the sources from an answer."""
    files: List[File] = [
        FakeFile(
            name="file1",
            id="1",
            docs=[
                Document(page_content="1", metadata={"kilder": "1"}),
                Document(page_content="2", metadata={"kilder": "2"}),
            ],
        ),
        FakeFile(
            name="file2",
            id="2",
            docs=[
                Document(page_content="3", metadata={"kilder": "3"}),
                Document(page_content="4", metadata={"kilder": "4"}),
            ],
        ),
    ]
    folder_index = FolderIndex(files=files, index=FakeVectorStore(texts=[]))

    answer = "Her er utkastet. KILDER: 1, 2, 3, 4"

    sources = get_sources(answer, folder_index)

    assert len(sources) == 4
    assert sources[0].metadata["kilder"] == "1"
    assert sources[1].metadata["kilder"] == "2"
    assert sources[2].metadata["kilder"] == "3"
    assert sources[3].metadata["kilder"] == "4"
