import pytest
import uuid
from landy.utils.lc_handler import LangChainHandler

# Test if the LangChainHandler builds templates correctly
@pytest.mark.asyncio
async def test_build_templates():
    """
    Test if the LangChainHandler's chat_template is not None and if the
    number of messages in the template is 2.
    """
    async with LangChainHandler() as handler:
        assert handler.chat_template is not None
        assert len(handler.chat_template.messages) == 2

# Test if the LangChainHandler can get ChromaDB
@pytest.mark.asyncio
async def test_get_chroma_db():
    """
    Test if the LangChainHandler's db attribute is not None and if the
    db.persist_directory ends with "db".
    """
    async with LangChainHandler() as handler:
        assert handler.db is not None
