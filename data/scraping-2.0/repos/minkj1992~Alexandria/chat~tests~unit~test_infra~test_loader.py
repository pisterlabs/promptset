import httpx
import pytest
from langchain.docstore.document import Document
from src.infra.langchain.loader import (AsyncRecursiveUrlLoader,
                                        get_docs_from_urls, load_url)

pytestmark = pytest.mark.asyncio
MAX_DEPTH = 8

def mock_get(*args, **kwargs):
    url = args[1]
    mock_request = httpx.Request("GET", url)
    match url.split("/")[-1]:
        case "valid":
            content = """
            <html>
                <body>
                    Valid Content
                </body>
            </html>
            """
            return httpx.Response(200, content=content, request=mock_request)
        case "invalid":
            return httpx.Response(404, content="Not Found", request=mock_request)
        case "depth1":
            content = """
            <html>
                <body>
                    Specific content for depth1
                    <a href="http://test.com/depth1/depth2a">Link to Depth 2A</a>
                    <a href="http://test.com/depth1/depth2b">Link to Depth 2B</a>
                </body>
            </html>
            """
            return httpx.Response(200, content=content, request=mock_request)

        case "depth2a":
            content = """
            <html>
                <body>
                    Specific content for depth 2a
                    <a href="http://test.com/depth1/depth2a/depth3a">Link to Depth 3A</a>
                </body>
            </html>
            """
            return httpx.Response(200, content=content, request=mock_request)

        case "depth2b":
            content = """
            <html>
                <body>
                    Specific content for depth 2b
                </body>
            </html>
            """
            return httpx.Response(200, content=content, request=mock_request)

        case "depth3a":
            content = """
            <html>
                <body>
                    Specific content for depth 3a
                </body>
            </html>
            """
            return httpx.Response(200, content=content, request=mock_request)
        case _:
            return httpx.Response(200, content="Mocked Content", request=mock_request)


@pytest.fixture(autouse=True)
def setup_mocker(mocker):
    mocker.patch.object(httpx.AsyncClient, "request", side_effect=mock_get)


async def test_aload_with_valid_url():
    loader = AsyncRecursiveUrlLoader(url="http://test.com/valid")
    result = await loader.aload()

    assert result is not None
    assert "Valid Content" in result[0].page_content


async def test_aload_with_invalid_url():
    loader = AsyncRecursiveUrlLoader(url="http://test.com/invalid")
    result = await loader.aload()
    assert result == []


async def test_exclude_dirs():
    loader = AsyncRecursiveUrlLoader(
        url="http://test.com/excluded", exclude_dirs=["http://test.com/excluded"]
    )
    result = await loader.aload()

    assert result == []


async def test_max_depth():
    loader = AsyncRecursiveUrlLoader(url="http://test.com/depth1", max_depth=1)
    result = await loader.aload()

    assert len(result) == 3


async def test_extractor_functionality():
    loader = AsyncRecursiveUrlLoader(
        url="http://test.com/valid", extractor=lambda x: "EXTRACTED"
    )
    result = await loader.aload()

    assert "EXTRACTED" in result[0].page_content


async def test_invalid_extractor():
    loader = AsyncRecursiveUrlLoader(url="http://test.com", extractor=lambda x: None)
    result = await loader.aload()
    assert result == []


# Test for load_url function with depth
@pytest.mark.asyncio
async def test_load_url_with_depth():
    expect = set(
        [
            "http://test.com/depth1",
            "http://test.com/depth1/depth2b",
            "http://test.com/depth1/depth2a",
            "http://test.com/depth1/depth2a/depth3a",
        ]
    )

    result = await load_url("http://test.com/depth1", MAX_DEPTH)

    assert len(result) == 4  # depth1, depth2a, depth2b, depth3a

    assert expect == set([r.metadata["source"] for r in result])


# Test for get_docs_from_urls function with multiple URLs
@pytest.mark.asyncio
async def test_get_docs_from_urls_multiple():
    expect = [
        Document(
            page_content="Valid Content",
            metadata={"source": "http://test.com/valid", "language": None, "title": ""},
        ),
        Document(
            page_content="Specific content for depth1 Link to Depth 2A Link to Depth 2B",
            metadata={
                "source": "http://test.com/depth1",
                "language": None,
                "title": "",
            },
        ),
        Document(
            page_content="Specific content for depth 2b",
            metadata={
                "source": "http://test.com/depth1/depth2b",
                "language": None,
                "title": "",
            },
        ),
        Document(
            page_content="Specific content for depth 2a Link to Depth 3A",
            metadata={
                "source": "http://test.com/depth1/depth2a",
                "language": None,
                "title": "",
            },
        ),
        Document(
            page_content="Specific content for depth 3a",
            metadata={
                "source": "http://test.com/depth1/depth2a/depth3a",
                "language": None,
                "title": "",
            },
        ),
    ]
    result = await get_docs_from_urls(
        ["http://test.com/valid", "http://test.com/depth1"], MAX_DEPTH
    )

    assert len(result) == 5  # valid, depth1, depth2a, depth2b, depth3a
    # I did because, Document is not hashable
    for r in result:
        assert r in expect


# Test for load_url function with deeper depth
@pytest.mark.asyncio
async def test_load_url_deeper_depth():
    result = await load_url("http://test.com/depth1/depth2a", MAX_DEPTH)

    assert len(result) == 2  # depth2a, depth3a
    assert "Specific content for depth 2a" in result[0].page_content
    assert "Specific content for depth 3a" in result[1].page_content
