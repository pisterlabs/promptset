from __future__ import annotations

from typing import Any

from langchain.chains import create_extraction_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BaseTool

from config import config


schema = {
    "properties": {
        "article_title": {"type": "string"},
        "article_summary": {"type": "string"},
    },
    "required": ["article_title", "article_summary"],
}


def extract(content: str, schema: dict, llm: Any):
    return create_extraction_chain(schema=schema, llm=llm).run(content)


def get_web_content_from_url(urls: list[str]):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106")
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=4096,
        chunk_overlap=0,
    )
    splits = splitter.split_documents(docs_transformed)
    results: list[str] = []
    for split in splits[: config.web_scrapy_max_splits]:
        extracted_content = extract(
            schema=schema,
            content=split.page_content,
            llm=llm,
        )
        results.extend(extracted_content)
    return results


class WebBrowsingTool(BaseTool):
    name = "web_browsing_tool"
    description = (
        "Use this tool to obtain content of web pages" "use parameter `url` as input"
    )

    def _run(self, url: str) -> str:
        web_content = get_web_content_from_url([url])
        return "\n".join(map(str, web_content)) + "\n"

    def _arun(self, image_path: str) -> list[str]:
        raise NotImplementedError("This tool does not support async")


if __name__ == "__main__":
    from pyrootutils import setup_root
    import pprint

    setup_root(".", dotenv=True)
    pprint.pprint(
        get_web_content_from_url(
            [
                "https://www.piyao.org.cn/20231124/88da748b841b4be99374d2b651674d9e/c.html"
            ],
        ),
    )
