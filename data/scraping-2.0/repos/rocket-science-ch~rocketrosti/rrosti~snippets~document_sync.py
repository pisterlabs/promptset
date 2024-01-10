# Copyright (c) 2023 Rocket Science AG, Switzerland

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Parse documents from `config.document_sync.source_docs_path` into `config.document_sync.parsed_docs_path`.

Generate snippets and maintain an embedding cache.
"""

import asyncio
import hashlib
from pathlib import Path
from typing import Awaitable, Iterable

import orjson
from aiopath import AsyncPath  # type: ignore[import]
from loguru import logger
from tqdm import tqdm

from rrosti.llm_api import openai_api, openai_api_direct
from rrosti.snippets.parsed_document import ParsedDocument
from rrosti.snippets.snippet import EmbeddingCache, Snippet
from rrosti.utils.config import config


async def _handle_source_document(doc_path: Path) -> bool:
    """
    Parse a source document and output it to `config.document_sync.parsed_docs_path`.

    Returns True if something was updated.
    """

    if doc_path.suffix.lower() not in [".pdf", ".txt"]:
        logger.warning(f"Unknown file type, ignoring document: {doc_path}")
        return False

    target_apath = AsyncPath(config.document_sync.parsed_docs_path / (doc_path.name + ".json"))

    doc_apath = AsyncPath(doc_path)
    doc_data = await doc_apath.read_bytes()

    # If the target exists, we check (using the sha256) if it's the same document.
    if await target_apath.exists():
        json_text = await target_apath.read_bytes()
        # TODO: handle corrupted JSON
        parsed_document = ParsedDocument.from_dict(orjson.loads(json_text))
        new_sha = hashlib.sha256(doc_data).hexdigest()
        if parsed_document.sha256 == new_sha:
            logger.trace(f"Document already parsed: {doc_path}")
            return False
        logger.info("Document changed: {}; old={}, new={}", doc_path, parsed_document.sha256, new_sha)
    else:
        logger.info("New document: {}", doc_path)

    # Look at the suffix. We handle pdf and txt.
    if doc_path.suffix.lower() == ".pdf":
        raise NotImplementedError("PDF parsing is not yet implemented")
        # parsed_document = await pdf_parse.parse_pdf(doc_path)
    elif doc_path.suffix.lower() == ".txt":  # noqa: RET506 (unnecessary elif after raise)
        parsed_document = ParsedDocument.from_textfile_bytes(doc_data, name=doc_path.name, path=str(doc_path))
    else:
        assert False

    # Write the parsed document to `config.document_sync.parsed_docs_path`. Add .json to the end of the filename.
    output_apath = AsyncPath(config.document_sync.parsed_docs_path / (doc_path.name + ".json"))
    await output_apath.parent.mkdir(parents=True, exist_ok=True)
    await output_apath.write_bytes(orjson.dumps(parsed_document.to_dict()))

    return True


def _snippetize_document(path: Path) -> list[Snippet]:
    """Take in a parsed document and output a list of snippets."""
    # logger.info("Reading bytes from {}", path)
    data = path.read_bytes()
    # logger.info("Parsing JSON")
    json_dict = orjson.loads(data)
    # logger.info("Creating ParsedDocument")
    doc = ParsedDocument.from_dict(json_dict)
    # logger.info("Getting snippets")
    return doc.get_snippets(images=True)


async def _snippetize_documents(paths: Iterable[Path]) -> list[Snippet]:
    """Take in a list of parsed documents and output a list of snippets."""
    aws: list[Awaitable[list[Snippet]]] = [asyncio.to_thread(_snippetize_document, path) for path in paths]

    snippets: list[Snippet] = []
    for aw in aws:
        snippets.extend(await aw)

    return snippets


async def sync_and_get_snippets() -> list[Snippet]:
    config.document_sync.data_gen_path.mkdir(parents=True, exist_ok=True)
    openai_provider = openai_api_direct.DirectOpenAIApiProvider()  # TODO

    logger.info("Looking for source documents...")
    source_documents = [p for p in config.document_sync.source_docs_path.glob("*") if p.is_file()]
    logger.info(f"Found {len(source_documents)} source documents")

    if not source_documents:
        logger.error("No source documents found")
        raise RuntimeError("No source documents found")

    # Parse all source documents.

    # Ok, we could do this in parallel, but there's something fishy in the PDF parsing library;
    # some PDFs seem to randomly cause it to segfault. Doing it sequentially mitigates this,
    # at least in the sense that we will compute one PDF at a time and save the result.

    updated = [await _handle_source_document(path) for path in source_documents]

    num_updated = sum(updated)
    if num_updated == 0:
        logger.info("No documents updated")
        # TODO: Detect removed documents so we don't need to always regenerate the snippets?
    else:
        logger.info("Updated {} documents.", num_updated)

    # Snippetize all parsed documents.
    logger.info("Generating snippets...")
    snippets = await _snippetize_documents(config.document_sync.parsed_docs_path.glob("*.json"))

    # Load cached embeddings so we won't recompute for snippets we've already seen.
    cache = await asyncio.to_thread(EmbeddingCache.load)
    await asyncio.to_thread(cache.sync_with_snippets, snippets)

    no_embedding = [s for s in snippets if not s.has_embedding]
    logger.info("Found {} (out of {}) snippets without embeddings.", len(no_embedding), len(snippets))

    async def ensure_embeddings_and_sync(
        openai_provider: openai_api.OpenAIApiProvider, snippets: list[Snippet]
    ) -> None:
        await Snippet.async_ensure_embeddings(openai_provider, snippets)
        cache.sync_with_snippets(snippets)

    if no_embedding:
        logger.info("Querying embeddings...")
        aws: list[Awaitable[None]] = []
        for i in tqdm(range(0, len(no_embedding), config.openai_api.endpoint.max_embedding_requests_per_query)):
            chunk = no_embedding[i : i + config.openai_api.endpoint.max_embedding_requests_per_query]
            aws.append(ensure_embeddings_and_sync(openai_provider, chunk))

        await asyncio.gather(*aws)
        await asyncio.to_thread(cache.save)

    return snippets


if __name__ == "__main__":
    asyncio.run(sync_and_get_snippets())
