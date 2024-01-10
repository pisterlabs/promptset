import os
from typing import List

from steamship import Block
from steamship import Steamship
from steamship.data.file import File
from steamship.data.tags import TagKind, TagValueKey

from openai.api_spec import MODEL_TO_DIMENSIONALITY

EMBEDDER_HANDLE = "openai-embedder"
PROFILE = "prod"
MODEL = "text-embedding-ada-002"


def _read_test_file_lines(filename: str) -> List[str]:
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, "..", "test_data", "inputs", filename), "r") as f:
        lines = list(map(lambda line: line, f.read().split("\n")))
    return lines


def _read_test_file(client: Steamship, filename: str) -> File:
    lines = _read_test_file_lines(filename)
    blocks = list(map(lambda t: Block(text=t[1]), enumerate(lines)))
    return File.create(client=client, blocks=blocks)


def test_embed_english_sentence():
    FILE = "roses.txt"

    with Steamship.temporary_workspace(profile=PROFILE) as client:

        embedder = client.use_plugin(EMBEDDER_HANDLE,
                                     config={"model": MODEL, "dimensionality": MODEL_TO_DIMENSIONALITY[MODEL]})

        file = _read_test_file(client, FILE)

        e1 = file.tag(embedder.handle)
        e1.wait()

        for block in e1.output.file.blocks:
            for tag in block.tags:
                assert tag.kind == TagKind.EMBEDDING
                assert tag.value.get(TagValueKey.VECTOR_VALUE) is not None
                assert len(tag.value.get(TagValueKey.VECTOR_VALUE)) == MODEL_TO_DIMENSIONALITY[MODEL]
