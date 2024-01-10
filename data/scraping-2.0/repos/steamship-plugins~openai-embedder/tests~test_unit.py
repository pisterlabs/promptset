import os
from typing import List

import pytest
from steamship import Block, SteamshipError
from steamship.data.file import File
from steamship.data.tags import DocTag, Tag, TagKind, TagValueKey
from steamship.plugin.inputs.block_and_tag_plugin_input import BlockAndTagPluginInput
from steamship.plugin.request import PluginRequest

from api import OpenAIEmbedderPlugin
from openai.api_spec import MODEL_TO_DIMENSIONALITY
from tagger.span import Granularity


def _read_test_file_lines(filename: str) -> List[str]:
    folder = os.path.dirname(os.path.abspath(__file__))
    lines = []
    with open(os.path.join(folder, "..", "test_data", "inputs", filename), "r") as f:
        lines = list(map(lambda line: line, f.read().split("\n")))
    return lines


def _read_test_file(filename: str) -> File:
    lines = _read_test_file_lines(filename)
    blocks = list(map(lambda t: Block(id=t[0], text=t[1]), enumerate(lines)))
    return File(id="XYZ", blocks=blocks)


def _file_from_string(string: str) -> File:
    lines = string.split("\n")
    blocks = list(map(lambda t: Block(id=t[0], text=t[1]), enumerate(lines)))
    return File(id="XYZ", blocks=blocks)


def test_embed_english_sentence():
    FILE = "roses.txt"
    MODEL = "text-embedding-ada-002"

    embedder_block_text = OpenAIEmbedderPlugin(config={
        "api_key":"",
        "model": MODEL,
    })

    file = _read_test_file(FILE)

    request = PluginRequest(data=BlockAndTagPluginInput(file=file))
    response = embedder_block_text.run(request)

    for block in response.file.blocks:
        for tag in block.tags:
            assert tag.kind == TagKind.EMBEDDING
            assert tag.value.get(TagValueKey.VECTOR_VALUE) is not None
            assert len(tag.value.get(TagValueKey.VECTOR_VALUE)) == MODEL_TO_DIMENSIONALITY[MODEL]

    assert response.usage is not None
    assert len(response.usage) == 3

    embedder_tokens_text = OpenAIEmbedderPlugin(
        config={
            "api_key": "",
            "model": MODEL,
            "granularity": Granularity.TAG,
            "kind_filter": TagKind.DOCUMENT,
            "name_filter": DocTag.TOKEN,
        }
    )

    # Add the tokens.
    for block in file.blocks:
        start_idx = 0
        tokens = block.text.split(" ")
        for token in tokens:
            block.tags.append(Tag(
                file_id=file.id,
                block_id=block.id,
                kind=TagKind.DOCUMENT,
                name=TagKind.TOKEN,
                start_idx=start_idx,
                end_idx=start_idx + len(token)
            ))
            start_idx += len(token)

    request2 = PluginRequest(data=BlockAndTagPluginInput(file=file))
    response2 = embedder_tokens_text.run(request2)

    for (block_in, block_out) in zip(file.blocks, response2.file.blocks):
        tags_in, tags_out = block_in.tags, block_out.tags
        filtered_tags_in = [tag for tag in tags_in if tag.start_idx != tag.end_idx]
        assert len(tags_out) == len(filtered_tags_in)
        for tag_1, tag_2 in zip(filtered_tags_in, tags_out):
            assert tag_1.kind == TagKind.DOCUMENT
            assert tag_2.kind == TagKind.EMBEDDING
            assert tag_1.start_idx == tag_2.start_idx
            assert tag_1.end_idx == tag_2.end_idx

    # Now try without a file_id, which is how the embedding index will call it.
    file.id = None
    for block in file.blocks:
        block.id = None

    request = PluginRequest(data=BlockAndTagPluginInput(file=file))
    response = embedder_block_text.run(request)
    assert len(response.file.blocks) > 0
    block = response.file.blocks[0]
    assert len(block.tags) > 0
    for tag in block.tags:
        assert tag.kind == TagKind.EMBEDDING
        assert tag.value.get(TagValueKey.VECTOR_VALUE) is not None
        assert len(tag.value.get(TagValueKey.VECTOR_VALUE)) == MODEL_TO_DIMENSIONALITY[MODEL]

def test_invalid_model_for_billing():

    with pytest.raises(SteamshipError) as e:
        _ = OpenAIEmbedderPlugin(config={'model': 'a model that does not exist', 'api_key':""})
        assert "This plugin cannot be used with model" in str(e)