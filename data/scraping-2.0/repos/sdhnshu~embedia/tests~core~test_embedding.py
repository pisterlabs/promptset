import json

import pytest
from embedia import TextDoc

from tests.core.definitions import OpenAIEmbedding

lorem = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis eu arcu risus. Proin sed fringilla tellus. Donec scelerisque elit sed sapien bibendum rutrum. Morbi blandit justo in urna semper volutpat. Nunc consectetur ex vitae consequat blandit. Duis sit amet metus quis mi molestie bibendum rutrum et ante. Nam aliquam metus magna, eget porta lacus dictum sit amet. Morbi dictum tellus a semper tristique. Duis ipsum ex, pharetra non rhoncus in, gravida quis magna. Nam pretium enim non lectus efficitur, sit amet sagittis elit finibus. Vivamus varius ligula turpis, sit amet vehicula mi eleifend eget. Cras dignissim mauris eu feugiat euismod. Integer dapibus dolor eu nulla euismod finibus."""


@pytest.mark.asyncio
async def test_emb_model():
    embmodel = OpenAIEmbedding()

    emb = await embmodel(lorem)
    assert len(emb) == 1536

    doc = TextDoc.from_file("./README.md")
    text = doc.contents
    emb = await embmodel(text)
    assert len(emb) == 1536

    doc = TextDoc.from_file("./README.md", meta={"desc": "Embedia Readme"})
    text = "metadata:" + json.dumps(doc.meta) + "\n" + "content:" + doc.contents
    emb = await embmodel(text)
    assert len(emb) == 1536
