from pathlib import Path
import pytest
import requests

from openai.types import ImagesResponse

from autobots.conn.openai.openai_client import get_openai
from autobots.conn.openai.openai_images.image_model import ImageEdit, ImageCreateVariation


@pytest.mark.asyncio
async def test_create_image_edit_file_happy_path(set_test_settings):
    SOURCE_FILE="tests/resources/openai/test_images/suit_1.png"
    path = Path(SOURCE_FILE)

    params = ImageEdit(
        image=path,
        prompt="smiling face"
    )
    resp: ImagesResponse | None = await get_openai().openai_images.create_image_edit(params)
    assert resp is not None


@pytest.mark.asyncio
async def test_create_image_variation_file_happy_path(set_test_settings):
    SOURCE_FILE="tests/resources/openai/test_images/suit_1.png"

    params = ImageCreateVariation(
        image=SOURCE_FILE
    )
    resp: ImagesResponse | None = await get_openai().openai_images.create_image_variation(params)
    assert resp is not None


# @pytest.mark.asyncio
# async def test_create_image_variation_url_happy_path(set_test_settings):
#     URL="https://oaidalleapiprodscus.blob.core.windows.net/private/org-AfBQ4L7xwO3FSr3DDrHinp6O/user-7x0UdZ4ZsdM1oiWxHqug4Wm2/img-XfNVXa9yxFEOcQmHJPxTqlnP.png?st=2023-12-21T03%3A03%3A36Z&se=2023-12-21T05%3A03%3A36Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-12-20T23%3A06%3A45Z&ske=2023-12-21T23%3A06%3A45Z&sks=b&skv=2021-08-06&sig=jeDJJjq9bDSEZoksaVDEQc22Yoqpag3XrmATZ/VTgXA%3D"
#     params = ImageCreateVariation(
#         image=URL
#     )
#     resp: ImagesResponse | None = await get_openai().openai_images.create_image_variation(params)
#     assert resp is not None
