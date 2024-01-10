import asyncio

import pytest
from guidance.llms import Mock

from wet_toast_talk_radio.media_store import MediaStore, VirtualMediaStore
from wet_toast_talk_radio.media_store.media_store import ShowId
from wet_toast_talk_radio.scriptwriter.adverts import Advert


def test_advert(
    product_description,
    strategies,
    fake_llm,
    virtual_media_store,
    show_id,
):
    show = Advert(
        product_description=product_description,
        strategies=strategies,
        llm=fake_llm,
        media_store=virtual_media_store,
    )
    asyncio.run(show.arun(show_id=show_id))
    script_shows = virtual_media_store.list_script_shows()
    assert script_shows == [show_id]


def test_create_advert(fake_llm, virtual_media_store):
    # smoke test
    Advert.create(llm=fake_llm, media_store=virtual_media_store)


@pytest.fixture()
def product_description() -> str:
    return "The funniest book ever made"


@pytest.fixture()
def strategies() -> list[str]:
    return ["Invent a statistic", "Make a joke", "Make a discount"]


@pytest.fixture()
def show_id() -> ShowId:
    return ShowId(show_i=0, date="2021-01-01")


@pytest.fixture()
def fake_llm() -> Mock:
    show = "And now for a word from our sponsors."
    return Mock(output=[show])


@pytest.fixture()
def virtual_media_store() -> MediaStore:
    return VirtualMediaStore(load_test_data=False)
