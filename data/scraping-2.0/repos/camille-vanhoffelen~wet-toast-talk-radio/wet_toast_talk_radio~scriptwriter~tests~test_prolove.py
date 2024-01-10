import asyncio

import pytest
from guidance.llms import Mock

from wet_toast_talk_radio.media_store import MediaStore, VirtualMediaStore
from wet_toast_talk_radio.media_store.media_store import ShowId
from wet_toast_talk_radio.scriptwriter.prolove import Prolove
from wet_toast_talk_radio.scriptwriter.prolove.genders import Gender
from wet_toast_talk_radio.scriptwriter.prolove.missions import (
    GuestMissions,
    HostMissions,
)
from wet_toast_talk_radio.scriptwriter.prolove.show import Guest


def test_prolove(  # noqa: PLR0913
    guest,
    host_missions,
    guest_missions,
    host_messages,
    fake_llm,
    virtual_media_store,
    show_id,
):
    show = Prolove(
        guest=guest,
        host_missions=host_missions,
        guest_missions=guest_missions,
        host_messages=host_messages,
        llm=fake_llm,
        media_store=virtual_media_store,
    )
    asyncio.run(show.arun(show_id=show_id))
    script_shows = virtual_media_store.list_script_shows()
    assert script_shows == [show_id]


def test_create_prolove(fake_llm, virtual_media_store):
    # smoke test
    Prolove.create(llm=fake_llm, media_store=virtual_media_store)


@pytest.fixture()
def guest() -> Guest:
    return Guest(
        name="Jenny",
        gender=Gender.FEMALE,
        age=30,
        sexual_orientation="heterosexual",
        voice_gender="female",
        trait="excited",
        topic="Am I in love?",
        placeholder_name="Linda",
    )


@pytest.fixture()
def host_missions(guest) -> HostMissions:
    return HostMissions(
        anecdote="I once went on a date",
        k=2,
        lesson="self-love",
        product="book",
        guest_name=guest.placeholder_name,
    )


@pytest.fixture()
def guest_missions() -> GuestMissions:
    return GuestMissions(topic="Am I in love?", k=2)


@pytest.fixture()
def host_messages() -> list[str]:
    return ["Oh, we have a caller!", "What's your problem?"]


@pytest.fixture()
def show_id() -> ShowId:
    return ShowId(show_i=0, date="2021-01-01")


@pytest.fixture()
def fake_llm() -> Mock:
    show = "Welcome to Prolove!"
    return Mock(output=[show])


@pytest.fixture()
def virtual_media_store() -> MediaStore:
    return VirtualMediaStore(load_test_data=False)
