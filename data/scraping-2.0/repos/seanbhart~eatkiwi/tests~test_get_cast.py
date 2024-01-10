import logging
import openai
import pytest
from decouple import config
from farcaster import Warpcast
from farcaster.models import ApiUser, ApiCast
from eatkiwi.commands.eat import Eat


def set_config():
    openai.api_key = config("OPENAI_KEY")


@pytest.fixture
def fcc_instance():
    fcc = Warpcast(config("FARCASTER_MNEMONIC_DEV01"), rotation_duration=1)
    return fcc


def test_get_cast(fcc_instance):
    set_config()

    cast = fcc_instance.get_cast("0xa79007d0b63a7d87cd085363439e4d637fa1fd7d")
    logging.info(f"Cast: {cast}")
    logging.info(f"Cast parent_source: {cast.cast.parent_source}")
