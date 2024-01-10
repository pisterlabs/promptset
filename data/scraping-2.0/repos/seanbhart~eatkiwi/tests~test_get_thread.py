import logging
import openai
import pytest
from decouple import config
from farcaster import Warpcast
from farcaster.models import ApiUser, ApiCast
from eatkiwi.commands.eat import Eat


def set_config():
    openai.api_key = config("OPENAI_KEY")
    logging.basicConfig(level=logging.INFO)


@pytest.fixture
def fcc_instance():
    fcc = Warpcast(config("FARCASTER_MNEMONIC_DEV01"), rotation_duration=1)
    return fcc


def test_get_all_casts_in_thread(fcc_instance):
    set_config()

    # Call the cast method
    casts = fcc_instance.get_all_casts_in_thread("0xa79007d0b63a7d87cd085363439e4d637fa1fd7d")
    logging.info(f"Casts: {casts}")
    # for cast in casts:
    #     logging.info(f"Cast: {cast}")
