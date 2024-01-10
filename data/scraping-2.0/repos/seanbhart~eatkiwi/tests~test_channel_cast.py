
import openai
import pytest
from decouple import config
from farcaster import Warpcast
from farcaster.models import ApiUser, ApiCast, ParentSource


def set_config():
    openai.api_key = config("OPENAI_KEY")


@pytest.fixture
def channel_instance():
    fcc = Warpcast(config("FARCASTER_MNEMONIC_DEV01"), rotation_duration=1)
    return fcc


def test_cast_success(channel_instance):
    set_config()

    # Create Cast objects
    # cast_to_eat = ApiCast(
    #     hash="cast_hash",
    #     thread_hash=None,
    #     parent_hash=None,
    #     author=ApiUser(fid=1, profile={"bio": {"text": "value", "mentions": ["mention"]}}, follower_count=10, following_count=5),
    #     parent_author=None,
    #     parent_source=None,
    #     text="https://www.theblock.co/post/232754/multichain-team-says-it-cant-contact-ceo-amid-protocol-problems",
    #     timestamp=1,
    #     mentions=None,
    #     attachments=None,
    #     embeds=None,
    #     ancestors=None,
    #     replies={"count": 0},
    #     reactions={"count": 0},
    #     recasts={"count": 0},
    #     watches={"count": 0},
    #     deleted=None,
    #     recast=None,
    #     viewer_context=None
    # )

    # Create the ParentSource object
    parent_source = ParentSource(
        type="url",
        url="chain://eip155:1/erc721:0x8edceb20795ac2b93ab8635af77e96cae123d045",
    )
    cast_for_channel = ApiCast(
        parent_source=parent_source,
        text="for purpler",
    )

    # cast_content = fcc.post_cast(f"🥝 {title}\n{link}")
    channel_instance.post_cast(cast_for_channel)
