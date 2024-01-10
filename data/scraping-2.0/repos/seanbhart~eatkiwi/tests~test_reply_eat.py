
import openai
import pytest
from decouple import config
from farcaster import Warpcast
from farcaster.models import ApiNotificationCastMention, ApiUser, ApiPfp, ApiProfile, Bio, CastContent, ApiCast, Replies, Reactions, Recasts, Watches, ViewerContext2
from eatkiwi.commands.manager import Commands
from eatkiwi.farcaster.reply import reply

bot_fname = config("FARCASTER_FNAME_DEV01")

def set_config():
    openai.api_key = config("OPENAI_KEY")


@pytest.fixture
def commands_instance():
    fcc = Warpcast(config("FARCASTER_MNEMONIC_DEV01"), rotation_duration=1)
    dev_mode = True
    return Commands(fcc, bot_fname, dev_mode)


@pytest.fixture
def test_notification():
    actor = ApiUser(
        fid=13469,
        username='fcdevtest02',
        display_name='dev test account 02',
        registered_at=None,
        pfp=ApiPfp(url='https://i.imgur.com/0LQhBnu.jpg', verified=False),
        profile=ApiProfile(bio=Bio(text='', mentions=[])),
        follower_count=9,
        following_count=0,
        referrer_username=None,
        viewer_context=None
    )

    cast = ApiCast(
        hash='0x13acc996fef8f4c37337b398c516c2bfa726add7',
        thread_hash='0x62adcf85c2641bd88256080b2a0391813b06aeca',
        parent_hash='0x62adcf85c2641bd88256080b2a0391813b06aeca',
        author=actor,
        text=f"{bot_fname} eat",
        timestamp=1688422619000,
        mentions=[],
        attachments=None,
        ancestors=None,
        replies=Replies(count=0),
        reactions=Reactions(count=0),
        recasts=Recasts(count=0, recasters=[]),
        watches=Watches(count=0),
        deleted=None,
        recast=None,
        viewer_context=ViewerContext2(reacted=False, recast=False, watched=False)
    )

    content = CastContent(cast=cast)

    notification = ApiNotificationCastMention(
        type='cast-reply',
        id='0x13acc996fef8f4c37337b398c516c2bfa726add7',
        timestamp=1688422619000,
        actor=actor,
        content=content
    )

    return notification


def test_reply(commands_instance, test_notification):
    set_config()

    # Call the mention function with the test notification
    reply(commands_instance, test_notification)
