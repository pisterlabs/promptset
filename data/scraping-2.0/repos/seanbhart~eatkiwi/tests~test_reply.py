
import openai
import pytest
from decouple import config
from farcaster import Warpcast
from farcaster.models import ApiNotificationCastMention, ApiUser, ApiPfp, ApiProfile, Bio, CastContent, ApiCast, Replies, Reactions, Recasts, Watches, ViewerContext2
from eatkiwi.commands.manager import Commands
from eatkiwi.farcaster.reply import reply


def set_config():
    openai.api_key = config("OPENAI_KEY")


@pytest.fixture
def commands_instance():
    fcc = Warpcast(config("FARCASTER_MNEMONIC_DEV01"), rotation_duration=1)
    bot_fname = config("FARCASTER_FNAME_DEV01")
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

    text = '\n'.join([
        # "@fcdevtest01 ",
        # "eat ",
        # "in the style of ",
        # "Hacker News",
        # "magical realism, where fantastical elements blend seamlessly with reality",
        # "stream of consciousness, where the character’s thoughts and emotions flow uninterrupted onto the page",
        "epistolary fiction, where the story is told through a series of letters, diary entries, or other documents",
        # "flash fiction, where the entire story is condensed into a few hundred words or less",
        # "experimental literature, where a traditional narrative structure is abandoned in favor of unconventional forms and techniques",
        # "metafiction, where the story acknowledges its own status as a work of fiction",
        # "noir fiction, where the protagonist is a cynical and hard-boiled detective navigating a corrupt world",
        # "gothic literature, where dark, supernatural elements are woven into a brooding and atmospheric tale",
        # "historical fiction, where the story is set in a specific time period and strives for historical accuracy",
        # "bildungsroman, where the story follows the protagonist’s coming-of-age and personal growth",
        # "Scooby-Doo voice",
        # " https://www.theblock.co/post/232983/republican-draft-bill-would-create-new-definition-of-decentralized-network",
    ])

    cast = ApiCast(
        hash='0x143776e0b11ac03f010dd2e6657f2bbf583e893a',
        thread_hash='0x3bcd07ecc469d2e70c1cb09ebf29b49e884169b9',
        parent_hash='0x3bcd07ecc469d2e70c1cb09ebf29b49e884169b9',
        author=actor,
        text=text,
        timestamp=1685748412000,
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
        id='0x143776e0b11ac03f010dd2e6657f2bbf583e893a',
        timestamp=1685748412000,
        actor=actor,
        content=content
    )

    return notification


def test_reply(commands_instance, test_notification):
    set_config()

    # Call the mention function with the test notification
    reply(commands_instance, test_notification)
