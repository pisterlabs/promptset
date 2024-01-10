import os
import sys

import openai
import pytest
from dotenv import load_dotenv
from farcaster import Warpcast
from google.cloud import translate

from ditti.commands.command_manager import Commands
from supabase import Client, create_client

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../ditti")))

load_dotenv()
access_token = os.getenv("FARC_SECRET")
BOT_USERNAME = os.getenv("USERNAME")
url = os.environ.get("SUPABASE_URL", "VALUE NOT SET")
key = os.environ.get("SUPABASE_KEY", "VALUE NOT SET")

supabase: Client = create_client(url, key)
gtc = translate.TranslationServiceClient()
openai.api_key = os.getenv("OPENAI_KEY")


@pytest.fixture(scope="session")
def fcc():
    """
    A fixture that returns a Warpcast instance.
    """
    return Warpcast(access_token=access_token)


@pytest.fixture(scope="session")
def commands(fcc):
    """
    A fixture that returns a Commands instance.
    """
    return Commands(
        fcc=fcc,
        supabase=supabase,
        gtc=gtc,
        bot_username=BOT_USERNAME,
    )
