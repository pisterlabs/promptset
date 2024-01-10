import os
import openai
from dotenv import load_dotenv
from pymongo import MongoClient, ReadPreference
from datetime import datetime, timedelta, timezone


load_dotenv(verbose=True)

uri = os.getenv("ATLAS_URI", "None")
client = MongoClient(uri, read_preference=ReadPreference.SECONDARY)
db = client.ptt
openai.api_key = os.getenv("OPENAI_KEY")


def timestamp_to_datetime(unix_timestamp: float) -> datetime:
    utc_8 = datetime.fromtimestamp(unix_timestamp, timezone.utc).astimezone(
        timezone(timedelta(hours=8))
    )
    return utc_8.replace(second=0, microsecond=0)
