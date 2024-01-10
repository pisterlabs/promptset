#!/usr/bin/python
import asyncio
import logging

import nest_asyncio
import openai
from transformers import pipeline

from tailoredscoop import api, config, utils
from tailoredscoop.db.init import SetupMongoDB
from tailoredscoop.today_story import MySQL

# %% [markdown]
"""
Configuration
"""

# %%
# nest_asyncio.apply()
utils.Logger().setup_logger()
logger = logging.getLogger("tailoredscoops.testing")

secrets = config.setup()
openai.api_key = secrets["openai"]["api_key"]

newsapi = api.NewsAPI(api_key=secrets["newsapi"]["api_key"])

mongo_client = SetupMongoDB(mongo_url=secrets["mongodb"]["url"]).setup_mongodb()
db = mongo_client.db1

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sender = api.EmailSummary(news_downloader=newsapi, db=db, summarizer=summarizer)

# %% [markdown]
"""
Get Summary
"""

# %%
kw = "us,business"
summary_id = sender.summary_hash(kw=kw)
summary = db.summaries.find_one({"summary_id": summary_id})

if summary:
    print("Summary already loaded. Exit.")

else:
    summary = asyncio.run(
        sender.create_summary(
            email="today_story", news_downloader=newsapi, summary_id=summary_id, kw=kw
        )
    )

    sources = []
    for url, headline in zip(summary["encoded_urls"], summary["titles"]):
        sources.append(f"""- <a href="{url}">{headline}</a>""")

    summary["summary"] += "\n\nSources:\n" + "\n".join(sources)

    # cache subject for emails
    subject = asyncio.run(
        sender.get_subject(plain_text_content=summary["summary"], summary_id=summary_id)
    )

    # %% [markdown]
    """
    Update
    """

    # %%
    updater = MySQL(secrets=secrets)

    updater.update(content=sender.plain_text_to_html(summary["summary"], no_head=True))

    # %%
