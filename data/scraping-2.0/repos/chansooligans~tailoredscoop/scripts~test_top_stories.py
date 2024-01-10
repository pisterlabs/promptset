#!/usr/bin/python
from IPython import get_ipython

if get_ipython() is not None:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")
import asyncio
import multiprocessing

import openai
from transformers import pipeline

from tailoredscoop import api, config
from tailoredscoop.db.init import SetupMongoDB
from tailoredscoop.news import newsapi_with_google_kw, users
from tailoredscoop.utils import RecipientList

# import nest_asyncio
# nest_asyncio.apply()

# %% [markdown]
"""
Configuration
"""

# %%
secrets = config.setup()
openai.api_key = secrets["openai"]["api_key"]

num_cpus = multiprocessing.cpu_count()
print("Number of CPUs: ", num_cpus)

newsapi = newsapi_with_google_kw.NewsAPI(api_key=secrets["newsapi"]["api_key"])

mongo_client = SetupMongoDB(mongo_url=secrets["mongodb"]["url"]).setup_mongodb()
db = mongo_client.db1

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sender = api.EmailSummary(news_downloader=newsapi, db=db, summarizer=summarizer)

# %% [markdown]
"""
Get Recipient List
"""

# %%
df_users = RecipientList(db=db).filter_sent(
    users.Users().get_range(start=int(secrets["start"]))
)
df_users = df_users.loc[df_users["email"].str.contains("chansoosong")].copy()

# %% [markdown]
"""
Send Emails
"""

# %%
chunk_size = 5
df_list = [df_users[i : i + chunk_size] for i in range(0, len(df_users), chunk_size)]

for chunk in df_list:
    asyncio.run(sender.send(subscribed_users=chunk))

# %%
