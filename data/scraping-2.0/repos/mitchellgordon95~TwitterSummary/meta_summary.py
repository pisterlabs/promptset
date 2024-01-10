import openai
from twitter import Thread
import asyncio
from clustering import with_retries, TweetCluster
import re
import time
from collections import Counter
import pickle

META_SUMMARY_PROMPT="""\
TWEET_SUMMARIES
\"\"\"
{summaries}
\"\"\"

What common theme unites all these tweets? Rules:

- The theme must begin with "{num_tweets} tweets are about"
- The theme must be no more than 1 sentence.
- The theme must be discussed in a majority of the tweets.

Think out loud, then state the topic prefixed with the TOPIC label."""

RESUMMARY_PROMPT = """\
TWEETS:
\"\"\"
{tweets_text}
\"\"\"

What topic do all TWEETS have in common? Rules:

- The topic must be no more than 1 sentence.
- The topic must be discussed in a majority of the tweets.
- The topic must be related to {hashtags}
- The topic must begin with "{num_cluster_tweets} tweets are about {cluster_summary}.  More specifically, {num_tweets} are about"

Do not think. Just say the topic and only the topic."""


async def resummarize(cluster):
  """Given a meta-cluster, resummarize the subclusters to be more specific."""
  async def resummarize_subcluster(subcluster):
    tweets_text = "\n\n".join([thread.text for thread in subcluster.threads])
    messages = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": RESUMMARY_PROMPT.format(
        tweets_text=tweets_text,
        num_tweets=subcluster.num_tweets,
        num_cluster_tweets=cluster.num_tweets,
        cluster_summary=cluster.summary,
        hashtags=" ".join(subcluster.hashtags)
      )}
    ]

    async def get_summary():
      print("sending request...")
      response = await openai.ChatCompletion.acreate(
        model="gpt-4",
        # model="gpt-3.5-turbo",
        messages=messages
      )
      return response.choices[0].message['content'].strip()

    response_text = await with_retries(get_summary, "API error")
    try:
      summary = response_text.strip('"')
      _, summary = summary.split('specifically,', 1)
      _, summary = summary.split('about', 1)
    except:
      summary = f'Error parsing model output: {response_text}'
    return TweetCluster(subcluster.threads, hashtags=subcluster.hashtags, summary=summary, subclusters=subcluster.subclusters) 

  subclusters = await asyncio.gather(*[resummarize_subcluster(c) for c in cluster.subclusters])
  return TweetCluster(cluster.threads, hashtags=cluster.hashtags, summary=cluster.summary, subclusters=subclusters)



async def generate_meta_summary(cluster):
  if cluster.summary:
    return cluster

  summaries = "\n\n".join([c.summary for c in cluster.subclusters])
  messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": META_SUMMARY_PROMPT.format(
      summaries=summaries,
      num_tweets=cluster.num_tweets,
    )}
  ]

  async def get_summary():
    print("sending request...")
    response = await openai.ChatCompletion.acreate(
      model="gpt-4",
      # model="gpt-3.5-turbo",
      messages=messages
    )
    return response.choices[0].message['content'].strip()

  response_text = await with_retries(get_summary, "API error")

  try:
    lines = response_text.split("\n")
    summary = None
    for line in lines:
      if "TOPIC" in line:
        summary = line[len("TOPIC")+1:]

    summary = summary.strip('"')
    _, summary = summary.split('about', 1)
  except:
    summary = f"Error parsing model output: {response_text}"

  out = TweetCluster(cluster.threads, hashtags=cluster.hashtags, summary=summary, subclusters=cluster.subclusters)
  return await resummarize(out)


async def meta_summarize(clusters):
  clusters = await asyncio.gather(*[generate_meta_summary(cluster) for cluster in clusters])
  # with open('meta_summaries.pkl', 'wb') as file_:
  #   pickle.dump(clusters, file_)
  # with open('meta_summaries.pkl', 'rb') as file_:
  #   clusters = pickle.load(file_)

  return clusters
