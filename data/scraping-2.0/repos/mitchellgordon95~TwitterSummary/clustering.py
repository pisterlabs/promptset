import openai
import asyncio
from twitter import Thread
import re
import time
from collections import Counter
import pickle

class HashtagsThread:
  def __init__(self, thread, hashtags):
    self.text = thread.text
    self.conversation_id = thread.conversation_id
    self.thread_ids = thread.thread_ids
    self.hashtags = hashtags

class TweetCluster:
  def __init__(self, threads, hashtags=None, summary=None, subclusters=[]):
    self.threads = threads
    self.hashtags = hashtags
    self.summary = summary
    self.subclusters = subclusters or []

  @property
  def num_tweets(self):
    count = len(self.threads)
    if self.subclusters:
      for cluster in self.subclusters:
        count += cluster.num_tweets
    return count

async def with_retries(func, err_return):
  for attempt in range(1, 4):  # 3 attempts with exponential backoff
    try:
      return await func()
    except Exception as e:
      wait_time = 2 ** attempt
      print(f"Error generating summary on attempt {attempt}. Retrying in {wait_time} seconds. Error: {str(e)}")
      time.sleep(wait_time)
  return err_return
  
HASHTAG_PROMPT = """\
TWEET:
{tweet}

Generate 30 possible hashtags that could go with TWEET.

Rules:
If TWEET refers to a location or event, include at least one hashtag containing the name of the event.
If TWEET refers to a specific object or thing, include at least one hashtag containing the name of that thing.
"""

async def add_hashtags(thread):
  # TODO - count tokens
  messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": HASHTAG_PROMPT.format(tweet=thread.text)}
  ]

  async def get_hashtags():
    print("sending request...")
    response = await openai.ChatCompletion.acreate(
      # model="gpt-4",
      model="gpt-3.5-turbo",
      messages=messages
    )
    response_text = response.choices[0].message['content'].strip()
    hashtags = re.findall(r'#\w+', response_text)

    return [h.lower() for h in hashtags]

  hashtags = await with_retries(get_hashtags, [])

  return HashtagsThread(thread, hashtags)


def count_hashtags(threads : HashtagsThread | TweetCluster):
  hashtag_counter = Counter()
  for thread in threads:
    for h in thread.hashtags:
      hashtag_counter[h] += 1
  return hashtag_counter


def pack_cluster(relevant_threads, threads, hashtag):
  # Grab more threads that seem relevant until we hit 7
  all_cluster_hashtags = count_hashtags(relevant_threads)
  pivot_hashtags = set([hashtag])
  while len(relevant_threads) < 7:
    found = False
    for c_hashtag, _ in all_cluster_hashtags.most_common():
      try:
        another_relevant_thread = next(iter([thread for thread in threads if c_hashtag in thread.hashtags]))
      except Exception:
        continue

      found = True
      pivot_hashtags.add(c_hashtag)
      relevant_threads.add(another_relevant_thread)
      threads.remove(another_relevant_thread)
      break

    if not found:
      break

  # Also add hashtags that most threads have, but were not originally
  # used to pivot
  all_cluster_hashtags = count_hashtags(relevant_threads)
  pivot_hashtags.update([h for h, count in all_cluster_hashtags.most_common()
                         if count > len(relevant_threads) / 2])
  return pivot_hashtags


async def cluster_threads(threads):
  threads = await asyncio.gather(*[add_hashtags(thread) for thread in threads])
  # with open('hashtag_threads.pkl', 'wb') as file_:
  #   pickle.dump(threads, file_)
  # with open('hashtag_threads.pkl', 'rb') as file_:
  #   threads = pickle.load(file_)

  hashtag_counter = count_hashtags(threads)

  clusters = []
  threads = set(threads)
  for hashtag, _ in hashtag_counter.most_common():
    relevant_threads = set([thread for thread in threads if hashtag in thread.hashtags])
    if len(relevant_threads) < 8:
      threads = threads - relevant_threads
      # Note: this mutates threads and relevant_threads
      pivot_hashtags = pack_cluster(relevant_threads, threads, hashtag)
      if len(relevant_threads) > 3:
        clusters.append(TweetCluster(relevant_threads, hashtags=pivot_hashtags))
      else:
        threads.update(relevant_threads)

  misc = []
  for thread in threads:
    found = False
    for c in clusters:
      for t in c.threads:
        found = found or thread.conversation_id == t.conversation_id
    if not found:
      misc.append(thread)
  clusters.append(TweetCluster(misc, hashtags=[], summary="misc"))

  return clusters


def meta_cluster(clusters):
  hashtag_counter = count_hashtags(clusters)
  meta_clusters = []
  clusters = set(clusters)
  for hashtag, _ in hashtag_counter.most_common():
    relevant_clusters = set([c for c in clusters if hashtag in c.hashtags])
    clusters -= relevant_clusters
    if len(relevant_clusters) == 1:
      meta_clusters.append(list(relevant_clusters)[0])
    elif len(relevant_clusters) > 1:
      meta_cluster_hashtags = count_hashtags(relevant_clusters)
      meta_cluster_pivot_hashtags = [h for h, count in meta_cluster_hashtags.most_common()
                            if count > len(relevant_clusters) / 2]
      meta_clusters.append(TweetCluster([], hashtags=meta_cluster_pivot_hashtags, subclusters=relevant_clusters))

  return meta_clusters
