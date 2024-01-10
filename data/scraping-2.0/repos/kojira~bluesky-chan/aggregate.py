from atproto import Client, models
import atproto
import os
from dotenv import load_dotenv
import openai
import time
import sqlite3
from tqdm import tqdm
import inspect

connection = sqlite3.connect("bluesky.db")
cur = connection.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users
  (id INTEGER PRIMARY KEY AUTOINCREMENT,
   did TEXT,
   handle TEXT,
   displayName TEXT,
   indexed_at DATETIME,
   updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
   )
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS posts
  (id INTEGER PRIMARY KEY AUTOINCREMENT,
   did TEXT,
   count INTEGER,
   created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
   )
""")
cur.execute("""
CREATE TRIGGER IF NOT EXISTS update_timestamp
AFTER UPDATE ON users
FOR EACH ROW
BEGIN
    UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;
""")
connection.commit()

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

username = os.environ.get("BOT_HANDLE")
password = os.environ.get("BOT_PASSWORD")

client = Client()


def login(username, password):
  profile = client.login(username, password)
  return profile


def get_follow(actor, cursor):
  response = None
  try:
    response = client.bsky.graph.get_follows(
        {"actor": actor, "cursor": cursor, "limit": 100}
    )
  except atproto.exceptions.RequestException as e:
    print(actor)
    print(e)
    if e.args[0].content.error == 'ExpiredToken':
      login(username, password)
      try:
        response = get_follow(actor, cursor)
      except atproto.exceptions.RequestException as e:
        print(e)

  return response


def get_profile_detail(actor):
  profile = None
  while True and profile is None:
    try:
      profile = client.bsky.actor.get_profile({"actor": actor})
    except atproto.exceptions.RequestException as e:
      print(actor)
      print(e)
      if e.args[0].content.error == 'ExpiredToken':
        login(username, password)
        try:
          profile = get_profile_detail(actor)
        except atproto.exceptions.RequestException as e:
          print(e)
      break
    except atproto.exceptions.InvokeTimeoutError as e:
      print(e)
      break

  return profile


profile = login(username, password)

actor = "kojira.bsky.social"
# actor = "masonicursa.bsky.social"
# actor = "peelingoranges.bsky.social"
cursor = None

member_index = 0

cur.execute("SELECT * FROM users ORDER BY RANDOM()")
rows = cur.fetchall()
columns = [column[0] for column in cur.description]
all_members = [dict(zip(columns, row)) for row in rows]

prev_actor = None

while True:
  if actor != prev_actor:
    profile = get_profile_detail(actor)
    if profile:
      print(f"{profile.displayName}@{profile.handle}")
      sql = "INSERT INTO posts (did, count) VALUES (?, ?)"
      cur.execute(sql, (profile.did, profile.postsCount, ))
      connection.commit()

    response = get_follow(actor, cursor)
    if response:
      follows = response.follows
      members = [{"did": follow.did,
                  "handle": follow.handle,
                  "displayName": follow.displayName,
                  "indexed_at": follow.indexedAt}
                 for follow in follows
                 if all(item["did"] != follow.did for item in all_members)
                 ]
      print(len(members))
      all_members.extend(members)
      sql = """
        INSERT INTO users (did, handle, displayName, indexed_at)
                  VALUES (:did, :handle, :displayName, :indexed_at)
      """
      cur.executemany(sql, members)

      connection.commit()
      prev_actor = actor
      if response.cursor is None:
        cursor = None
        if member_index > (len(all_members) - 1):
          member_index = 0
        actor = all_members[member_index]["handle"]
        member_index += 1
        print("all:", len(all_members))
      else:
        # print("cursor:", response.cursor)
        cursor = response.cursor
    else:
      if member_index > (len(all_members) - 1):
        member_index = 0
      actor = all_members[member_index]["handle"]
      member_index += 1
      print("all:", len(all_members))
  else:
    if member_index > (len(all_members) - 1):
      member_index = 0
    actor = all_members[member_index]["handle"]
    member_index += 1
    print("all:", len(all_members))

  time.sleep(0.05)


# params = models.AppBskyActorGetProfile.Params("kojira.bsky.social")
# print(follow)
# print(client.com.atproto.identity.resolve_handle(params))
# client.send_post(text="Hello World!")
