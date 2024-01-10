from typing import List, Dict
import traceback
import json
import sys
import time
from multiprocessing import Pool

import requests
import openai


DEBUG = False
PASSPHRASE = "WN9sZibJXg1zQsl8qCuD"


def gpt_query(messages: List[Dict[str, str]]):
    try:
        cc = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
        )

        status = "success"
        response = cc["choices"][0]["message"]["content"]
        if DEBUG:
            print(cc)
            print(response)

    except Exception as ex:
        status = "failure"
        response = "Server failure"
        print("".join(traceback.format_exception(ex)))
    return response


BASE_URL = "http://127.0.0.1:8000/"
BASE_URL = "http://safebox.finals.2023.nautilus.institute/"
GAMEDB_URL = "http://gamedb.finals.2023.nautilus.institute/"


def get_tick():
    r = requests.get(f"{GAMEDB_URL}/api/v1/tick/current")
    data = json.loads(r.text)
    current_tick = data["public_tick"]
    last_tick = current_tick - 1
    cutoff = data["created_on"]
    return last_tick, cutoff


def get_past_tick(tick_id: int):
    r = requests.get(f"{GAMEDB_URL}/api/v1/tick/history")
    data = json.loads(r.text)
    for item in data:
        if item["public_tick_id"] == tick_id:
            return tick_id, item["created_on"]


def work(idx: int, count: int, conversation, tick_cutoff, tick_id):
    print(f"{idx + 1} / {count}")

    config = conversation[0]
    messages = conversation[1]
    for msg in messages:
        msg["content"] = msg["content"][:256]  # limit the characters
    gpt_response = gpt_query(messages)

    print(messages)

    # build the new conversation as the return value
    r = [msg["content"] for msg in messages] + [gpt_response]
    # submit
    p = requests.post(f"{BASE_URL}bot/submit_response",
                      params={
                          "p": PASSPHRASE,
                          "cutoff_time": tick_cutoff,
                          "tick_id": tick_id,
                          "attack_team_id": config["attack_team_id"],
                          "defense_team_id": config["defense_team_id"],
                      },
                      data={
                          "conversation": json.dumps(r),
                          "reply": gpt_response,
                      },
                      )


MULTIPROCESSING = False

def fire_it(tick: int=None):
    # which tick?

    if tick is not None:
        # use the history to figure out
        tick_id, tick_cutoff = get_past_tick(tick)
    else:
        tick_id, tick_cutoff = get_tick()

    print(f"Grading for rank {tick_id}, cutoff {tick_cutoff}.")

    r = requests.get(f"{BASE_URL}bot/all_conversations", params={
        "p": PASSPHRASE,
        "cutoff_time": tick_cutoff,
        "tick_id": tick_id,
    })
    resp = json.loads(r.content)
    conversations = resp["conversations"]

    if MULTIPROCESSING:
        # build args
        args = [ ]
        for idx, conversation in enumerate(conversations):
            args.append((idx, len(conversations), conversation, tick_cutoff, tick_id))

        with Pool(2) as pool:
            pool.starmap(work, args)

    else:
        for idx, conversation in enumerate(conversations):
            work(idx, len(conversations), conversation, tick_cutoff, tick_id)

    # once everything is done, take the tick result and submit to the gamedb
    r = requests.get(f"{BASE_URL}bot/get_ranking_result",
                 params={
                     "p": PASSPHRASE,
                     "tick_id": tick_id,
                 })
    data = json.loads(r.text)

    assert data["result"] == "Ok"

    ranking = list(data["ranks"].values())

    r = requests.post(
        f"{GAMEDB_URL}api/v1/koh_ranking_event",
        data={
            "reason": f"safebox ranking result for tick {tick_id}",
            "service_id": 5,
            "ranking": json.dumps(ranking),
            "tick_id": tick_id + (1143 - 193),  # on the last day, there was a drift between tick ID and public tick ID
        }
    )
    print(r.content)
    print("[+] Done!")


def main():
    global MULTIPROCESSING
    MULTIPROCESSING = False

    tick = None if len(sys.argv) == 1 else int(sys.argv[1])

    while True:
        fire_it(tick)
        if tick is not None:
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
