import os
import time
from typing import Any

import openai

from jarvis.nlp import nlp_utils

from . import caching
from . import completion_utils


openai.api_key = os.getenv("OPENAI_API_KEY")


def send_message_completion(cmd: str, engine="davinci", cache: Any = None):
    prompt = f"""Convert the following text into commands:

    Q: Text mom I love her
    A: `SendMessage` PARAMS to=>mom ### body=>I Love her ### application=>??? <<END>>
    Q: text message steve and ask if he's coming to the meeting
    A: `SendMessage` PARAMS to=>steve ### body=>are you coming to the meeting? ### application=>??? <<END>>
    Q: msg Jackie and let her know I'll be home by 10 tonight
    A: `SendMessage` PARAMS to=>Jackie ### body=>I'll be home by 10pm ### application=>??? <<END>>
    Q: text Colin on Facebook Messenger and ask him if he's free for tennis tomorrow
    A: `SendMessage` PARAMS to=>Colin ### body=>Are you free for tennis tomorrow? ### application=>Facebook Messenger <<END>>
    Q: Want to hang out tonight?
    A: `SendMessage` PARAMS to=>??? ### body=>Want to hang out tonight? ### application=>??? <<END>>
    Q: Reply to Sam Fortuner on WhatsApp
    A: `SendMessage` PARAMS to=>Sam Fortuner ### body=>??? ### application=>WhatsApp <<END>>
    Q: slack Sean Bean and tell him I'm running late to the meeting
    A: `SendMessage` PARAMS to=>Sean Bean ### body=>Hey, running late to our meeting ### application=>Slack <<END>>
    Q: email David
    A: `SendMessage` PARAMS to=>David ### body=>??? ### application=>email <<END>>
    Q: Let Hari know I just pushed my latest changes to the github repo
    A: `SendMessage` PARAMS to=>Hari ### body=>I pushed my latest changes to the repo ### application=>??? <<END>>
    Q: tell Dad I'll see him next month
    A: `SendMessage` PARAMS to=>Dad ### body=>See you next month ### application=>??? <<END>>
    Q: Reply Sounds fun!
    A: `SendMessage` PARAMS to=>??? ### body=>Sounds fun! ### application=>??? <<END>>
    Q: {cmd}
    A:"""
    cache = cache if cache is not None else caching.get_default_cache()
    cache_key = nlp_utils.hash_normalized_text(prompt)
    if cache_key not in cache:
        start = time.time()
        response = openai.Completion.create(
            engine=engine,
            model=None,
            prompt=prompt,
            temperature=0.2,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.2,
            presence_penalty=0.0,
            stop=["<<END>>"],
        )
        # print(f"Time: {time.time() - start:.2f}")
        answer = response["choices"][0]["text"].strip("Q:").strip()
        cache.add(
            key=cache_key,
            value={
                "cmd": cmd,
                "answer": answer,
                "response": response
            }
        )
    else:
        answer = cache[cache_key]["answer"]
        response = cache[cache_key]["response"]

    return answer


if __name__ == "__main__":
    # Send message completions
    examples = [
        "message Liam Briggs and see if he wants to get together",
        "send an email to Xin letting him know I'm leaving Cruise soon",
        "whatsapp Kabir how are you doing?",
        "This is something isn't it",
        "Can you ping Joe Boring and say thanks",
        "msg Stew on Slack are you coming to Burning man?",
        "text Colin on iMessage and see if he's still going to the store",
    ]
    for text in examples:
        answer = send_message_completion(text)
        intent = completion_utils.convert_string_to_action_chain(answer)
        print(f"Q: {text}\nA: {answer}\nI: {intent}")
