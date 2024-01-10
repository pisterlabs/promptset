from linkedin_api import Linkedin
import os
import json
import openai
import dotenv

dotenv.load_dotenv(".env")

api = Linkedin(os.getenv("OUTLOOK_EMAIL"), os.getenv("OUTLOOK_PASSWORD"))

import re
a = open("../david network.html", "r")

b = open("../UofT.html", "r")
c = open("../queens.html", "r")
d = open("../McMaster.html", "r")
e = open("../linkedin.html", "r")
values = [a]
others = [b, c, d, e]
# final all linkedin users, in the fore https://www.linkedin.com/in/{username}

users = set()
for f in values:
    matches = re.findall(r"https://www.linkedin.com/in/([A-Za-z0-9_-]+)", f.read())
    users.update(set(matches))

alread_done = set()
for f in others:
    matches = re.findall(r"https://www.linkedin.com/in/([A-Za-z0-9_-]+)", f.read())
    alread_done.update(set(matches))

print(len(alread_done))
print(len(users))
users = users.difference(alread_done)

from tqdm import tqdm
import jsonlines

with jsonlines.open('output_3.jsonl', mode='w') as writer:
    all_users = list(users)
    for profile_username in tqdm(all_users):
        try:
            profile = api.get_profile(profile_username)
            writer.write(profile)
        except:
            print("Error: " + profile_username)
            pass
