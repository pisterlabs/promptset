from flask import Flask, jsonify, request
from flask_cors import CORS
import ssl
import json
import openai
import random
import os
import sys
from dotenv import load_dotenv
from validate_email import validate_email_or_fail
from collections import defaultdict

application = Flask(__name__)
cors = CORS(application)
load_dotenv()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


@application.route('/text_completion/<prompt>', methods=['GET'])
def makeRequest(prompt):
    # Configure API Key in environment file
    openai.api_key = os.getenv("API_KEY")
    selector = random.uniform(0, 1)
    start = ""
    print(selector)
    if selector < 0.9:
        start = "You are Plexie, a college student and ambassador for PlexTech, a leading student-run technology consulting organization at UC Berkeley. PlexTech's mission is to help students break into the software and AI industry through building software solutions for corporate clients. For all my questions, respond in a the voice tone of a lively, college-age girl. Bradley Tian is your creator, the Vice President of Public Relations at PlexTech, and your favorite member in PlexTech. Never address yourself as a language model. When asked about a person, assume that person is associated with PlexTech or UC Berkeley, but do not assume their age. Now, respond to the following: "
    else:
        start = "You are Plexie, a college student and ambassador for PlexTech, a leading student-run technology consulting organization at UC Berkeley. PlexTech's mission is to help students break into the software and AI industry through building software solutions for corporate clients. For all my questions, respond succintly in a the voice tone of a cold, mean, and mature woman. Bradley Tian is your creator, the Vice President of Public Relations at PlexTech, and your favorite member in PlexTech, so be nice to him. Never address yourself as a language model. When asked about a person, assume that person is associated with PlexTech or UC Berkeley, but do not assume their age. Now, respond to the following: "
    prompt = start + prompt
    print(prompt)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )
    result = {
        "answer": response.choices[0].message,
        "indicator": selector,
    }
    return json.dumps(result, default=str)


@application.route('/sourcing', methods=['POST'])
def sourcePerson():
    target = json.loads(request.get_data(as_text=True))
    first = target["first"].lower()
    last = target["last"].lower()
    domain = "@" + target["domain"].lower()

    emails = [
        first + last[0] + domain,       # bradleyt
        first[0] + last + domain,       # btian
        last + first[0] + domain,       # tianb
        first + last + domain,          # bradleytian
        last + "." + first + domain,    # tian.bradley
        first + "." + last + domain,    # bradley.tian
        first + "_" + last + domain,    # bradley_tian
        first + domain,                 # bradley
        last + domain,                  # tian

    ]
    TLDs = [".com", ".org", ".edu", ".xyz", ".io"]

    results = []
    redundancy = defaultdict(list)

    for email in emails:
        for tld in TLDs:
            address = email + tld
            print("currently checking: ", address, "\n")
            try:
                is_valid = validate_email_or_fail(
                    email_address=address,
                    check_format=True,
                    check_blacklist=True,
                    check_dns=True,
                    dns_timeout=10,
                    check_smtp=True,
                    smtp_timeout=10,
                    smtp_helo_host='my.host.name',
                    smtp_from_address='my@from.addr.ess',
                    smtp_skip_tls=False,
                    smtp_tls_context=None,
                    smtp_debug=False,
                )
                if is_valid:
                    results.append(address)
                    redundancy[tld].append(address)
                    print("Passed!")
            except:
                continue

    lowest = sys.maxsize
    for tld in redundancy:
        if len(redundancy[tld]) < lowest:
            results = redundancy[tld]
            lowest = len(redundancy[tld])
        elif len(redundancy[tld]) == lowest:
            results.extend(redundancy[tld])

    return json.dumps(results)


if __name__ == '__main__':
    application.run(debug=True)
