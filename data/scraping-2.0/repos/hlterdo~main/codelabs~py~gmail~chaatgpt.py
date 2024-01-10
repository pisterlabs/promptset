import logging
from gmail_utils import fetch_emails
from inde_email import IndeEmail
import os
import json
from openai_utils import get_openai_embedding
from openai_utils import embedding_similarity
from openai_utils import call_chatgpt_per_email
from openai_utils import CHATGPT_DOESNT_KNOW
from openai_utils import call_chatgpt_for_multiple_answers
from log_level import LOG_LEVEL

# Number of latest emails to fetch from gmail account.
_NUM_EMAILS_TO_READ = 25
# Number of emails to try to answer users questions. Only this many top (based on their similarity)
_NUM_TOP_EMAILS_TO_CHECK_FOR_ANSWER = 15
# Filter that's applied (positives will be kept) when fetching emails.
_EMAIL_FILTER = "from:erenbiri@gmail.com"

# Where the tokens are kept. See the README
_ACCESS_TOKENS_FILE = "tokens.json"

# Where a copy of each email that's found is kept.
_EMAIL_DUMPS_FOLDER = "emails/"

# Set up logging. Modify log_level.py to change the logging level.
logging.basicConfig()
LOG = logging.getLogger("chaatgpt")
LOG.setLevel(LOG_LEVEL)


# *** MAIN STARTS *** #

print(
    f"Trying to fetch {_NUM_EMAILS_TO_READ} emails using the filter \"{_EMAIL_FILTER}\"")

emails = fetch_emails(_ACCESS_TOKENS_FILE, _NUM_EMAILS_TO_READ, _EMAIL_FILTER)
emails_dict = {email.get_email_id(): email for email in emails}

assert len(emails_dict) == len(emails), f"There cannot be duplicate email ids."

print(f"\n\nWill use {len(emails)} emails as the knowledge base.")


LOG.info(f"Writing emails to the folder {_EMAIL_DUMPS_FOLDER}")
for email in emails:
    os.makedirs(os.path.dirname(_EMAIL_DUMPS_FOLDER), exist_ok=True)
    path = os.path.join(_EMAIL_DUMPS_FOLDER, email.get_subject(
    ) + "-" + email.get_email_id() + ".json")
    with open(path, 'w') as f:
        json.dump(email.to_dict(), f, indent='\n')
LOG.info(f"Done writing emails to the folder {_EMAIL_DUMPS_FOLDER}")


LOG.info(f"Calculating embeddings for each email")
for email in emails:
    embedding = get_openai_embedding(email.get_clean_body())
    email.set_embedding(embedding)
LOG.info(f"Done calculating embeddings for each email")

# Continues get questions from the user until they type quit
while True:
    question = input("\n\nEnter a question (type quit to quit): ")
    if question == "quit":
        break

    query_embedding = get_openai_embedding(question)

    # email_id -> similarity to the query.
    query_email_similarities = {}
    for email in emails:
        email_id = email.get_email_id()
        email_embedding = email.get_embedding()
        similarity = embedding_similarity(query_embedding, email_embedding)
        query_email_similarities[email_id] = similarity

    # sort each email based on their similarity to the query.
    sorted_query_email_similarities = dict(
        sorted(query_email_similarities.items(), key=lambda x: x[1], reverse=True))

    print(
        "\n\n\n=== RANKED ORDER OF EMAILS TO THE QUESTION BASED ON THEIR SIMILARITY ===\n\n")
    for email_id, similarity in sorted_query_email_similarities.items():
        subject = emails_dict[email_id].get_subject()
        print(f"{subject}: {similarity}")

    print(
        f"\n\n==== Checking top {_NUM_TOP_EMAILS_TO_CHECK_FOR_ANSWER} top emails to check whether it can answer your question ====\n\n")
    num_emails_checked = 0
    answering_emails = []
    answers = []
    for email_id, similarity in sorted_query_email_similarities.items():
        email = emails_dict[email_id]
        subject = email.get_subject()
        date = email.get_date()
        clean_body = email.get_clean_body()

        chatgpt_answer = call_chatgpt_per_email(clean_body, question)

        print(
            f"Answer from email \"{subject}\" sent on {date}:\n\n{chatgpt_answer}:\n\n\n")

        num_emails_checked += 1
        if (num_emails_checked >= _NUM_TOP_EMAILS_TO_CHECK_FOR_ANSWER):
            break

        if chatgpt_answer != CHATGPT_DOESNT_KNOW:
            answering_emails.append(email)
            answers.append(chatgpt_answer)

    print(
        f"\n\n\nFound {len(answering_emails)} emails that contains an answer to the question.")
    for i in range(len(answering_emails)):
        print(f"{i + 1}. {answering_emails[i].get_subject()}")

    print("\n\n\n=== ***** FINAL ANSWER ***** ===\n\n")

    if len(answering_emails) == 0:
        print(f"We couldn't answer this question using the emails listed above\n\n")
        continue

    final_answer = call_chatgpt_for_multiple_answers(
        answering_emails, answers, question)

    print(final_answer)
