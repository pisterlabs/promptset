import os
import openai
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
import requests
from typing import Any, Dict, Optional, TypedDict

from bullshit_bot_bot_bot.middleware import (
    GenericMessage,
    with_messages_processed,
)
from bullshit_bot_bot_bot.utils import print_messages, truncate_middle

# Ask GPT: Summarise this article into three claims of up to 6 words each.
# Save phrases into temporary variables.
# Send searches to Google Fact Check API
# Return top three outputs from Google Fact Check API / link to search results (depending on what’s easier)

CLAIMS_SYSTEM = """
You are a helpful assistant that takes in an article and extracts a set of six or less sets of claim keywords that this article makes.

These keyword sets will be used to search google fact checker. Google fact checked is EXTREMELY sensitive, and. They should be short (~three words), and should consist of the nouns involved in the claim.

Specifically, focus on factual claims and focus on those that seem more likely to be incorrect / open to scrutiny

Output the search terms in the following format
- Search one
- Search two
...
- Search six
""".strip()

CLAIMS_USER_TEMPLATE = """
Here is the article to extract searches from:

\"\"\"
{transcript}
\"\"\"
""".strip()

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]


class Publisher(TypedDict):
    name: str
    site: str


class ClaimReview(TypedDict):
    publisher: Publisher
    url: str
    title: str
    reviewDate: str
    textualRating: str
    languageCode: str


class Claim(TypedDict):
    text: str
    claimant: str
    claimDate: str
    claimReview: list[ClaimReview]


class Response(TypedDict):
    claims: list[Claim] | None


def google_fact_check_api(query: str) -> Response:
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    headers = {"Content-Type": "application/json"}
    params = {
        "query": query,
        "key": GOOGLE_API_KEY,
        "languageCode": "en",
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code} {response.text}")


@with_messages_processed
async def factcheck(
    messages: list[GenericMessage], update: Update, context: ContextTypes.DEFAULT_TYPE
):
    printed_transcript = print_messages(messages)

    print(len(messages))

    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": CLAIMS_SYSTEM},
            {
                "role": "user",
                "content": CLAIMS_USER_TEMPLATE.format(
                    transcript=truncate_middle(
                        printed_transcript, 12000 - len(CLAIMS_USER_TEMPLATE)
                    )
                ),
            },
        ],
    )
    content: str = chat_completion.choices[0].message.content.strip()

    # Parse out each bullet point
    claim_searches = [claim.strip() for claim in content.split("- ")[1:]]

    print(str(claim_searches))

    check_results = [google_fact_check_api(claim) for claim in claim_searches]

    print(str(check_results))

    # claims_str = "\n".join([f"- {claim}" for claim in claims])
    # await context.bot.send_message(
    #     chat_id=update.effective_chat.id, text=f"Found the following claims:\n{claims_str}"
    # )

    claims = [
        claim for response in check_results for claim in response.get("claims", [])
    ]

    if len(claims) == 0:
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text="No fact check results found"
        )

    claim_reviews = [
        claim_review
        for claim in claims
        for claim_review in claim.get("claimReview", [])
    ]
    print(str(claim_reviews))

    claim_review_strs = [
        f"""
Title: {claim_review['title']}
Url: {claim_review['url']}
        """.strip()
        for claim_review in claim_reviews
    ]

    uniq_claim_review_strs = list(set(claim_review_strs))
    claim_review_str = "\n\n".join(uniq_claim_review_strs)

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"Found the following fact check results:\n{claim_review_str}",
    )
