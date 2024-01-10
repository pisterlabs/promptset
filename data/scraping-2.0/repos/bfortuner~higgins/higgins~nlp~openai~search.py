import os
from typing import List

import openai


openai.api_key = os.getenv("OPENAI_API_KEY")


def find_similar_documents(documents, query):
    # Openai /search 
    # Run semantic search over list of strings
    resp = openai.Engine("davinci").search(
        search_model="davinci",
        documents=documents,
        query=query,
        max_rerank=5,
    )
    return resp


PERSONAL_QUESTION_CONTEXT = "Full Name: Stephanie Joy\nBirthday: December 18, 2000\nAge: 21\nEmail address: stephanie123@gmail.com"
PERSONAL_QUESTION_EXAMPLES = [
    ["How old is Stephanie?", "21"],
    ["What is Stephanie's last name?", "Joy"],
    ["When was Stephanie born?", "December 18, 2000"],
]
PERSONAL_QUESTION_DOCUMENTS = [
    "Brendan Fortuner was born on February 10, 1990. He is 31 years old. His email address address is bfortuner@gmail.com. His phone number is 860-459-8424. His Amazon password is MyPassword567. His girlfriend is Jackie First.",
    "David Brewster was born on March 1, 2005. He is 15 years old. His email address is dbrewster@gmail.com. His Amazon password is BD17826. His Facebook username is dboy253.",
]
PERSONAL_QUESTION_FILE_ID = "file-u7jUNn5dIIvV4cVMMeROSyRI"

CHAT_QUESTION_CONTEXT = "text Mom and ask her what she wants to do for dinner tonight. Mom not found in contacts. Who do you mean by Mom? Erin Fortuner."
CHAT_QUESTION_EXAMPLES = [
    ["Which contact does Mom refer to?", "Erin Fortuner"],
    ["What is the default messaging app for Jackie First?", "iMessage"],
    ["What is Colin Fortuner's email address?", "cfortuner@gmail.com"],
]
# Q/A Episodes
CHAT_QUESTION_DOCUMENTS = [
    "Brendan: Tell Dad I'm coming home on WhatsApp. Higgins: Which contact does Dad refer to? Brendan: Bill Fortuner.",
    "Brendan: Email Colin and ask if he wants me to pickup toilet paper. Higgins: Which contact do you mean by Colin? Brendan: Colin Fortuner.",
    "Brendan: Text Jackie First and ask her if she wants to come over tonight. Higgins: Which messaging app should we use? Brendan: iMessage. Higgins: Should we make iMessage default for Jackie? Brendan: Yes.",
    "Brendan: Login to my Github account. Higgins: What is your github username?. Brendan: bfortuner. Higgins: What is your github password? Brendan: brenman90",
]


def ask_question(
    question: str,
    examples_context: str,
    examples: List[str],
    file_id: str = None,
    documents: List[str] = None
):
    # inputs = set([file_id, documents])
    # assert len(inputs) == 2 and None in inputs, "Must provide one of `file_id` or `documents`"
    # Openai /answers
    # Answer user query, doing a 2-stage search -> answer.
    # Pass file_id OR documents, not both
    # file_id: id of the uploaded document
    # documents: List of strings
    resp = openai.Answer.create(
        search_model="ada",
        model="curie",
        question=question,
        file=file_id,
        documents=documents,
        examples_context=examples_context,
        examples=examples,
        max_rerank=5,
        max_tokens=10,
        stop=["\n", "<|endoftext|>"]
    )
    return resp


def upload_document(fpath, purpose="search"):  # or "answers"
    """Upload document to OpenAI servers.

    # List files: openai api files.list
    # Upload files: openai api files.create -f path_to_file -p [answers|search]
    # OAR.upload_document(data_fpath, purpose="answers")
    # print(openai.File.list())

    Args:
        fpath ([type]): [description]
        purpose (str, optional): [description]. Defaults to "search".
    """
    openai.File.create(file=open(fpath), purpose=purpose)


if __name__ == "__main__":
    # resp = ask_question(
    #     question="Who is Brendan dating?",
    #     examples_context=PERSONAL_QUESTION_CONTEXT,
    #     examples=PERSONAL_QUESTION_EXAMPLES,
    #     documents=PERSONAL_QUESTION_DOCUMENTS,
    # )
    # print(resp)

    # resp = ask_question(
    #     question="What is Colin's discord alias?",
    #     examples_context=PERSONAL_QUESTION_CONTEXT,
    #     examples=PERSONAL_QUESTION_EXAMPLES,
    #     file_id=PERSONAL_QUESTION_FILE_ID,
    # )
    # print(resp)

    # resp = ask_question(
    #     question="Which contact do you mean by Colin?",
    #     examples_context=CHAT_QUESTION_CONTEXT,
    #     examples=CHAT_QUESTION_EXAMPLES,
    #     documents=CHAT_QUESTION_DOCUMENTS,
    # )
    # print(resp)

    # resp = ask_question(
    #     question="What is your github password?",
    #     examples_context=CHAT_QUESTION_CONTEXT,
    #     examples=CHAT_QUESTION_EXAMPLES,
    #     documents=CHAT_QUESTION_DOCUMENTS,
    # )
    # print(resp)

    from higgins.utils import jsonl_utils
    documents = jsonl_utils.open_jsonl("data/episodes.jsonl")
    print(documents)
    documents = [d["text"] for d in documents]
    resp = ask_question(
        question="What did you ask Colin for?",   # "Which contact do you mean by Dad?"
        examples_context=CHAT_QUESTION_CONTEXT,
        examples=CHAT_QUESTION_EXAMPLES,
        documents=documents,
    )
    print(resp)
