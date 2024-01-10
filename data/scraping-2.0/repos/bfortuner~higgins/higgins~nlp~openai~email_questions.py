import copy
import math
import os
import pprint
import random
from re import U
from typing import Any, Dict, List

import openai

from higgins.automation.email import email_utils
from higgins.nlp import nlp_utils

from . import caching


openai.api_key = os.getenv("OPENAI_API_KEY")

tokenizer = nlp_utils.get_tokenizer()

pp = pprint.PrettyPrinter(indent=2)


def build_email_question_completion_prompt(
    user_email: Dict,
    user_question: str,
    prompt_emails: List[Dict],
    task_description: str = None,
) -> str:
    prompt = ""
    if task_description is not None:
        prompt += f"{task_description}"

    for example in prompt_emails:
        prompt += f"\n\nEMAIL\n{example['plain']}\n"
        prompt += "\nQUESTIONS"

        for question, answer in example["model_labels"]["questions"]:
            prompt += f"\nQ: {question}"
            prompt += f"\nA: {answer} <<END>>"

    prompt += "\n\nEMAIL\n{data}\n".format(data=user_email["plain"])
    prompt += "\nQUESTIONS"
    prompt += "\nQ: {question}".format(question=user_question)
    prompt += "\nA:"
    return prompt


def email_question_completion(
    user_email: Dict,
    user_question: str,
    prompt_emails: List[Dict],
    completion_tokens: int = 30,
    task_description: str = "Answer questions about the following emails",
    engine="davinci",
    cache: Any = None,
):
    num_tokens = 100000
    i = 0
    while num_tokens > 2040 - completion_tokens:
        prompt = build_email_question_completion_prompt(
            user_email=user_email,
            user_question=user_question,
            prompt_emails=prompt_emails[i:],
        )
        num_tokens = nlp_utils.get_num_tokens(prompt, tokenizer)
        i += 1
        print(f"prompt tokens: {num_tokens}")
    print(prompt)
    cache = cache if cache is not None else caching.get_default_cache()
    cache_key = nlp_utils.hash_normalized_text(prompt)
    if cache_key not in cache:
        response = openai.Completion.create(
            engine=engine,
            model=None,
            prompt=prompt,
            temperature=0.1,
            max_tokens=completion_tokens,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["<<END>>"],
        )
        answer = response["choices"][0]["text"].strip()
        cache.add(
            key=cache_key,
            value={
                "question": user_question,
                "data": user_email,
                "answer": answer,
                "response": response,
            },
        )
    else:
        answer = cache[cache_key]["answer"]
        response = cache[cache_key]["response"]

    return answer


def make_email_false_positive_questions(emails: List[Dict], questions: List[str]):
    """Create new false positive emails with questions that have ??? answer."""
    emails = copy.deepcopy(emails)
    shuffled = random.sample(questions, len(questions))
    for fp in emails:
        fp["model_labels"]["questions"] = [(shuffled[0], "???")]
    return emails


def make_train_test_split(examples: List[Dict], sort_key: str, train_pct: float = 0.7):
    examples = sorted(examples, key=lambda dct: dct[sort_key])
    num_examples = len(examples)
    train_count = math.floor(num_examples * train_pct)
    return examples[:train_count], examples[train_count:]


def extract_existing_train_test_split(examples: List[Dict]):
    train = []
    test = []
    for example in examples:
        if example["model_labels"].get("split") == "train":
            train.append(example)
        else:
            test.append(example)
    return train, test


EXAMPLE_EMAIL_DOCUMENT = """<body>From: Brendan Fortuner <brendan.fortuner@getcruise.com>
Date: 2021-09-08 16:11:36-07:00
Subject: Coffee

Hey Joe,
Would you like to grab coffee next week? I'm free Thursday and Friday.
Looking forward to connecting soon!
Brendan

--
Brendan Fortuner
Software Engineer, ML Platform
GM Cruise LLC 
brendan.fortuner@getcruise.com
</body>
"""

EXAMPLE_EMAIL_QUESTIONS = [
    # ["Who wrote the email?", "Brendan Fortuner <brendan.fortuner@getcruise.com>"],
    # ["When was the email sent?", "2021-09-08 16:11:36-07:00"],
    ["What is Brendan's job?", "Software Engineer"],
    ["What company does he work at?", "Software Engineer"],
    # ["What is the subject?", "Coffee"],
    ["When is Brendan free?", "Thursday and Friday"],
]


def email_question_answers(
    question: str,
    example_email: str = EXAMPLE_EMAIL_DOCUMENT,
    example_questions: List[List[str]] = EXAMPLE_EMAIL_QUESTIONS,
    documents: List[str] = None,
    file_id: str = None,
):
    """Answer question over multiple emails or emails snippets using answers/ API.

    Performs a 2-stage semantic search -> answer. If documents and file_id are None,
    the model will answer the question based on the content of the example_email.

    Args:
        question: User question
        file_id: id of the uploaded document
        documents: List of emails or snippets
    """
    resp = openai.Answer.create(
        search_model="ada",
        model="davinci",
        question=question,
        file=file_id,
        documents=documents,
        examples_context=example_email,
        examples=example_questions,
        temperature=0.1,
        max_rerank=3,
        max_tokens=30,
        stop=["\n", "<|endoftext|>"],
    )
    return resp


def rank_strings(
    query: str, documents: List[str], engine: str = "davinci"
) -> List[Dict]:
    resp = openai.Engine("davinci").search(documents=documents, query=query)
    docs = sorted(resp["data"], key=lambda x: x["score"], reverse=True)
    return docs


def generate_email_question_context(email: Dict):
    document = email_utils.get_email_body_extended(email)
    document = email_utils.remove_whitespace(document)
    questions = [
        ["Who sent the email?", email["sender"]],
        ["What is the subject?", email["subject"]],
        ["When was it sent?", email["date"]],
    ]
    return document, questions


def generate_email_question_context_from_labeled_email(
    email: Dict, plain_body: bool = True
):
    if plain_body:
        document = email_utils.get_email_body_extended(email)
    else:
        document = email["html"]
    document = email_utils.remove_whitespace(document)
    questions = email["model_labels"]["questions"]
    return document, questions


def get_email_chunks(email, tokens_per_chunk, plain_body=True):
    if plain_body:
        document = email_utils.get_email_body_extended(email)
    else:
        document = email["html"]
    chunks = create_email_chunks(document, tokens_per_chunk)
    return chunks


def test_email_question_answers():
    use_plain_body = True
    # Generate an example
    example_email_id = "365b47046e7027df274d0a26de33461f99c4cb054e47a6aaa4a6bc791e263585"  # Amazon payment
    example_email = email_utils.load_email(example_email_id)
    (
        example_document,
        example_questions,
    ) = generate_email_question_context_from_labeled_email(
        example_email, plain_body=use_plain_body
    )
    print(example_questions)
    example_num_tokens = nlp_utils.get_num_tokens(example_document, tokenizer)
    print(f"num tokens in example {example_num_tokens}")
    # Load the email
    email_id = "0f84cbaf27e24d6d3eb96b734e8f8c776a8fc7250ed479ba252d47a9fc56fcc2"

    # Southwest email, forwarded from jackie, 1 flight -- Works well!
    email_id = "033d3dee30b0a8969e551334ef59543d430807ae78bd8c569b66d269e821d31c"
    email = email_utils.load_email(email_id)
    documents = get_email_chunks(email, 300, use_plain_body)
    for doc in documents:
        print(f"----------------- {nlp_utils.get_num_tokens(doc, tokenizer)}")
        print(doc)
    # document, _ = generate_email_question_context_from_labeled_email(email, plain_body=use_plain_body)
    # documents = [documents]
    questions = [
        "Who sent the email?",
        "What is the subject of the email?",
        "When was the email sent?",
        "What is the arrival time?",
        "What is the departure time?",
        "What is the confirmation code?",
        "Where is the flight departing from?",
        "What is the destination city?",
        "Who is the passenger?",
    ]
    # for question in questions:
    #     import time
    #     time.sleep(1)
    #     answer = email_question_answers(
    #         question=question,
    #         documents=documents,
    #         example_email=example_document,
    #         example_questions=example_questions,
    #     )
    #     print("-------------------------")
    #     print(question, answer["answers"][0])
    #     print(answer["selected_documents"][0]["text"])


def create_email_chunks(text: str, max_tokens_per_chunk: int = 500) -> List[str]:
    num_tokens = nlp_utils.get_num_tokens(text, tokenizer)
    print(f"Num tokens before cleaning: {num_tokens}")
    chunks = []
    lines = text.split("\n")
    chunk_tokens = 0
    current_chunk = []
    i = 0
    while i < len(lines):
        line = email_utils.remove_whitespace(lines[i])
        if chunk_tokens >= max_tokens_per_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            chunk_tokens = 0
            i -= 1  # Overlapping chunks
        else:
            current_chunk.append(line)
            chunk_tokens += nlp_utils.get_num_tokens(line, tokenizer)
        i += 1
    chunks.append(" ".join(current_chunk))
    print(f"Num chunks {len(chunks)}")
    return chunks


def test_email_question_completion():
    # Assumption is that the
    questions = [
        "What is the verification code?",
        "Get verification code",
        "Get security code",
    ]
    example_emails = email_utils.search_local_emails(categories=["verification_code"])
    # train_emails, test_emails = make_train_test_split(
    #     example_emails, sort_key="email_id"
    # )
    train_emails, test_emails = extract_existing_train_test_split(example_emails)

    false_positives = make_email_false_positive_questions(
        emails=email_utils.search_local_emails(categories=["personal"])[:2],
        questions=questions,
    )
    false_positives += make_email_false_positive_questions(
        emails=email_utils.search_local_emails(categories=["recruiting"])[:2],
        questions=questions,
    )
    random.shuffle(false_positives)
    train_emails += false_positives[:2]
    test_emails += false_positives[2:4]

    random.shuffle(train_emails)
    print(f"Train: {len(train_emails)} Test: {len(test_emails)}")

    for email in train_emails + test_emails:
        num_tokens = nlp_utils.get_num_tokens(email["plain"], tokenizer)
        print(f"Num tokens before cleaning: {num_tokens}")
        email["plain"] = email_utils.remove_whitespace(email["plain"])
        num_tokens = nlp_utils.get_num_tokens(email["plain"], tokenizer)
        print(f"Num tokens after cleaning: {num_tokens}")
        email["plain"] = nlp_utils.trim_tokens(
            email["plain"], max_tokens=500, tokenizer=tokenizer
        )
        print(
            f"Email_id: {email['email_id']} Subject: {email['subject']} num_tokens: {num_tokens}"
        )

    failed = 0
    for example in test_emails:
        for question, expected_answer in example["model_labels"]["questions"]:
            answer = email_question_completion(
                user_email=example,
                user_question=random.choice(questions),
                prompt_emails=train_emails,
                completion_tokens=10,
            )
            if answer != expected_answer:
                print("Answers below do not match ----")
                print(example["email_id"], example["model_labels"]["categories"])
                print(f"Q: {question}\nA: {answer}\nE: {expected_answer}")
                failed += 1
    print(f"Failed: {failed}/{len(test_emails)}")


def test_email_question_completion_flights():
    # Assumption is that the
    questions = [
        "Extract flight details",
        "Get flight details",
    ]
    example_emails = email_utils.search_local_emails(
        categories=["flights", "flights_false_positive"], match_all=False
    )
    # train_emails, test_emails = make_train_test_split(
    #     example_emails, sort_key="email_id"
    # )
    train_emails, test_emails = extract_existing_train_test_split(example_emails)

    # false_positives = make_email_false_positive_questions(
    #     emails=email_utils.search_local_emails(categories=["personal"])[:2],
    #     questions=questions
    # )
    # false_positives += make_email_false_positive_questions(
    #     emails=email_utils.search_local_emails(categories=["recruiting"])[:2],
    #     questions=questions
    # )
    # random.shuffle(false_positives)
    # train_emails += false_positives[:1]
    # test_emails += false_positives[1:2]

    random.shuffle(train_emails)
    train_emails = random.sample(train_emails, 2)
    print(f"Train: {len(train_emails)} Test: {len(test_emails)}")

    for email in train_emails + test_emails:
        num_tokens = nlp_utils.get_num_tokens(email["plain"], tokenizer)
        print(f"Num tokens before cleaning: {num_tokens}")
        email["plain"] = email_utils.remove_whitespace(email["plain"])
        num_tokens = nlp_utils.get_num_tokens(email["plain"], tokenizer)
        print(f"Num tokens after cleaning: {num_tokens}")
        email["plain"] = nlp_utils.trim_tokens(
            email["plain"], max_tokens=500, tokenizer=tokenizer
        )
        print(
            f"Email_id: {email['email_id']} Subject: {email['subject']} num_tokens: {num_tokens}"
        )

    failed = 0
    for example in test_emails:
        for _, expected_answer in example["model_labels"].get(
            "questions", [(None, {})]
        ):
            answer = email_question_completion(
                user_email=example,
                user_question="flight details",
                prompt_emails=train_emails,
                completion_tokens=200,
                task_description="Extract flight details from the following emails. Input '???' to indicate missing or unknown fields.",
            )
            # if answer != expected_answer:
            #     print("Answers below do not match ----")
            print(example["email_id"], example["model_labels"]["categories"])
            pp.pprint(answer)
            failed += 1
    print(f"Failed: {failed}/{len(test_emails)}")


if __name__ == "__main__":
    # test_email_question_completion()
    # test_email_question_completion_flights()
    test_email_question_answers()
