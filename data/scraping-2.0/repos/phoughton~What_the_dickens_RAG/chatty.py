from decouple import config
import argparse
import time
import sys
import db_query
import openai


parser = argparse.ArgumentParser(
    description='Chat with a Charles Dickens expert.')

parser.add_argument('-v', '--verbose',
                    help='Show details of the query being handled',
                    required=False,
                    default=False,
                    action='store_true')
parser.add_argument('-n', '--noai',
                    help='Disable the AI and just use the database',
                    required=False,
                    default=False,
                    action='store_true')
args = parser.parse_args()

openai.api_key = config("API_KEY")


def type_text(text: str) -> None:
    """Types the text out on the screen"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.01)


def extract_context_additions(data: dict) -> str:
    """Extracts the context additions from the data"""
    context_additions = ""

    for index, doc in enumerate(data["documents"][0]):
        the_id = data["ids"][0][index]
        context_additions += the_id + f": {doc}\n\n"
    return context_additions


def make_openai_call(msg_flow: list, verbose: bool) -> str:
    """Makes a call to the OpenAI API"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=msg_flow,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    if verbose:
        print("Response: ")
        print(response)
        print()

    if "choices" not in response:
        return ("Sorry, I can't answer that question, "
                "please try rephrasing your question.")
    else:
        answer = response.get("choices")[0].get("message").get("content")
        return answer


def remove_question(question: str) -> str:
    """Removes the question from the string"""

    msg_flow = [
        {
            "role": "system",
                    "content": (
"""You are a linguist. You will analyse the text and provide an answer based on the grammar and language provided. Your answer will be a list of words in the users text, removing the questioning words and...

Also add to the list common alternatives and their Victorian English conceptual equivalents with out questioning words. do not provide the results under a heading, just a list of words. """)
        },
        {
            "role": "assistant",
            "content": (
"""Answer the question as a list of comma separated words E.g.:
Word1, word2, word2, word4""")
        },
        {
            "role": "user",
            "content": (f"{question}")
        }
    ]

    return make_openai_call(msg_flow, verbose=args.verbose)


type_text("Hello, I'm an AI with access to the works of Charles Dickens. "
          "I can answer questions about his work")

while True:
    question = input(": ")
    if question in ["exit", "quit", ""]:
        break

    concepts_not_questions = remove_question(question)

    query_for_db = concepts_not_questions + ". " + question

    if args.verbose:
        print("\nQuery for DB: ")
        print(query_for_db)
        print()

    related_texts = db_query.get_chroma_response(query_for_db)

    context_additions = extract_context_additions(related_texts)

    if args.verbose:
        print("\nContext additions: ")
        print(context_additions)
        print()

    msg_flow = [
        {
            "role": "system",
            "content": ("You are a expert librarian. You will answer "
                        "questions about his works based only on the "
                        "extracts provided by the assistant. You will "
                        "not use your own knowledge. If the extracts "
                        "mention subject or action that you do not "
                        "know about, you will not make up an answer.")
        },
        {
            "role": "assistant",
            "content":
                "The response should be based only on the following sections from the books, contained between these back ticks:\n"  # noqa: E501
                f"```{context_additions}```\n"  # noqa: E501
                "Make the response sound authoritative and use the data provided above to answer the question.\n"  # noqa: E501
                "Quote the extracts provided with section numbers. The quotes and section numbers must match must exactly match those provided.\n"  # noqa: E501
                "Do not comment on the accuracy of the extracts provided or if they are anachronistic.\n"  # noqa: E501
        },
        {
            "role": "user", "content": question
        }
    ]

    if args.verbose:
        print("Message flow: ")
        print(msg_flow)
        print()

    if args.noai:
        print("AI answering is disabled.")

    else:
        type_text(make_openai_call(msg_flow, verbose=args.verbose))

    print()
