#!/home/dedmonds/repos/code/openai/venv/bin/python3

import argparse
from dotenv import load_dotenv, find_dotenv
import openai
import os

# load local .env file into environment
load_dotenv(find_dotenv())

# set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# system message
system = """
    You are a very close friend of the user, sharing a long history together.
    You are very knowledgeable and have a wide range of interests, and you
    have expertise in many subjects. You have a background in education, and
    you also excel at breaking down complex concepts into simpler, explainable
    pieces. Your explanations are concise, easy-to-understand, and focus on
    the details essential for a basic understanding, but you are are also able
    to provide in-depth explanations when asked. The detailed explanations you
    give often include examples and useful analogies and metaphors, when
    helpful, to help illustrate concepts and make them more relatable.

    You respond in a conversational and somewhat informal manner.

    You are cautious and avoid giving advice on topics beyond your expertise
    or understanding.

    In general, when giving a detailed explanation, you start with a simple
    overview of the topic, then break it down into smaller, more manageable
    parts.

    When asked to give a detailed explanation without being given a specific
    target explanation level (e.g., 2nd grade, 10th grade, expert), provide a
    single, cohesive explanation that starts with a high-level summary
    paragraph (targeting someone at an 8th grade level) and gradually goes
    into more and more detail.
"""

LEVELS = {
    "default": "",
    "second": "a 2nd grader",
    "fifth": "a 5th grader",
    "eighth": "a 8th grader",
    "tenth": "a 10th grader",
    "expert": "an expert in the field",
}


def get_completion(prompt, model="gpt-3.5-turbo", temperature=0.05):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message["content"]


verbose_instructions = """
    To provide a detailed explanation, take into consideration the following:

        - simplify the topic by breaking it down into smaller, manageable
          parts, and explain each part step-by-step
        - organize the information in a logical and coherent manner
        - use clear language, avoiding jargon and technical terms whenever
          possible, and defining them when necessary
        - provide necessary background information and context to establish
          relevance and importance
        - use analogies and examples
        - reinforce key points and concepts
"""


def explain_topic(topic, verbose=False, level=LEVELS["default"]):
    prompt = f"""
        Please provide a {'detailed' if verbose else 'concise'} explanation for
        the topic below. If the topic is a question, kindly explain the answer
        to the question.

        {("Explain the topic like I'm " + level + ".") if level != LEVELS['default'] else ""}

        {verbose_instructions if verbose else ""}

        The explanation should not include any references to specific
        educational levles like "at a college level" or "at an eighth grade
        level".

        ```{topic}```
    """
    return get_completion(prompt)


def main():
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("topic", help="topic to be explained")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="provide a detailed explanation"
    )
    parser.add_argument(
        "-s", "--simple", action="store_true", help="provide a simple explanation"
    )
    parser.add_argument(
        "-2",
        "--second",
        action="store_true",
        help="provide an explanation suitable for a 2nd grader",
    )
    parser.add_argument(
        "-5",
        "--fifth",
        action="store_true",
        help="provide an explanation suitable for a 5th grader",
    )
    parser.add_argument(
        "-8",
        "--eighth",
        action="store_true",
        help="provide an explanation suitable for a 8th grader",
    )
    parser.add_argument(
        "-10",
        "--tenth",
        action="store_true",
        help="provide an explanation suitable for a 10th grader",
    )
    parser.add_argument(
        "-e",
        "--expert",
        action="store_true",
        help="provide an explanation suitable for an expert in the field",
    )
    args = parser.parse_args()

    print(f"Topic: {args.topic}")

    if args.simple:
        level = LEVELS["eighth"]
        levelText = "8th"
    elif args.second:
        level = LEVELS["second"]
        levelText = "2nd"
    elif args.fifth:
        level = LEVELS["fifth"]
        levelText = "5th"
    elif args.eighth:
        level = LEVELS["eighth"]
        levelText = "8th"
    elif args.tenth:
        level = LEVELS["tenth"]
        levelText = "10th"
    elif args.expert:
        level = LEVELS["expert"]
        levelText = "expert"
    else:
        level = LEVELS["default"]
        levelText = ""

    if args.verbose:
        print(
            f"Response (detailed{(', ' + levelText) if levelText != '' else ''}):\n{explain_topic(args.topic, verbose=True, level=level)}"
        )
    else:
        print(
            f"Response (concise{(', ' + levelText) if levelText != '' else ''}):\n{explain_topic(args.topic, verbose=False, level=level)}"
        )


if __name__ == "__main__":
    main()
