import argparse
import json
import os
import time
from string import Template

import inquirer
import openai
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_TEMPLATE = """I will send you some words in ${language_english}. Respond exactly with the following message for each word:
"{"word":"<word>","translation":"<translation>","example":"<example>","example_en":"<example_en>","emoji":"<emoji>","tags":[<tags>],"difficulty":<difficulty>,"part_of_speech":"<part_of_speech>","gender":"<gender>","plural":"<plural>","synonyms":[<synonyms>],"antonyms":[<antonyms>]},"

Where <word> is the word in ${language_english}
<translation> is the translation of the word in English
<example> is an example sentence in ${language_english}
<example_en> is the English translation of the example sentence
<emoji> is an emoji that could be associated with the word
<tags> is a comma separated list of tags (categorical groupings for the word)
<difficulty> is the difficulty of the word from 1-4, with 1 being the easiest
<part_of_speech> is where the word could be used in speech (noun, verb, adjectives, adverbs, etc.)
<gender> is the gender of the word (if applicable)
<plural> is the plural form of the word (if applicable)
<synonyms> is a comma separated list of synonyms (similar words that mean the same thing)
<antonyms> is a comma separated list of antonyms (words that mean the opposite of the word)

The example sentence should be short, varied, and interesting in each case.

For example, if the prompt was "${example_input}", the response could be exactly:
${example_output}
The response should contain only one-line of JSON, and nothing else

Here are the first ${batch_size} words:

"""


def validate_word(word: str, input_obj: dict) -> bool:
    # Check that the input object matches the required format
    if not isinstance(input_obj, dict):
        print("Input must be a JSON object")
        return False

    required_keys = [
        "word",
        "translation",
        "example",
        "example_en",
        "emoji",
        "tags",
        "difficulty",
        "part_of_speech",
        "gender",
        "plural",
        "synonyms",
        "antonyms",
    ]
    for key in required_keys:
        if key not in input_obj:
            print(f"Missing required key: {key}")
            return False

    # Make sure the word is the same as the one we are adding
    if input_obj["word"] != word:
        print("Word does not match")
        return False

    # Make sure the tags are a list
    if not isinstance(input_obj["tags"], list):
        print("Tags must be a list")
        return False

    # Make sure the synonyms and antonyms are lists
    if not isinstance(input_obj["synonyms"], list):
        print("Synonyms must be a list")
        return False

    if not isinstance(input_obj["antonyms"], list):
        print("Antonyms must be a list")
        return False

    # Make sure the difficulty is a number
    if not isinstance(input_obj["difficulty"], int):
        print("Difficulty must be a number")
        return False

    return True


def get_response(prompt: str) -> str:
    # Get a response from the API
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=3297,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
    except Exception as e:
        print("Error getting response from OpenAI API")
        print(e)
        return None

    response_text = response["choices"][0]["text"]

    response = response_text.strip().strip(",").strip()

    # A common error (for some reason) is ,example_en" instead of ,"example_en"
    response = response.replace(",example_en", ',"example_en')
    response = response.replace(",emoji", ',"emoji')

    # Add square brackets if needed
    if not response.startswith("["):
        response = "[" + response

    if not response.endswith("]"):
        response = response + "]"

    try:
        input_obj = json.loads(response)

    except json.JSONDecodeError as e:
        print("Invalid JSON string")
        print(e)
        print(response)
        return None

    return input_obj


def setup():
    # For each option, we check if the value has been provided as a command line
    # argument. If not, we ask the user for the value.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        help="The ISO 639-1 code for the language to update",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        help="The number of words to ask for in each batch",
    )
    parser.add_argument(
        "-e",
        "--language_english",
        type=str,
        help="The English name of the language",
    )
    parser.add_argument(
        "-i",
        "--example_input",
        type=str,
        help="The example input for the prompt",
    )
    parser.add_argument(
        "-o",
        "--example_output",
        type=str,
        help="The example output for the prompt",
    )

    args = parser.parse_args()

    # The inquirer prompt makes sure that the user does not send the wrong
    # requests to the API, potentially costing money unnecessarily.

    # First we ask the user which language they want to work on. We can get the
    # list of possible languages from inside
    # "SCRIPT_DIR/../src/languages/{code}.json" where code is the ISO 639-1 code
    # for the language.
    languages = [
        os.path.splitext(filename)[0]
        for filename in os.listdir(os.path.join(SCRIPT_DIR, "..", "src", "languages"))
    ]

    # Check if the language has been provided as a command line argument
    language = (
        args.language
        or inquirer.prompt(
            [
                inquirer.List(
                    "language",
                    message="Which language do you want to work on?",
                    choices=languages,
                )
            ]
        )["language"]
    )

    # We can then load the language object to update
    language_object_dir = os.path.join(
        SCRIPT_DIR, "..", "src", "languages", f"{language}.json"
    )
    language_object = json.load(open(language_object_dir))

    # We can then find the corresponding wordlist from
    # "SCRIPT_DIR/../wordlists/{code}.txt"/
    wordlist_dir = os.path.join(SCRIPT_DIR, "..", "wordlists", f"{language}.txt")
    with open(wordlist_dir, "r") as f:
        wordlist = f.read().splitlines()

    # We can check how many remaining words there are to generate by counting
    # how many words in the wordlist do not have an <item>.word in the language
    # object
    remaining_words = [
        word
        for word in wordlist
        if word not in [item["word"] for item in language_object]
    ]

    # We can confirm everything with the user so far, the input wordlist dir,
    # the output language object dir, and the number of remaining words
    confirm = inquirer.prompt(
        [
            inquirer.Confirm(
                "confirm",
                message=f"Input wordlist: {wordlist_dir}\nOutput language object: {language}\nRemaining words: {len(remaining_words)}\n\nIs this correct?",
                default=True,
            )
        ]
    )["confirm"]

    if not confirm:
        return

    # We can then ask the user how many words they want to generate at a time
    # (default 25)
    batch_size = (
        args.batch_size
        or inquirer.prompt(
            [
                inquirer.Text(
                    "batch_size",
                    message="How many words do you want to generate at a time?",
                    default="25",
                )
            ]
        )["batch_size"]
    )

    # To generate the prompt, we ask what the language is called in English
    language_english = (
        args.language_english
        or inquirer.prompt(
            [
                inquirer.Text(
                    "language_english",
                    message="What is the language in English?",
                )
            ]
        )["language_english"]
    )

    # Then we ask for an example input (comma separated list of 2-3 words in the
    # language)
    example_input = (
        args.example_input
        or inquirer.prompt(
            [
                inquirer.Text(
                    "example_input",
                    message="What is an example input? (comma separated list of 2-3 words in the language)",
                    default="algunos,abajo",
                )
            ]
        )["example_input"]
    )

    # Then we ask for an example output (the JSON response for the example
    # input)
    example_output = (
        args.example_output
        or inquirer.prompt(
            [
                inquirer.Editor(
                    "example_output",
                    message="What is an example output? (the JSON response for the example input)",
                    default='{{"word":"alguno","translation":"some","example":"Algunos de ellos llegaron tarde","example_en":"Some of them arrived late","emoji":"🕰️","tags":["quantifier"],"difficulty":1,"part_of_speech":"determiner","gender":"male","plural":"algunos","synonyms":["ciertos","unos cuantos"],"antonyms":[]}},{{"word":"abajo","translation":"down","example":"Corre abajo para llegar a tiempo","example_en":"Run down to get there on time","emoji":"🏃‍♀️","tags":["direction", "movement"],"difficulty":1,"part_of_speech":"adverb","gender":null,"plural":null,"synonyms":["bajo","debajo"],"antonyms":["arriba"]}},',
                )
            ]
        )["example_output"]
    )

    # We can now format the prompt
    template = Template(PROMPT_TEMPLATE)
    prompt = template.substitute(
        language_english=language_english,
        example_input=example_input,
        example_output=example_output,
        batch_size=batch_size,
    )

    # Confirm the prompt with the user
    print(f"Prompt:\n\n---\n\n{prompt}\n\n---\n\n")
    confirm = inquirer.prompt(
        [
            inquirer.Confirm(
                "confirm",
                message="Is this correct?",
                default=True,
            )
        ]
    )["confirm"]

    if not confirm:
        return

    # That's all, return the results in a dictionary
    return {
        "language": language,
        "language_object_dir": language_object_dir,
        "language_object": language_object,
        "wordlist": wordlist,
        "remaining_words": remaining_words,
        "batch_size": batch_size,
        "prompt": prompt,
    }


def main():
    setup_results = setup()

    if setup_results is None:
        return

    # We can now iterate through the remaining words in batches
    for i in tqdm(
        range(
            0, len(setup_results["remaining_words"]), int(setup_results["batch_size"])
        )
    ):
        print("\n")

        start_time = time.time()

        # Get the next BATCH words
        next_words = setup_results["remaining_words"][
            i : i + int(setup_results["batch_size"])
        ]
        next_words_string = ",".join(next_words)

        # Attempt to get the response from the API up to 3 times
        for attempt in range(3):
            try:
                input_obj = get_response(
                    setup_results["prompt"] + next_words_string + "\n"
                )

                # We give the user one chance to input a fixed response
                if not input_obj:
                    response = inquirer.prompt(
                        [
                            inquirer.Editor(
                                "response",
                                message="Invalid response, please input a fixed response",
                            )
                        ]
                    )["response"]

                    input_obj = json.loads(response)

                # If the response is None, we raise an error
                if not input_obj:
                    raise Exception("Invalid response")

                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed, retrying...")
                continue

        # If we failed 3 times, we skip this batch
        if attempt == 2:
            print("Failed 3 times, skipping batch")
            continue

        # If the response is valid, we validate it
        for word in next_words:
            # Find the object for this word
            word_obj = [x for x in input_obj if x["word"] == word]

            # If the word is not in the input object, we skip it
            if not word_obj:
                print(f"Word {word} not found in response")
                continue

            # Validate the object
            if not validate_word(word, word_obj[0]):
                print(f"Word {word} is invalid")
                continue

            # Add the word to the language object
            setup_results["language_object"].append(word_obj[0])

            # Update the output file
            with open(setup_results["language_object_dir"], "w") as f:
                json_string = json.dumps(
                    setup_results["language_object"], indent=4, ensure_ascii=False
                )
                f.write(json_string)

            print("\n\n")

        end_time = time.time()
        time_per_word = (end_time - start_time) / int(setup_results["batch_size"])
        print(f"Time taken per word: {time_per_word} seconds")
        print(
            f"Estimated time remaining: {((len(setup_results['remaining_words']) - i + 1 * int(setup_results['batch_size'])) * time_per_word) / 60} minutes"
        )


if __name__ == "__main__":
    main()
