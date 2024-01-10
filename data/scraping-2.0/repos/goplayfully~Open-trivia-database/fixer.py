import json
import os
import random
import time
import logging
import coloredlogs
import hashlib
import argparse
import vertexai
import openai
from itertools import islice
from vertexai.preview.language_models import ChatModel, InputOutputTextPair
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

# Firebase for caching
# Use the application default credentials.
cred = credentials.ApplicationDefault()
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
    database_handle = firestore.client()


VERTEX_STORY_MODEL = "chat-bison@001"
# OPENAI_MODEL="gpt-4-0613"
OPENAI_CHEAP_MODEL="gpt-3.5-turbo-0613"
OPENAI_TEMPERATURE = 0.2

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')

def read_lines(filename):
    """Reads the contents of a file and returns a list of lines.
    Args:
      filename: The name of the file to read.
    Returns:
      A list of lines in the file.
    """

    with open(filename, "r") as f:
        lines = f.readlines()
    return lines


def parse_json_line(line):
    """Parses a line of text into JSON and returns the result.
    Args:
      line: The line of text to parse.
    Returns:
      A JSON object or None if the line could not be parsed.
    """

    # chomp last comma
    line = line[:-2]
    try:
        json_object = json.loads(line)
        return json_object
    except json.JSONDecodeError:
        return None


def banter(filename, type):
    """Parse an array of trivia objects and store in db"""
    file = open(filename)
    trash_array = json.load(file)
    for banter in trash_array:
        logger.info("writing trash to db: %s", banter)
        banter["random_1"] = int(random.getrandbits(32))
        banter["random_2"] = int(random.getrandbits(32))
        banter["random_3"] = int(random.getrandbits(32))
        banter["type"] = type
        database_handle.collection("banter").add(banter)
    file.close()


def check_and_fix_question(question_dict):
    """Validates a question against heuristics and LLMs"""

    if question_dict["correct_answer"].lower() in question_dict["question"].lower():
        logging.warning(
            'Found answer in question: "%s" (%s) (%s)',
            question_dict["question"],
            question_dict["correct_answer"],
            question_dict["content_id"])
        question_dict["question"] = (question_dict["question"]
                                        .lower()
                                        .replace(
            question_dict["correct_answer"].lower(),
                                                "_____")).capitalize()
        logging.info('New question: "%s"', question_dict["question"])
        (database_handle
            .collection("trivia")
            .document(question_dict["content_id"])
            .update({
                'question': question_dict["question"],
                'to_review': True}))
    else:
        # Set up Vertex/PaLM
        chat_model = ChatModel.from_pretrained(VERTEX_STORY_MODEL)
        parameters = {
            "temperature": 0.2,
            "max_output_tokens": 1024,
        }
        chat = chat_model.start_chat(
            context="You are an editor and producer for a trivia gameshow. Please rate and respond to the following trivia questions.",
            examples=[
                InputOutputTextPair(
                    input_text="""The phrase \"Homo sapiens\" means ____. ANSWERS: [Man who thinks, Man of steel, Man of wisdom, Man of Saturn]. CORRECT ANSWER: Man who thinks.""",
                    output_text="""{"problems": [], "humor": "low", "difficulty": "low"}""",
                ),
                InputOutputTextPair(
                    input_text="""__________ and short_tailed shrews get by on only two hours of sleep a day. ANSWERS: [Elephants, Mice, Gerbils, Giraffes]. CORRECT ANSWER: Elephants""",
                    output_text="""{"problems": [], "humor": "low", "difficulty": "medium"}""",
                ),
                InputOutputTextPair(
                    input_text="""__________ and short_tailed shrews get by on only two hours of sleep a day. ANSWERS: [Elephants, Large Elephants, Gerbils, Giraffes]. CORRECT ANSWER: Elephants""",
                    output_text="""{"problems": ["Elephants and 'Large Elephants' are too similar, and both could be considered correct"], "humor": "low", "difficulty": "medium"}""",
                ),
                InputOutputTextPair(
                    input_text="""Elephants and short_tailed shrews get by on only two hours of sleep a day. ANSWERS: [Elephants, Mice, Gerbils, Giraffes]. CORRECT ANSWER: Elephants""",
                    output_text="""{"problems": ["input is not a question"]}""",
                ),
                InputOutputTextPair(
                    input_text="""Alligators and frogs can hear notes only up to ____ vibrations a second. ANSWERS: [4000, 5000, 3000, 2000]. CORRECT ANSWER: 4000""",
                    output_text="""{"problems": ["Answers are too precise. Better answers would be further apart, like 4000, 20000, 100, 40000."], humor": "low", "difficulty": "high"}""",
                ),
            ],
        )
        try:
            response = chat.send_message(
                f"""Review the following question and return a JSON object
                listing any problems, then
                score it for humor (low/medium/high) and
                difficulty (low/medium/high).
                A question has a PROBLEM
                if the correct answer is not correct, if the answers are too
                similar, or if the answers are overly quantitative.
                DIFFICULTY is based on how many people would know the answer
                (high = more than 50%, medium = 25%, low = below 25%) based
                on how often the topic is discussed online.

                INPUT
                ---
                QUESTION: {question_dict["question"]}
                ANSWERS: {question_dict["answers"]}
                CORRECT ANSWER: {question_dict["correct_answer"]}
                ---
                """,
                **parameters)
        except:
            logger.warning('Error from VertexAI (sleeping): %s')
            time.sleep(5)
        if "false" in response.text:
            logger.warning('Found problem with question %s', question_dict['content_id'])
        logger.debug(f"Response from Model for '{question_dict['question']}':\n\t({question_dict['content_id']}): {response.text}")
        try:
            output_obj = json.loads(response.text)
        except json.JSONDecodeError as err:
            logger.error('Could not parse JSON: %s (%s)', response.text, err)
            return
        output_obj['timestamp'] = firestore.SERVER_TIMESTAMP
        output_obj['question'] = question_dict['question']
        if len(output_obj['problems']) > 0:
            (database_handle
            .collection("trivia_feedback")
            .document("fixer.py-problems")
            .set({
                question_dict['content_id']: output_obj,
            }, merge=True))
        else:
            (database_handle
            .collection("trivia_feedback")
            .document("fixer.py-good")
            .set({
                question_dict['content_id']: output_obj,
            }, merge=True))
        (database_handle
            .collection("trivia")
            .document(question_dict["content_id"])
            .update({
                'question': question_dict["question"],
                'robo_reviewed': True}))


def trivia_check(category):
    """Check each entry in the database for correctness"""
    questions = (database_handle
                 .collection("trivia"))
    if category != "":
        questions = questions.where(
            filter=FieldFilter("category_id", "==", category))

    questions = questions.stream()

                 # .where(
                 #       filter=FieldFilter("proofed", "!=", True))
                 # .limit(50)

    for question in questions:
        question_dict = question.to_dict()
        logging.debug('Checking question %s', question_dict["content_id"])
        if question_dict.get("proofed"):
            logging.debug('Question already proofed, skipping')
            continue
        check_and_fix_question(question_dict)

    if category != "":
        logger.info('category set; skipping user feedback')
        return

    logger.info('Moving to user feedback')
    users = (database_handle
                 .collection("trivia_feedback")
                 .stream())
    for user in users:
        feedback_report = user.to_dict()
        for feedback in feedback_report:
            if feedback_report[feedback].get("type") == 'problem':
                logging.warning("""Problematic question
                                with content_id %s reported by
                                user %s""",
                                feedback,
                                user.id)
                question = (database_handle
                            .collection("trivia")
                            .document(feedback)
                            .get())
                if question.exists:
                    logging.warning(question.to_dict())
            else:
                logging.info("Feedback for content %s: %s",
                             feedback,
                             feedback_report[feedback].get("type"),)

def main():
    '''Process input files from Open-trivia-database and ask LLM to
    fix formatting problems and add multiple-choice answers'''

    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        help="The trivia question JSON list")
    parser.add_argument(
        "--nodb",
        help="Don't write to the database",
        action="store_true",
        )
    parser.add_argument(
        "--limit",
        help="The max number of rows to parse",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--mode",
        help="""Should this read <trivia> (default) or <trash>
        or <congrats> or <trivia_check>?""",
        default="trivia"
    )
    parser.add_argument(
        "--category",
        help="Which category to check",
        default="",
        type=str,
    )
    parser.add_argument(
        "--skip",
        help="Number of rows to skip, for resuming previous task",
        default=0,
        type=int,
    )
    args = parser.parse_args()

    filename = args.filename
    BATCH_SIZE = 10
    lines = 0

    logger.info('mode = %s', args.mode)
    if args.mode == "trash" or args.mode == "congrats":
        return banter(filename, args.mode)
    elif args.mode == "trivia_check":
        return trivia_check(args.category)

    pathless_filename = os.path.basename(filename)
    out_file = open(pathless_filename + ".out", "a")
    problem_file = open(pathless_filename + ".problems", "a")

    # Set up Vertex/PaLM
    chat_model = ChatModel.from_pretrained(VERTEX_STORY_MODEL)
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 1024,
    }
    chat = chat_model.start_chat(
        context="You are a code editor for a set of trivia questions stored as JSON objects. Please fix the following trivia questions.",
        examples=[
            InputOutputTextPair(
                input_text="""{"category_id":"SCIENCE_AND_NATURE", "lang":"en", "tags":[], "question":"The phrase "Homo sapiens " means ____", "answer":0, "answers":["Man of knowledge"], "source":""},""",
                output_text="""{"category_id":"SCIENCE_AND_NATURE", "lang":"en", "tags":[], "question":"The phrase \"Homo sapiens\" means ____", "answer":0, "answers":["Man of knowledge", "Man of science", "Man who thinks", "Man of steel"], "source":"", "explanation":"The phrase \"Homo sapiens\" was first coined by Carl Linnaeus in 1758, a Swedish biologist who created the Latin binomial nomenclature"},""",
            ),
            InputOutputTextPair(
                input_text="""{"category_id":"SCIENCE_AND_NATURE", "lang":"en", "tags":["SCIENCE_AND_NATURE"], "question":"__________ and short_tailed shrews get by on only two hours of sleep a day.", "answer":0, "answers":["Elephants"], "source":""}""",
                output_text="""{"category_id":"SCIENCE_AND_NATURE", "lang":"en", "tags":["SCIENCE_AND_NATURE"], "question":"__________ and short tailed shrews get by on only two hours of sleep a day.", "answer":0, "answers":["Elephants", "Horses", "Hippos", "Voles"], "source":"", "explanation":"Elephants are the shortest sleeping mammal, and only dream every three to four days"}""",
            )
        ],
    )

    prompt_template = """For each line in the following JSON array, please perform
        the following steps:
        1) Inside the "question" element, escape all
        unescaped or duplicated quotation marks with a \ character so the line is valid JSON
        2) Fix unnecessary title casing
        3) Remove unncessary spaces near quotation marks
        4) Add three plausible, unique, wrong entries to the `answers` array
        5) Ensure the "answer" does not appear in the "question"; if there is a "____" in the input, DO NOT REPLACE IT
        6) Please provide an explanation and interesting factoid in the "explanation" parameter
        7) Remove the "source" and "tags" parameters
        8) Please format the output with each object on ONE SINGLE line

        EXAMPLES:
            input_text={"category_id":"SCIENCE_AND_NATURE", "lang":"en", "tags":[], "question":"The phrase "Homo sapiens " means ____", "answer":0, "answers":["Man of knowledge"], "source":""},
            output_text={"category_id":"SCIENCE_AND_NATURE", "lang":"en", "question":"The phrase \"Homo sapiens\" means ____", "answer":0, "answers":["Man of knowledge", "Man of science", "Man who thinks", "Man of steel"], "explanation":"The phrase \"Homo sapiens\" was first coined by Carl Linnaeus in 1758, a Swedish biologist who created the Latin binomial nomenclature"},

            input_text={"category_id":"SCIENCE_AND_NATURE", "lang":"en", "tags":["SCIENCE_AND_NATURE"], "question":"__________ and short_tailed shrews get by on only two hours of sleep a day.", "answer":0, "answers":["Elephants"], "source":""},
            output_text={"category_id":"SCIENCE_AND_NATURE", "lang":"en", "question":"__________ and short tailed shrews get by on only two hours of sleep a day.", "answer":0, "answers":["Elephants", "Horses", "Hippos", "Voles"], "explanation":"Elephants are the shortest sleeping mammal, and only dream every three to four days"},

        INPUT:
        """

    doc_ref = database_handle.collection("trivia")

    with open(filename) as file:
        # skip first line
        next_n_lines = list(islice(file, 1))
        if (args.skip > 0):
            next_n_lines = list(islice(file, args.skip))
        while True:
            logger.info("Processing a batch starting at %s...", lines)
            lines = lines + BATCH_SIZE
            next_n_lines = list(islice(file, BATCH_SIZE))
            if not next_n_lines:
                break

            # create a concatenated string
            input_lines = ""
            for line in next_n_lines:
                input_lines = input_lines + line

            logger.info('Submitting to model: input %s',
                        input_lines)

            messages = [
                {
                    "role": "system",
                    "content": "You are a code editor for a set of trivia questions stored as JSON objects. Please fix the following trivia questions.",
                },
                {
                    "role": "user",
                    "content": prompt_template + input_lines,
                }
            ]
            functions = [
                {
                    "name": "ask_trivia_question",
                    "description": "Ask user trivia question, evaluate answer, and pring explanation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category_id": {
                                "type": "string",
                                "description": "The category of the question, e.g. SCIENCE_AND_NATURE"
                            },
                            "lang": {
                                "type": "string",
                                "description": "The language code for the question, e.g. en"
                            },
                            "question": {
                                "type": "string",
                                "description": "Sentence-cased text of the trivia question, with any quotes escaped with '/' "
                            },
                            "answer": {
                                "type": "integer",
                                "description": "The index of the correct answer from the 'answers' array"
                            },
                            "answers": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "An array containing possible answer choices"
                            },
                            "source": {
                                "type": "string",
                                "description": "The source from which the question was derived"
                            },
                            "explanation": {
                                "type": "string",
                                "description": "A brief explanation or factoid related to the question and answer"
                            }
                        }
                    }
                }
            ]

            # vertex/PaLM call
            # response = chat.send_message(
            #     prompt_template + input_lines,
            #     **parameters)
            # response_message = response.text

            # Chat GPT model
            response = openai.ChatCompletion.create(
                model=OPENAI_CHEAP_MODEL,
                messages=messages,
                temperature=OPENAI_TEMPERATURE,
                max_tokens=2500,
            )
            response_message = response["choices"][0]["message"]["content"]

            logger.info(f"Response from Model: {response_message}")
            out_file.write(response_message + "\n")

            # write to DB
            for response_line in response_message.splitlines():
                trimmed_line = response_line
                question_obj = set()
                try:
                    if not trimmed_line.strip():
                        # skip blank lines
                        continue
                    # chomp the trailing , so we can parse this individually
                    if trimmed_line[-1] == ',':
                        trimmed_line = trimmed_line[:-1]
                    question_obj = json.loads(trimmed_line)
                except json.JSONDecodeError as err:
                    logger.error(
                        "Could not parse line '%s', %s @ %s",
                        response_line,
                        err.msg,
                        err.pos)
                    problem_file.write(response_line + "\n")
                    continue

                question_obj["random_1"] = int(random.getrandbits(32))
                question_obj["random_2"] = int(random.getrandbits(32))
                question_obj["random_3"] = int(random.getrandbits(32))
                try:
                    question_obj["correct_answer"] = question_obj["answers"][0]
                except KeyError as err:
                    logger.error("Missing answer in question %s (%s)",
                                 trimmed_line, err)
                    problem_file.write(response_line + "\n")

                # TODO(mrisher): This hashes based on the post-LLM topic
                # so potentially we already have an essentially identical
                # question in the db. In the future, maybe we should hash
                # based on the raw input question (from the json file)
                # but then we would need to map the input to output to
                # get the key
                # save to database
                if not args.nodb:
                    question_key = hashlib.sha1(
                        question_obj["question"].encode("utf-8")).hexdigest()
                    if doc_ref.document(question_key).get().exists:
                        logger.info(
                            'Already found entry with key %s',
                            question_key)
                        continue
                    logger.info(
                        "Adding line to database: '%s'",
                        response_line[:-1])
                    question_obj["content_id"] = question_key
                    question_obj["proofed"] = False

                    # remove "answer" key, which was a complicated array index
                    try:
                        question_obj.pop("answer")
                    except (KeyError):
                        # ignore
                        logger.debug('question had no "answer" field')

                    doc_ref.document(question_key).set(question_obj)

            # early exit
            if lines >= args.limit:
                logger.info('Bailing after %s lines', lines)
                break
    out_file.close()
    problem_file.close()

    # Then please add plausible but wrong answers to the `answers` array, such that answer[0] is the correct one. Finally, p
    # for line in lines:
    #     json_object = parse_json_line(line)
    #     if json_object is None:
    #         print(line)


if __name__ == "__main__":
    main()
