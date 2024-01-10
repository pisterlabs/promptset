import json
import openai
import time

### METRICS
import bleu as BLEU
import rouge as ROUGE

### CUSTOM
from dataset_parsing import read_qald, parse_qald


def get_direct_answer(question, expected_answer=None):
    """
    Get direct answer

    :param question: question
    :param expected_answer: expected answer
    :return: answer
    """
    # set hyperparameters
    # This model's maximum context length is 4097 tokens
    answer_length = (
        min(len(str(expected_answer)) * 2, 4000) if expected_answer else 1000
    )

    print(answer_length, expected_answer)
    print(question)

    openai.api_key = OPENAI_API_KEY
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=question,
        temperature=0,
        max_tokens=answer_length,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        # don't put stop words (SPARQL queries have \n)
    )

    return response["choices"]


def direct_answers(ids, questions, answers):
    """
    Get direct answers and save them to a file

    :param ids: ids
    :param questions: questions
    :param answers: answers
    """

    data = {}

    for i in range(len(questions)):
        print(i)

        direct_answers = get_direct_answer(questions[i], answers[i])

        insides = []

        for direct_answer in direct_answers:
            print(direct_answer["text"])
            insides.append(direct_answer["text"])

        data[ids[i]] = insides

        if i % 10 == 0:
            with open("output/gpt/direct_answers.json", "w") as outfile:
                json.dump(data, outfile)

            print("Saved")
            print("sleeping for 1 minute")
            # sleep for 1 minute to avoid rate limit
            time.sleep(60)

    # last time save (to overwrite the file)
    with open("output/gpt/direct_answers.json", "w") as outfile:
        json.dump(data, outfile)


def get_sparql_answer(question, expected_answer=None):
    """
    Get SPARQL answer
    Example:
     Turn this into a DBpedia SPARQL query: "<question>"

    :param question: question
    :param expected_answer: expected answer
    :return: answer
    """
    # set hyperparameters
    # This model's maximum context length is 4097 tokens
    answer_length = (
        min(len(str(expected_answer)) * 2, 4000) if expected_answer else 1000
    )

    print(answer_length, expected_answer)
    print(question)

    openai.api_key = OPENAI_API_KEY
    # Turn this into a DBpedia SPARQL query: "Who killed Caesar?"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f'Turn this into a DBpedia SPARQL query: "{question}"',
        temperature=0,
        max_tokens=answer_length,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        # don't put stop words (SPARQL queries have \n)
    )

    return response["choices"]


def sparql_answers(ids, questions, answers):
    """
    Get SPARQL answers and save them to a file

    :param ids: ids
    :param questions: questions
    :param answers: answers
    """
    data = {}

    for i in range(len(questions)):
        print(i)
        sparql_answers = get_sparql_answer(questions[i], answers[i])

        insides = []

        for sparql_answer in sparql_answers:
            print(sparql_answer["text"])
            insides.append(sparql_answer["text"])

        data[ids[i]] = insides

        if i % 10 == 0:
            with open("output/gpt/sparql_answers.json", "w") as outfile:
                json.dump(data, outfile)

            print("Saved")
            print("sleeping for 1 minute")
            # sleep for 1 minute to avoid rate limit
            time.sleep(60)

    # last time save (to overwrite the file)
    with open("output/gpt/sparql_answers.json", "w") as outfile:
        json.dump(data, outfile)


# Turn this into a DBpedia SPARQL query: "Who killed Caesar?"


def main():
    # read QALD-9 test dataset
    qald_test = read_qald("data/qald_9_test.json")
    ids, questions, keywords, queries, answers = parse_qald(qald_test)

    print("Number of questions:", len(questions))

    # direct_answers(ids, questions, answers)
    sparql_answers(ids, questions, answers)


if __name__ == "__main__":
    SEED = 42

    with open("data/key.txt", "r") as f:
        OPENAI_API_KEY = f.readline()

    main()
