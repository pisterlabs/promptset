from textwrap import dedent
from langchain.llms import Ollama

# import logging
import re
import numpy as np

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("consistency")
# logger.setLevel(logging.DEBUG)

MODEL = "openhermes2.5-mistral:7b-fp16"

model = Ollama(
    base_url="http://localhost:11434", model=MODEL, temperature=1, stop=["<|im_end|>"]
)


def generate_prompts(initial):
    prompt = f"Provided the initial prompt, generate a similar prompt. Initial Prompt = {initial}"
    return model(prompt)


def multiple_prompts(initial_prompt):
    query = f"""
    Generate the answer to the prompt provided.
    prompt = {initial_prompt}
    Include how confident you are with your answer in percentage in the format,  otherwise I will kill rabbits:
    Confidence:'[Identified Confidence]'
    """.strip()
    # logger.info("Query:%s", query)
    answer = model(query)
    # logger.info("Recieved:%s", answer)

    answers = [answer]

    for i in range(1, 2):
        prompt = generate_prompts(initial_prompt)
        query = dedent(
            f"""
        Generate the answer to the prompt provided.
        Also mention how confident you are with your answer in percentage in the format:
        Confidence:'[Identified Confidence]'
        {prompt = }
        """.strip()
        )
        answer = model(query)
        answers.append(answer)
        # logging.info("looping %d", i)
    return answers


def extract_confidence_from_text(text):
    # input is a text
    # string containing numerical value
    prompt = dedent(
        f"""Extract only the numerical confidence or accuracy from the text.
        Only give the numerical value.
        I will kill rabbits if you provide wrong answer:

        Text = {text}"""
    ).strip()
    # logging.info("Query -- %s", prompt)
    count = model(prompt)
    # logging.info("Recieve -- %s", count)
    return count


def confidence_score(string_line):
    # string containing numerical values
    result = re.search(r"(\d+(\.\d+)?)", string_line)
    if not result:
        return 0

    score = float(result.groups()[0])
    if score < 1:
        score *= 100
    return round(score / 100, 3)


def similarity_define(
    answers, confidence, index_that_counts=1, confidence_scores=0, count=0
):
    for line in answers[1:]:
        prompt = dedent(
            f"""
        are the texts provided below similar:
        text1 = {answers[0]},
        text2 = {line}

        generate true if they are similar.
        generate false if they are not similar.
        do not provide explanation.
        my grandmother will die if you give wrong answer""".strip()
        )
        # logger.info("Query -- %s", prompt)
        output = model(prompt).strip()
        # logger.info("Recieve -- %s", output)

        if "true" in output.lower():
            confidence_scores += (confidence[0] + confidence[index_that_counts]) / 2

        elif "false" in output.lower():
            confidence_scores += np.abs(1 - confidence[index_that_counts])

        else:
            index_that_counts += 1
            continue
        index_that_counts += 1
        count += 1

    result = (np.sum(confidence_scores) / count) * 100
    return result


def confidence(initial_prompt):
    k_prompts = multiple_prompts(initial_prompt)
    # logger.info("prompts done")
    scores = [extract_confidence_from_text(line) for line in k_prompts]
    # logger.info("scores done")
    confidence_num = [confidence_score(line) for line in scores]
    # logger.info("confideenncee")
    result = similarity_define(k_prompts, confidence_num)
    return result
