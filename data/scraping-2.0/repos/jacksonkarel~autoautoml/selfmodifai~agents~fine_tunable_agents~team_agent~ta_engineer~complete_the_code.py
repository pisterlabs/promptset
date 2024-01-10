import re
import logging
from transformers import pipeline
from tqdm import tqdm
from selfmodifai.agents.fine_tunable_agents.team_agent.helpers import code_from_gpt
from selfmodifai.helpers import openai_response, openai_r_and_update, new_openai_message


def complete_the_code(messages, pattern):
    engineer_response_content = openai_response("gpt-4", messages, "Engineer")

    logging.info(f"Engineer: {engineer_response_content}")

    code_files = code_from_gpt(engineer_response_content)

    complete_code = ""

    if code_files:
        for code in code_files:
            code_content = code[1]

            manager_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"Is this code complete?\n{code_content}",
                },
            ]

            manager_response_content, manager_messages = openai_r_and_update("gpt-4", manager_messages, "Manager")

            classifier = pipeline("zero-shot-classification")
            labels = [
                "complete",
                "incomplete",
            ]
            results = classifier(
                sequences=manager_response_content,
                candidate_labels=labels,
                hypothesis_template="This text says that the code is {}",
            )

            result_label = results["labels"][0]

            if result_label == "complete":
                complete_code += code_content

            else:
                logging.info(f"Incomplete code: {code_content}")
                extract_inc_prompt = f"Make separate files for each incomplete function or class: {code_content}"
                extract_response_content = new_openai_message(extract_inc_prompt, "Incomplete code extraction")
                snippets = re.findall(pattern, extract_response_content, re.DOTALL)

                logging.info("Code snippets")
                for snippet in tqdm(snippets):
                    snippet_message = {"role": "user", "content": f"Finish writing this code:\n{snippet[1]}\n"}
                    snippet_messages = messages.append(snippet_message)
                    complete_code += complete_the_code(snippet_messages, pattern)

        if complete_code:
            logging.info(f"Complete code: {complete_code}")

    else:
        logging.info("No code detected")

    return complete_code
