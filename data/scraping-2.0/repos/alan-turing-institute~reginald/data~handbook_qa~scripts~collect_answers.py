# Standard library imports
import json
import os
import pickle
import sys
import time
from os.path import isfile, join
from typing import Dict, List

# Third-party imports
import html2text
import markdown
import openai


def read_markdown_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        html = markdown.markdown(content)
        text = html2text.html2text(html)
        return text


def read_questions() -> Dict:
    data_name = "reg"
    # Load existing answers
    try:
        results = pickle.load(
            open(f"collected_questions/{data_name}_generated.pickle", "rb")
        )
    except:
        results = {}
    return results


if __name__ == "__main__":
    openai.api_key = sys.argv[1]
    reg_folder = (
        sys.argv[2]
        if len(sys.argv) >= 3
        else "/Users/myong/Documents/workspace/reginald/data/handbook"
    )
    data_name = "reg"
    if not os.path.exists("collected_answers"):
        os.makedirs("collected_answers")

    context_questions = read_questions()
    context_list = context_questions.keys()
    done_answers = [
        "editing_a_page.md",
        "lunchtime_tech_talks.md",
        "knowledge_sharing.md",
        "contributing_changes.md",
        "overtime.md",
        "lightning_talks.md",
        "defining-done.md",
        "creating_a_page.md",
        "meeting_record.md",
        "advanced.md",
        "overtime_tmp.md",
        "configuring_editors.md",
        "reviewing_changes.md",
        "systems_set_up.md",
        "reviewing_changes.md",
        "project_tracking.md",
        "buddy_system.md",
        "style_guide.md",
        "getting_started.md",
        "style_guide.md",
        "project_tracking.md",
        "buddy_system.md",
        "systems_set_up.md",
        "configuring_editors.md",
        "reviewing_changes.md",
        "change_logs.md",
        "drop-in_sessions.md",
        "discussions_and_issues.md",
        "reading_groups.md",
        "twitter.md",
        "python.md",
        "first_few_days.md",
        "coffee_chats.md",
    ]
    for c in context_list:
        if c not in done_answers:
            print("\n\nAwake and working on: {}".format(c))
            qa_pairs = []
            questions = context_questions[c][0]["question"].split("\n")
            context_file = context_questions[c][0]["context_name"]
            for q in questions[:]:
                q_tokens = q.split(". ")
                if len(q_tokens) > 1:
                    q_text = q_tokens[1]
                else:
                    q_text = q_tokens[0]

                context = read_markdown_file(context_file)
                question_prompt = "Answer this question: {} using the following context: {}. The answer should be maximally three sentences long".format(
                    q_text, context
                )
                answer_generator = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": question_prompt}],
                )
                answer = answer_generator["choices"][0]["message"]["content"].strip()
                print("Answered: {}".format(q_text))
                # print("Question: {}".format(q_text))
                # print(f"Answer: {answer}")
                qa_pairs.append({"prompt": q_text, "completion": answer})

            pickle.dump(
                qa_pairs,
                open(
                    "collected_answers/{}_generated.pickle".format(c.split(".md")[0]),
                    "wb",
                ),
            )
            print("Sleeping")
            time.sleep(21)
