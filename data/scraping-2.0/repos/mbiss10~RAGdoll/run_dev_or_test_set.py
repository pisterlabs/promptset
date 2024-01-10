"""
Given a document path to a JSON file containing a list of question objects (i.e. a Python dict
with the keys 'question', 'answer', and 'q_num'), this script will run the QA agent on each
question and save the results in a JSON file located at 'OUT_PATH'.

NOTE: ensure the agent has been configured to run with the appropriate configuration for the
given dev/test set, e.g. pointing it to the correct vectorstore location.
"""

import sys
from langchain.chat_models import ChatOpenAI
import pickle
from single_qa_agent import SingleQaAgent
import prompts as prompts
import json

# input:
QA_REF_PATH = "./data/test/questions_with_ref_answers.json"

# output:
OUT_PATH = "./results/test/test_answers_turbo.json"


if __name__ == "__main__":

    llm = ChatOpenAI(temperature=0)
    agent = SingleQaAgent(llm,
                          "./data/test/db_math_with_sources.pkl",
                          prompt=prompts.TURBO_PROMPT,
                          vectorstore_k=8,
                          vectorstore_sim_score_threshold=0.7,
                          passages_path="./data/test/math_data.pkl",
                          tfidf_k=10)

    # Ask questions and save the results in json format
    with open(QA_REF_PATH, "r") as questions:
        with open(OUT_PATH, "w") as out:
            res = []
            for q in json.load(questions):
                question = q["question"].strip()
                print(f"Asking: {question}")

                response = agent.ask(question, return_only_outputs=False)

                q["ragdoll_answer"] = response['output_text']
                res.append(q)

            json.dump(res, out)

    # Save pickled version of QA history
    with open(f"./results/test/turbo_agent_history_on_test_set.pkl", "wb") as f:
        pickle.dump(agent.history, f)

    # To open the history:
    # with open("./dev_history_1.pkl", "rb") as f:
    #     history = pickle.load(f)
    #     print(history)
