# Authors: marcusm117
# License: Apache 2.0


# Standard Library Modules
import argparse
import json

# External Modules
from codebleu import calc_codebleu
from evaluate import load
import numpy as np
import openai
import pandas as pd
from tqdm import tqdm


# add your OpenAI API key here
# only for calculating the text-embedding-ada-002 score, which is by default not enabled
# you don't need to add your API key if you don't want to calculate this metric
openai.api_key = ""


# summarize the test results of a single task
def summarize_test_results(PL_i_results):
    # if not defined, return "NA"
    if PL_i_results == "NA":
        return PL_i_results
    # if some test cases didn't pass, return the first Error
    for result in PL_i_results:
        if "Error" in result:
            return result
    return "Passed"


# compute cosine similariy of a list of reference and candidate pairs
# in each pair, reference and candidate are both text-embedding-ada-002 embeddings
# for more information, see the OpenAI documentation:
# https://platform.openai.com/docs/guides/embeddings/embedding-models
def text_embedding_ada_002_score(references, candidates):
    if len(references) != len(candidates):
        raise ValueError("The number of references and candidates must be equal.")

    cos_sim = []

    for i in range(len(references)):
        reference = references[i]
        candidate = candidates[i]
        # get response from OpenAI API
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=[reference, candidate],
        )
        # extract embeddings for reference and candidate from response
        embedding_ref = response['data'][0]['embedding']
        embedding_can = response['data'][1]['embedding']
        # if the length of the embeddings are not the same, raise an error
        if len(embedding_ref) != len(embedding_can):
            print(f"Length of Reference Embedding: {len(embedding_ref)}")
            print(f"Length of Candidate Embedding: {len(embedding_can)}")
            raise ValueError("The length of the reference and candidate embeddings must be equal.")
        # calculate cosine similarity
        cos_sim.append(
            np.dot(embedding_ref, embedding_can) / (np.linalg.norm(embedding_ref) * np.linalg.norm(embedding_can))
        )

    return cos_sim


# calculate Exact Match
def calculate_EM(reference, candidate):
    if candidate == reference:
        return True
    else:
        return False


# calculate BLEU score
def calculate_BLEU(reference, candidate):
    metric = load("bleu")
    result = metric.compute(references=[reference], predictions=[candidate])
    return result


# calculate text-embedding-ada-002 score
def calculate_text_embedding_ada_002(reference, candidate):
    result = text_embedding_ada_002_score(references=[reference], candidates=[candidate])
    return result[0]


# calculate CodeBLEU score
def calculate_CodeBLEU(reference, candidate):
    result = calc_codebleu(
        references=[reference], predictions=[candidate], lang="python", weights=(0.25, 0.25, 0.25, 0.25)
    )
    return result


# calculate Test Ouput Match (TOM) score
# TOM = number of matched test outputs / total number of test cases
# for example, if 2 test outputs are
#   [Passed, Passed, AssertionError: outputs 10, AssertionError: outputs 4, ValueError: message]
#   [Passed, Passed, AssertionError: outputs 10, AssertionError: outputs 7, TypeError: message]
# there are 3 matched test outputs, so TOM = 3 / 5 = 0.6
# note that we are also matching the Error Messages, if Error Messages are different, then still not matched
def calculate_TOM(reference_test_outputs, candidate_test_outputs, ignore_Timeout=True):
    # initialize match_res to store the matching result for each test case,
    # and match_count to store the number of matched test outputs
    match_res = []

    for i in range(len(reference_test_outputs)):
        # if ignore_Timeout is True, ignore all test outputs that are "Time Out"
        if ignore_Timeout and (reference_test_outputs[i] == "Time Out" or candidate_test_outputs[i] == "Time Out"):
            continue
        # if not matched, record 0; otherwise record 1
        if reference_test_outputs[i] != candidate_test_outputs[i]:
            match_res.append(0)
        else:
            match_res.append(1)

    # if all test outputs are "Time Out", then the TOM score is 0.0
    if len(match_res) == 0:
        return {"tom": 0.0, "match_res": match_res}

    # compute TOM score
    TOM_score = sum(match_res) / len(match_res)
    return {"tom": TOM_score, "match_res": match_res}


# EXAMPLE USAGE:
# if you want to calculate text-embedding-ada-002 score, set the --enable_ada flag (default is False)
# python analyze_results.py --input_path
# ../tmp/starcoderbase-1b/IDChain_starcoderbase-1b_tmp0.0g_len5_pb_all_m_v1_EvalPlus-Mini-v0.1.6_reformatted.jsonl
# --chain_length 5
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--chain_length", type=int, default=1)
    parser.add_argument("--enable_ada", action="store_true")
    parser.add_argument("--no_mask", action="store_true")
    args = parser.parse_args()

    # check if chain_length is non-negative
    if args.chain_length < 0:
        raise ValueError("--chain_length must be non-negative.")

    # read the Identity Chain results
    with open(args.input_path, "r", encoding="utf-8") as reader:
        tasks = reader.readlines()

    # load raw test results to a DataFrame raw, and
    # calculate detailed trajectory for each task
    raw = pd.DataFrame()
    detail_traj = pd.DataFrame()
    # iterate over each task
    for idx, task in enumerate(tqdm(tasks)):
        task = json.loads(task)
        # fetch results from task
        task_id = task["task_id"]
        # create row to add to raw and detail_traj
        row_raw = {"task_id": task_id}
        row_traj = {"task_id": task_id}

        # summarize results for each tasks
        for i in range(args.chain_length + 1):
            PL_i_results = task[f"PL_{i}_results"]
            # get summary of results
            PL_i_results_summary = summarize_test_results(PL_i_results)
            # update row for DataFrame raw
            row_raw[f"PL_{i} Results"] = PL_i_results
            row_raw[f"PL_{i} Results Summary"] = PL_i_results_summary

        # add row_raw to the DataFrame raw
        raw = pd.concat([raw, pd.DataFrame([row_raw])], ignore_index=True)

        # calculate detailed trajectory for each task
        for i in range(args.chain_length):
            PL_i_results_summary = raw.loc[idx][f"PL_{i} Results Summary"]
            PL_i_plus_1_results_summary = raw.loc[idx][f"PL_{i+1} Results Summary"]

            # get summary of trajectory
            summary = ""
            if PL_i_results_summary == "Passed":
                if PL_i_plus_1_results_summary == "Passed":
                    summary = "Pass-Pass"
                else:
                    summary = "Pass-Fail"
            else:
                if PL_i_plus_1_results_summary == "Passed":
                    summary = "Fail-Pass"
                else:
                    summary = "Fail-Fail"

            # process NL_0 before calculating metrics
            # remove the function signature from NL_0
            if i == 0:
                ref_nl = task[f"NL_{i}"].replace(task["function_signature"], "")
                # replace function name with "func" the parameter if we masked the function name running IdentityChain
                # by default, we assume it's masked, but if --no_mask is specified, then we do nothing
                if not args.no_mask:
                    ref_nl = ref_nl.replace(task["function_name"], "func")
            else:
                ref_nl = task[f"NL_{i}"]

            # calculate stats of trajectory
            em_NL = calculate_EM(ref_nl, task[f"NL_{i+1}"])
            em_PL = calculate_EM(task[f"PL_{i}"], task[f"PL_{i+1}"])
            # if --enable_ada is False, then ada_002 is by default 0.0
            ada_002 = 0.0
            if em_NL:
                # no need to calculate BLEU-NL if EM-NL is True
                bleu_NL = 1.0
                # only calculate ada-002 if --enable_ada is True
                if args.enable_ada:
                    ada_002 = 1.0
            else:
                # calculate BLEU-NL
                bleu_NL = calculate_BLEU(ref_nl, task[f"NL_{i+1}"])["bleu"]
                # only calculate ada-002 if --enable_ada is True
                if args.enable_ada:
                    ada_002 = calculate_text_embedding_ada_002(ref_nl, task[f"NL_{i+1}"])
            if em_PL:
                # no need to calculate TOM and CodeBLEU if EM-PL is True
                tom = 1.0
                codebleu = 1.0
            else:
                codebleu = calculate_CodeBLEU(task[f"PL_{i}"], task[f"PL_{i+1}"])["codebleu"]
                tom = calculate_TOM(task[f"PL_{i}_results"], task[f"PL_{i+1}_results"], ignore_Timeout=True)["tom"]

            # update row for DataFrame detail_traj
            row_traj[f"Step {i}-{i+1} Summary"] = summary
            row_traj[f"Step {i}-{i+1} EM-NL"] = str(int(em_NL))
            row_traj[f"Step {i}-{i+1} BLEU-NL"] = str(bleu_NL)
            row_traj[f"Step {i}-{i+1} ada-002"] = str(ada_002)
            row_traj[f"Step {i}-{i+1} EM-PL"] = str(int(em_PL))
            row_traj[f"Step {i}-{i+1} TOM"] = str(tom)
            row_traj[f"Step {i}-{i+1} CodeBLEU"] = str(codebleu)

        # add row_traj to the DataFrame detail_traj
        detail_traj = pd.concat([detail_traj, pd.DataFrame([row_traj])], ignore_index=True)

    # calculate detailed averaged statistics for each step in the Identity Chain
    detail_avg = pd.DataFrame()
    # iterate over each step
    for i in range(args.chain_length):
        # create row to store stats
        row = {
            "Step": f"{i}-{i+1}",
            "Pass-Pass Count": 0,
            "Pass-Pass EM-NL": 0.0,
            "Pass-Pass BLEU-NL": 0.0,
            "Pass-Pass ada-002": 0.0,
            "Pass-Pass EM-PL": 0.0,
            "Pass-Pass TOM": 0.0,
            "Pass-Pass CodeBLEU": 0.0,
            "Pass-Fail Count": 0,
            "Pass-Fail EM-NL": 0.0,
            "Pass-Fail BLEU-NL": 0.0,
            "Pass-Fail ada-002": 0.0,
            "Pass-Fail EM-PL": 0.0,
            "Pass-Fail TOM": 0.0,
            "Pass-Fail CodeBLEU": 0.0,
            "Fail-Fail Count": 0,
            "Fail-Fail EM-NL": 0.0,
            "Fail-Fail BLEU-NL": 0.0,
            "Fail-Fail ada-002": 0.0,
            "Fail-Fail EM-PL": 0.0,
            "Fail-Fail TOM": 0.0,
            "Fail-Fail CodeBLEU": 0.0,
            "Fail-Pass Count": 0,
            "Fail-Pass EM-NL": 0.0,
            "Fail-Pass BLEU-NL": 0.0,
            "Fail-Pass ada-002": 0.0,
            "Fail-Pass EM-PL": 0.0,
            "Fail-Pass TOM": 0.0,
            "Fail-Pass CodeBLEU": 0.0,
        }

        # calculate stats for each of the 4 cases
        for summary in ["Pass-Pass", "Pass-Fail", "Fail-Fail", "Fail-Pass"]:
            filtered_df = detail_traj.loc[detail_traj[f"Step {i}-{i+1} Summary"] == summary]
            # count the number of tasks in this case
            row[f"{summary} Count"] = str(len(filtered_df))
            # if no task in this case, skip it, all metrics will be by default 0.0
            if len(filtered_df) == 0:
                continue
            # calculate average of each metric
            row[f"{summary} EM-NL"] = str(filtered_df[f"Step {i}-{i+1} EM-NL"].astype(float).mean())
            row[f"{summary} BLEU-NL"] = str(filtered_df[f"Step {i}-{i+1} BLEU-NL"].astype(float).mean())
            row[f"{summary} ada-002"] = str(filtered_df[f"Step {i}-{i+1} ada-002"].astype(float).mean())
            row[f"{summary} EM-PL"] = str(filtered_df[f"Step {i}-{i+1} EM-PL"].astype(float).mean())
            row[f"{summary} TOM"] = str(filtered_df[f"Step {i}-{i+1} TOM"].astype(float).mean())
            row[f"{summary} CodeBLEU"] = str(filtered_df[f"Step {i}-{i+1} CodeBLEU"].astype(float).mean())

        # for debugging
        # print(row)

        # add to the DataFrame detail_avg
        detail_avg = pd.concat([detail_avg, pd.DataFrame([row])], ignore_index=True)

    # calculate summarized statistics for each step in the Identity Chain
    summary = pd.DataFrame(columns=["Step", "Total Count", "SSC Count", "SSC_N", "SC Count", "SC_N"])
    # iterate over each step
    for i in range(args.chain_length + 1):
        row = {"Step": str(i)}
        # count Total Tasks, only consider tasks that have a PL_i
        total_count = len(raw[raw[f"PL_{i} Results Summary"] != "NA"])
        row["Total Count"] = str(total_count)
        # count FixedPass and FixedPoint
        if i == 0:
            PL_0_fpass_count = len(raw[raw["PL_0 Results Summary"] == "Passed"])
            row["SSC Count"] = str(PL_0_fpass_count)
            row["SSC_N"] = str(PL_0_fpass_count / total_count)
            row["SC Count"] = ""
            row["SC_N"] = ""
        else:
            # create filter condition for FixedPass
            condition_fpass = detail_traj["Step 0-1 Summary"] == "Pass-Pass"
            for j in range(2, i + 1):
                condition_fpass = condition_fpass & (detail_traj[f"Step {j-1}-{j} Summary"] == "Pass-Pass")
            # create filter condition for FixedPoint
            condition_fpoint = detail_traj["Step 0-1 TOM"] == "1.0"
            for j in range(2, i + 1):
                condition_fpoint = condition_fpoint & (detail_traj[f"Step {j-1}-{j} TOM"] == "1.0")
            # if Pass at all N steps, then it is a FixedPass@N
            PL_i_fpass_count = len(detail_traj[condition_fpass])
            row["SSC Count"] = str(PL_i_fpass_count)
            row["SSC_N"] = str(PL_i_fpass_count / total_count)
            # if TOM = 1.0 at all N steps, then it is a FixedPoint@N
            PL_i_fpoint_count = len(detail_traj[condition_fpoint])
            row["SC Count"] = str(PL_i_fpoint_count)
            row["SC_N"] = str(PL_i_fpoint_count / total_count)

        summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)

    # for debugging
    print(summary)

    # write aggregated statistics and raw results to excel
    if args.output_path is None:
        output_path = args.input_path.replace(".jsonl", ".xlsx")
    else:
        output_path = args.output_path
    with pd.ExcelWriter(output_path) as writer:
        summary.to_excel(writer, sheet_name="summary", index=False, startrow=1, startcol=1, engine="xlsxwriter")
        detail_avg.to_excel(writer, sheet_name="detail-avg", index=False, startrow=1, startcol=1, engine="xlsxwriter")
        detail_traj.to_excel(writer, sheet_name="detail-traj", index=False, startrow=1, startcol=1, engine="xlsxwriter")
        raw.to_excel(writer, sheet_name="raw", index=False, startrow=1, startcol=1, engine="xlsxwriter")


if __name__ == "__main__":
    main()
