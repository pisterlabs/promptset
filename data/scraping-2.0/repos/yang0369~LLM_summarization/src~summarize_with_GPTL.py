import json
import sys
import time

import jsonlines
import tensorflow as tf
from langchain.embeddings import VertexAIEmbeddings
from load_and_chunk import ProcessingPipeline
from rouge_score import rouge_scorer
from summarize_long import summarize_long_text_by_llama2


def load_json_SLR_dataset(dir):

    with jsonlines.open(dir, 'r') as reader:
        input_headers = "judgment"
        testset = []
        for sample in reader:
            d = {}
            d["citation_number"] = sample["citation_number"]
            d["neutral_citation"] = sample["neutral_citation"]

            d["input"] = sample[input_headers]

            d["truth"] = 'Facts\n\n' + sample['headnotes']['facts'] + '\n\nHoldings\n\n' + sample['headnotes']['holdings']

            testset.append(d)

    return testset


def rouge(target, prediction, score_keys=None):
    """Computes rouge score.

    Args:
      target: string
      prediction: string
      score_keys: list of strings with the keys to compute.
    Returns:
      dict with score_key: rouge score of target and prediction
    """

    if score_keys is None:
        score_keys = ["rouge1", "rouge2", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer(score_keys)

    def _prepare_summary(summary):
        # Make sure the summary is not bytes-type
        # Add newlines between sentences so that rougeLsum is computed correctly.
        summary = summary.replace(" . ", " .\n")
        return summary

    target = _prepare_summary(target)
    prediction = _prepare_summary(prediction)
    return scorer.score(target=target, prediction=prediction)


def compute_metrics(reference_text, hypothesis_text):
    """ Compute ROUGE and Semantic-Similarity scores for a pair of text strings
        :return: a dict containg scores: Rouge-1,2,L and semantic-similarity
    """
    scores = {
        "rouge_1_r": 0.0,
        "rouge_1_p": 0.0,
        "rouge_1_f": 0.0,
        "rouge_2_r": 0.0,
        "rouge_L_r": 0.0,
        "semantic_similarity": 0.0,
    }

    if hypothesis_text.split():
        # Compute ROUGE scores
        rouge_scores = rouge(reference_text, hypothesis_text)
        scores['rouge_1_r'] = float(f"{rouge_scores['rouge1'].recall:.3f}")
        scores['rouge_1_p'] = float(f"{rouge_scores['rouge1'].precision:.3f}")
        scores['rouge_1_f'] = float(f"{rouge_scores['rouge1'].fmeasure:.3f}")
        scores['rouge_2_r'] = float(f"{rouge_scores['rouge2'].recall:.3f}")
        scores['rouge_L_r'] = float(f"{rouge_scores['rougeLsum'].recall:.3f}")

        # Compute Semantic Similarity scores
        # use_score = use_scores([reference_text], [hypothesis_text])
        # scores['semantic_similarity'] = float(f"{use_score[0][0]:.3f}")

    return scores


def average_score(scores: list):
    """ Compute the average of a list of numbers """
    return sum(scores) / len(scores)


def compute_average_scores(all_scores: list):
    """ Compute average ROUGE and Semantic-Similarity scores for the whole testset.
        :return: a dict containing average scores: Rouge-1,2,L and semantic-similarity
    """
    # Buffers for keeping scores
    scores_rouge_1_r = []
    scores_rouge_1_p = []
    scores_rouge_1_f = []
    scores_rouge_2_r = []
    scores_rouge_L_r = []

    # scores_semantic = []
    r_ratio = []

    # Place individual scores to according buffers
    for scores in all_scores:
        scores_rouge_1_r.append(scores["rouge_1_r"])
        scores_rouge_1_p.append(scores["rouge_1_p"])
        scores_rouge_1_f.append(scores["rouge_1_f"])
        scores_rouge_2_r.append(scores["rouge_2_r"])
        scores_rouge_L_r.append(scores["rouge_L_r"])

        # scores_semantic.append(scores["semantic_similarity"])
        if 'ratio' in scores.keys():
            r_ratio.append(1/scores['ratio'])

    # Compute average scores
    average_scores = {}
    average_scores['rouge_1_r'] = float(f"{average_score(scores_rouge_1_r):.3f}")
    average_scores['rouge_1_p'] = float(f"{average_score(scores_rouge_1_p):.3f}")
    average_scores['rouge_1_f'] = float(f"{average_score(scores_rouge_1_f):.3f}")
    average_scores['rouge_2_r'] = float(f"{average_score(scores_rouge_2_r):.3f}")
    average_scores['rouge_L_r'] = float(f"{average_score(scores_rouge_L_r):.3f}")

    # average_scores['semantic_similarity'] = float(f"{average_score(scores_semantic):.3f}")
    if len(r_ratio) > 0:
        average_scores['ratio'] = float(f"{1/average_score(r_ratio):.1f}")

    return average_scores


def extract_text_from_json(jsondict: list) -> str:
    """ Extract Judgment text from SAL JSON dict data.
        Output:
            text string for judgment
    """
    output_buf = []

    for data in jsondict:

        # Extract header text
        output_buf.append(data['header']['text'])

        # Extract paragraph text
        for parag_data in data['paragraphs']:
            if parag_data['paragraph_number']:
                output_buf.append(parag_data['paragraph_number'] + ' ' + parag_data['text'])
            else:
                output_buf.append(parag_data['text'])

        # Extract table text
        for table_data in data['tables']:
            rows = [row.replace('\t', ' | ') for row in table_data]
            output_buf.append('\n'.join(rows))

    text = '\n\n'.join(output_buf)
    return text


def summarize_long_by_clustering(text_or_jsondict):

    pro = ProcessingPipeline(VertexAIEmbeddings())
    chunks = pro.process_document(text_or_jsondict)
    sum_gen_d = summarize_long_text_by_llama2(chunks)

    return sum_gen_d


def benchmark(testset: list, task: str):

    # Buffers to store results
    output_lst = []
    all_scores = []

    # Benchmark loop
    for n, data in enumerate(testset):
        generated_text = summarize_long_by_clustering(data["input"])["summary"]

        print(f"{80 * '-'}\n#{n + 1}: {generated_text}", flush=True)

        if data["truth"]:  # ignore the test item that has empty truth
            # Compute evaluation scores
            with tf.device('/CPU:0'):
                # to calculate rouge score of large models, we have to raise the recursion limit
                sys.setrecursionlimit(5000 * 5000)
                scores = compute_metrics(data["truth"], generated_text)
                if task == "sm":
                    scores['ratio'] = float(f'{(len(data["input"].split()) / len(generated_text.split())):.1f}')

                sys.setrecursionlimit(1000)
            all_scores.append(scores)

            # Save info for output
            d = {}
            d["sn"] = n + 1
            d["truth"] = data["truth"]
            d["output"] = generated_text
            d["scores"] = scores
            for key in data.keys():
                if key not in ['input', 'truth']:
                    d[key] = data[key]

            output_lst.append(d)

            print(f"scores: {scores}", flush=True)

        del generated_text

    # Compute average scores for the whole testset
    average_scores = compute_average_scores(all_scores)
    return average_scores, output_lst


if __name__ == "__main__":

    # Load testset file
    input_dir = "/home/kewen_yang/gptx2/SLR_Short.jsonl"
    testset = load_json_SLR_dataset(input_dir)

    start = time.time()
    average_scores, output_lst = benchmark(testset, "sm")

    end = time.time()
    average_time = float(f"{(end - start) / len(testset):.1f}")
    print(f"\nBenchmark average time: {average_time} seconds")
    print(json.dumps(average_scores, indent=4))
