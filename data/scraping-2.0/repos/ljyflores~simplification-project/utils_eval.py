import json
import nltk
import numpy as np
import openai
import re
import textstat
import time

from collections import Counter
from easse.fkgl import corpus_fkgl
from easse.sari import corpus_sari
from evaluate import load
from nltk.util import ngrams
from rouge_score import rouge_scorer, scoring
from typing import List, Dict

metric_bertscore = load("bertscore")
metric_sari = load("sari")


def get_entities(input, ner_model_lst, linker_lst=None):

    SEMTYPES = ["T023","T028","T046","T047","T048",
                "T059","T060","T061","T074","T109",
                "T116","T121","T122","T123","T125",
                "T129","T184","T191","T195"]

    output_entities = set()

    if type(ner_model_lst) is not list:
        ner_model_lst = [ner_model_lst]
        linker_lst    = [linker_lst]

    for (ner_model, linker) in zip(ner_model_lst, linker_lst):
        entity_lst = ner_model(input).ents

        if "scispacy_linker" in ner_model.pipe_names:
            filtered_entities = []
            for e in set(entity_lst):
                if len(e._.kb_ents) > 0:
                    umls_ent_id, _ = e._.kb_ents[0]  # Get top hit from UMLS
                    umls_ent  = linker.kb.cui_to_entity[umls_ent_id]  # Get UMLS entity
                    umls_semt = umls_ent[3]
                    if any([t in SEMTYPES for t in umls_semt]):
                        e = str(e)
                        if e not in filtered_entities:
                            filtered_entities.append(e)
            output_entities.update(set(filtered_entities))
        else:
            output_entities.update(set([str(e) for e in entity_lst]))

    return output_entities

def check_unsupported_entities(input, output, ner_model_lst, linker_lst):
    input_entities  = get_entities(input.lower(), ner_model_lst, linker_lst)
    output_entities = get_entities(output.lower(), ner_model_lst, linker_lst)

    input_entities  = set(input_entities)
    output_entities = set(output_entities)

    differences = list(output_entities.difference(input_entities))
    if len(differences) > 0:
        return True, differences
    else:
        return False, differences

def add_newline_to_end_of_each_sentence(s):
    """This was added to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
    s = s.replace("\n", "")
    return "\n".join(nltk.sent_tokenize(s))

def check_n_gram_overlap(
        sources: List[str],
        predictions: List[str]
        ):
    
    def extract_ngrams(data, num):
        n_grams = ngrams(nltk.word_tokenize(data), num)
        return [ ' '.join(grams) for grams in n_grams]

    result = []
    for (s,p) in zip(sources, predictions):
        ngrams_4 = extract_ngrams(p, 4)
        num_overlap_ngrams = [1 if ngram in s else 0 for ngram in ngrams_4]
        result.append(np.mean(num_overlap_ngrams))
    return np.mean(result)


def calculate_g_eval(sources: List[str], 
                     predictions: List[str], 
                     model: str, 
                     **kwargs):

    openai.api_key_path = "openai_key"

    result = []
    for document, summary in zip(sources, predictions):
        time.sleep(1.0)
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Your task is to rate the summary on one metric.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Human Evaluation of Text Summarization Systems: \n"
                            "Factual Consistency: Does the summary have untruthful or "
                            "misleading facts that are not supported by the source text? \n"
                            f"Source Text: {document} \n"
                            f"Summary: {summary} \n"
                            "Does the summary contain factual inconsistencies? \n"
                            "Answer: "
                        ),
                    },
                ],
                **kwargs,
            )
        except Exception as e:
            response = {}
            print(e)
        result.append(response)
        # simplified_sen = response["choices"][0]["message"]["content"]
    return result

def calculate_g_explain(sources: List[str], 
                     predictions: List[str], 
                     model: str, 
                     **kwargs):

    openai.api_key_path = "openai_key"

    result = []
    for document, summary in zip(sources, predictions):
        time.sleep(0.3)
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Your task is to rate the summary on one metric.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Human Evaluation of Text Summarization Systems: \n"
                            "Factual Consistency: Does the summary have untruthful or "
                            "misleading facts that are not supported by the source text? \n"
                            f"Source Text: {document} \n"
                            f"Summary: {summary} \n"
                            "Does the summary contain factual inconsistencies? \n"
                            "Answer: \n"
                            "If yes, why: "
                        ),
                    },
                ],
                **kwargs,
            )
        except Exception as e:
            response = {}
            print(e)
        result.append(response)
        # simplified_sen = response["choices"][0]["message"]["content"]
    return result

# def calculate_bleurt(predictions, labels, scorer):
#     bleurt = []
#     for (p,l) in zip(predictions, labels):
#         candidates = [p]
#         references = l
#         scores = scorer.score(references=references, candidates=candidates)
#         bleurt.append(scores[0])
#     return np.mean(bleurt)


def calculate_rouge(
    predictions: List[str],
    references: List[List[str]],
):
    """Calculate rouge using rouge_scorer package.
    Args:
        pred_lns: list of summaries generated by model
        tgt_lns: list of groundtruth summaries (e.g. contents of val.target)
    Returns:
         Dict[score: value] if aggregate else defaultdict(list) keyed by rouge_keys
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
    )
    aggregator = scoring.BootstrapAggregator()
    for pred, tgt in zip(predictions, references):
        pred = add_newline_to_end_of_each_sentence(pred)
        tgt = [add_newline_to_end_of_each_sentence(s) for s in tgt]
        scores = scorer.score_multi(tgt, pred)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}


def get_readability_score(text: str, 
                          metric = "flesch_reading_grade"):
    """get the readability score and grade level of text"""
    if metric == "flesch_reading_ease":
        score = textstat.flesch_reading_ease(text)
        if score > 90:
            grade = "5th grade"
        elif score > 80:
            grade = "6th grade"
        elif score > 70:
            grade = "7th grade"
        elif score > 60:
            grade = "8th & 9th grade"
        elif score > 50:
            grade = "10th to 12th grade"
        elif score > 30:
            grade = "college"  # Collge student = average 13th to 15th grade
        elif score > 10:
            grade = "college graduate"
        else:
            grade = "professional"
        return score, grade

    elif metric == "flesch_kincaid_grade":
        score = textstat.flesch_kincaid_grade(
            text
        )  # Note: this score can be negative like -1
        grade = round(score)
        if grade > 16:
            grade = "college graduate"  # Collge graduate: >16th grade
        elif grade > 12:
            grade = "college"
        elif grade <= 4:
            grade = "4th grade or lower"
        else:
            grade = f"{grade}th grade"
        return score, grade

    elif metric == "ari":
        sents = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        num_sents = len(sents)
        num_words = len(words)
        num_chars = sum(len(w) for w in words)
        score = (
            4.71 * (num_chars / float(num_words))
            + 0.5 * (float(num_words) / num_sents)
            - 21.43
        )
        return score, "None"

    # elif metric == 'SMOG': # Note: SMOG formula needs at least three ten-sentence-long samples for valid calculation
    #     score = textstat.smog_index(text)
    #     grade = round(score)
    #     if grade > 16:
    #         grade = 'college graduate'
    #     elif grade > 12:
    #         grade = 'college'
    #     else:
    #         grade = f"{grade}th grade"
    #     return score, grade

    elif metric == "dale_chall":
        score = textstat.dale_chall_readability_score(text)
        if score >= 10:
            grade = "college graduate"
        elif score >= 9:
            grade = "college"  # Collge student = average 13th to 15th grade
        elif score >= 8:
            grade = "11th to 12th grade"
        elif score >= 7:
            grade = "9th to 10th grade"
        elif score >= 6:
            grade = "7th to 8th grade"
        elif score >= 5:
            grade = "5th to 6th grade"
        else:
            grade = "4th grade or lower"
        return score, grade

    elif metric == "gunning_fog":
        score = textstat.gunning_fog(text)
        grade = round(score)
        if grade > 16:
            grade = "college graduate"
        elif grade > 12:
            grade = "college"
        elif grade <= 4:
            grade = "4th grade or lower"
        else:
            grade = f"{grade}th grade"
        return score, grade

    else:
        raise ValueError(f"Unknown metric {metric}")

def calculate_bertscore(predictions: List[str], 
                        references: List[List[str]]):
    result_bert = []
    for (pred, label) in zip(predictions, references):
        result_bert_temp = metric_bertscore.compute(
            predictions=[pred] * len(label), 
            references=label, 
            lang="en"
        )
        if type(result_bert_temp["f1"]) == list:
            result_bert.append(result_bert_temp["f1"][0])
        else:
            result_bert.append(result_bert_temp["f1"])
    return np.mean(result_bert)

def calculate_sari(sources: List[str], 
                   predictions: List[str], 
                   references: List[List[str]]):
    result_sari = metric_sari.compute(sources=sources, 
                                        predictions=predictions, 
                                        references=references)[
        "sari"
    ]
    return result_sari

def calculate_fkgl_easse(predictions: List[str]):
    return corpus_fkgl(sentences=predictions,
                       tokenizer="13a")

def calculate_sari_easse(sources: List[str], 
                         predictions: List[str], 
                         references: List[List[str]]):
    return corpus_sari(orig_sents=sources, 
                       sys_sents=predictions, 
                       refs_sents=references,
                       lowercase=True,
                       tokenizer="13a")

def clean_string(s: str):
    s = s.replace("-lrb-"," ").replace("-rrb-", " ")
    s = s.replace("<s>", "").replace("</s>", "").replace("<pad>", "")
    return re.sub(" +", " ", s)

def compute_metrics(
    sources: List[str],
    predictions: List[str],
    labels: List[List[str]],
    metrics: List[str],
) -> Dict:
    """Test docstring.

    Args:
        sources (list[str]): List of input sources
        predictions (list[str]): List of output sources
        labels (list[list[str]]): List of list of reference strings
    Returns:
        dict: Output of computed metrics

    """
    assert type(sources) == list and type(sources[0]) == str, print(
        "Sources should be a list of strings"
    )
    assert type(predictions) == list and type(predictions[0]) == str, print(
        "Predictions should be a list of strings"
    )
    assert type(labels) == list and type(labels[0]) == list, print(
        "Labels should be a list of LISTS, each containing the labels"
    )

    # Clean inputs
    sources = [clean_string(s) for s in sources]
    predictions = [clean_string(s) for s in predictions]
    labels = [[clean_string(s) for s in lst] for lst in labels]

    result = {}

    if "rouge" in metrics:
        result_rouge = calculate_rouge(predictions, labels)
        for key in result_rouge.keys():
            result[key] = result_rouge[key]
    if "sari" in metrics:
        result_sari = calculate_sari(
            sources=sources, predictions=predictions, references=labels
        )
        result["sari"] = result_sari
    # if "bleurt" in metrics:
    #     scorer = score.BleurtScorer()
    #     result["bleurt"] = calculate_bleurt(predictions=predictions,
    #                                         labels=labels,
    #                                         scorer=scorer)
    if "sari_easse" in metrics:
        labels_transposed = [l for l in zip(*labels)]
        result["sari_easse"] = calculate_sari_easse(
            sources=sources, predictions=predictions, references=labels_transposed
        )
        
    if "fkgl_easse" in metrics:
        result["fkgl_easse"] = calculate_fkgl_easse(
            predictions=predictions
        )

    if "bert_score" in metrics:
        result["bert_score"] = calculate_bertscore(
            predictions=predictions, 
            references=labels
        )

    if "bert_score_l" in metrics:
        result["bert_score_l"] = calculate_bertscore(
            predictions=predictions, 
            references=[[item] for item in sources]
        )

    readability_dict = {}
    for metric in [
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "ari",
        "dale_chall",
        "gunning_fog",
    ]:
        if metric in metrics:
            result_readability = list(
                map(lambda s: get_readability_score(s, metric=metric), predictions)
            )
            readability_dict[f"{metric}_counts"] = Counter(
                list(map(lambda item: item[1], result_readability))
            )
            readability_dict[f"{metric}_score"] = np.mean(
                list(map(lambda item: item[0], result_readability))
            )
    result.update(readability_dict)

    if ("geval-3.5" in metrics) or ("geval-4" in metrics):
        if "geval-3.5" in metrics:
            geval_dict = calculate_g_eval(
                sources, predictions, model="gpt-3.5-turbo", temperature=0, max_tokens=1
            )
        else:
            geval_dict = calculate_g_eval(
                sources, predictions, model="gpt-4", n=1, temperature=0, top_p=1
            )
        geval_answers = [
            d["choices"][0]["message"]["content"] if "choices" in d else ""
            for d in geval_dict
        ]
        result["geval"] = (geval_answers, Counter(geval_answers))

    if ("gexplain-3.5" in metrics) or ("gexplain-4" in metrics):
        if "gexplain-3.5" in metrics:
            geval_dict = calculate_g_explain(
                sources, predictions, model="gpt-3.5-turbo", temperature=0
            )
        else:
            geval_dict = calculate_g_explain(
                sources, predictions, model="gpt-4", n=1, temperature=0, top_p=1
            )
        geval_answers = [
            d["choices"][0]["message"]["content"] if "choices" in d else ""
            for d in geval_dict
        ]
        result["geval"] = geval_answers

    if "check_entities" in metrics:
        import spacy
        import scispacy
        from scispacy.linking import EntityLinker
        ner_model_web = spacy.load("en_core_web_lg")
        ner_model_sci = spacy.load("en_core_sci_lg")
        ner_model_sci.add_pipe(
            "scispacy_linker",
            config={"resolve_abbreviations": True, "linker_name": "umls"},
        )
        linker_sci = ner_model_sci.get_pipe("scispacy_linker")
        ner_lst    = [ner_model_sci, ner_model_web]
        linker_lst = [linker_sci, None]
        
        check_entity_results = [
            check_unsupported_entities(i, o, ner_lst, linker_lst) \
                for (i,o) in zip(sources, predictions)]
        num_true = Counter([item[0] for item in check_entity_results])[True]
        result["check_entities"] = f"{num_true}/{len(check_entity_results)}"
        result["check_entities_ents"] = [item[1] for item in check_entity_results]

    if "check_overlap" in metrics:
        result["check_overlap"] = check_n_gram_overlap(sources, predictions)

    return {k: round(v, 4) if type(v) in [float, int] else v for k, v in result.items()}
