from utils import parse_pdf, parse_results
import openai
import numpy as np
import string
from sklearn.cluster import KMeans


def extract_keywords_from_prompt(chat_completion: openai.ChatCompletion, prompt: str):
    model_output = chat_completion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "system", "content":f"Your job is to extract key words from text. Generic words should never be extracted, only topic specific words."},
            {"role": "user", "content":f"Extract the keywords from the following instruction. Output a single list of comma separated values only once. \n\n {prompt}"}
        ]
    )
    output = parse_results(model_output)
    return output.split(",")


def generate_single_summary(chat_completion: openai.ChatCompletion, input_text: str, summary_prompt: str) -> str:
    model_output = chat_completion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "system", "content":f"Your job is to summarize text based on a prompt. If relevant data is not found, return nothing."},
            {"role": "user", "content":f"{input_text} \n\n {summary_prompt}"}
        ]
    )
    output = parse_results(model_output)
    return output


def generate_summary(
        chat_completion: openai.ChatCompletion,
        text_body: str,
        prompt: str,
        buffer: int = 600
) -> str:
    keywords = extract_keywords_from_prompt(chat_completion, prompt)
    keywords = [k.translate(str.maketrans('', '', string.punctuation)).strip().lower() for k in keywords]
    print(f"{len(keywords)} keywords found...")
    kw = []
    [[kw.append(w) for w in word.split(" ")] for word in keywords]
    arr_text = np.array(text_body.lower().split())

    print("Matching keywords in text...")
    idxs = np.array([])
    max_idx = len(arr_text)
    for keyw in kw:
        kw_idxs = np.where(arr_text == keyw)[0] / max_idx
        idxs = np.concatenate([idxs, kw_idxs])

    if len(idxs) == 0:
        return "This information is not available in the file."

    print("Clustering...")
    kmeans = KMeans(n_clusters = len(keywords))
    _ = kmeans.fit_predict(idxs.reshape(-1, 1))
    centroid_idxs = list((kmeans.cluster_centers_ * len(arr_text)).astype(int).reshape(-1))

    print("Generating summary...")
    summaries = []
    for centroid_idx in centroid_idxs:
        text_input = list(arr_text[max(0, centroid_idx - buffer):min(len(arr_text), centroid_idx + buffer)])
        text_input = " ".join(text_input)
        summary = generate_single_summary(chat_completion, text_input, prompt)
        summaries.append(summary)
    summaries = " ".join(summaries)
    final_summary = generate_single_summary(chat_completion, summaries, prompt)
    return final_summary
