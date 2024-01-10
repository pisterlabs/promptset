import sys
import os
import openai
import json
import numpy as np
#Work around for relative import - clean up later
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from eval.clip_scores import CLIPEval
from eval.distinct_2 import DistinctEval
from eval.imageability import ImageabilityEval
from tqdm import tqdm

def gen_prompt(title):
    """

    """
    return f"Create an English poem with title:\n\n{title}"

def main():
    with open("data/test_poem2img_gpt3.json", "r") as file:
        test_data = json.load(file)["poem2img"]
    
    distinct_eval = DistinctEval()
    imageability_eval = ImageabilityEval()
    clip_eval = CLIPEval()

    #Load extractor and Model
    openai.api_key = "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"

    # gen_poem_ut = GenPoemUtils("models/gpt_mlp/checkpoints/ki-009.pt")

    results = {
        "Mean Distinct-2 Score" : 0,
        "Median Distinct-2 Score" : 0,
        "Max Distinct-2 Score" : 0,
        "Min Distinct-2 Score" : 0,
        "Mean Imageability Score" : 0,
        "Median Imageability Score" : 0,
        "Max Imageability Score" : 0,
        "Min Imageability Score" : 0,
        "Poems" : [],
        "Distinct-2 Scores" : [],
        "Imageability Scores" : []
    }

    clip_pred = []
    
    for i in tqdm(range(len(test_data))):
        gpt_prompt = gen_prompt(test_data[i]["title"])

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=gpt_prompt,
            temperature=0.5,
            max_tokens=77,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        pred_poem = response['choices'][0]['text']
        results["Poems"].append(pred_poem)

        # print("==== Distinct-2 Score ====")
        distinct_score = distinct_eval.score_poem(pred_poem)
        results["Distinct-2 Scores"].append(distinct_score)

        # print("==== Imageability Score ====")
        imageability_score = imageability_eval.score_poem(pred_poem)
        results["Imageability Scores"].append(imageability_score)
    
    with open("models/gpt3/scores/score_final.json", "w") as f:
        json.dump(results, f)

    results["Mean Distinct-2 Score"] = np.mean(np.array(results["Distinct-2 Scores"]))
    results["Median Distinct-2 Score"] = np.median(np.array(results["Distinct-2 Scores"]))
    results["Min Distinct-2 Score"] = np.min(np.array(results["Distinct-2 Scores"]))
    results["Max Distinct-2 Score"] = np.max(np.array(results["Distinct-2 Scores"]))

    results["Mean Imageability Score"] = np.mean(np.array(results["Imageability Scores"]))
    results["Median Imageability Score"] = np.median(np.array(results["Imageability Scores"]))
    results["Min Imageability Score"] = np.min(np.array(results["Imageability Scores"]))
    results["Max Imageability Score"] = np.max(np.array(results["Imageability Scores"]))

    with open("models/gpt3/scores/score_final.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    sys.exit(main())