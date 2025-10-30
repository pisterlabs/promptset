TypeError: the JSON object must be str, bytes or bytearray, not dict
with open("classification.json", "w") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
import re
import json
from collections import Counter
import logging
from sllim import chat, system, user, estimate, logger
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# --- Setup logging ---
logger.setLevel(logging.WARNING)

logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)

# --- Prompts ---
system_prompt = """<YOUR FULL SYSTEM PROMPT HERE>"""  # (shortened for brevity)
user_prompt = """Prompt:\n\n\"\"\"\n{prompt}\n\"\"\""""

# --- Function to classify a prompt ---
def classify_pattern(prompt):
    """Call the LLM to classify a prompt into a category."""
    try:
        response = chat(
            messages=[
                system(system_prompt),
                user(user_prompt.format(prompt=prompt)),
            ],
            model="gpt-4-1106-preview",
            max_tokens=40,
            temperature=0,
            response_format={"type": "json_object"},
        )

        # Depending on sllim, adjust this line:
        # If `response` is already dict-like, return it directly
        if isinstance(response, dict):
            return response
        elif hasattr(response, "content"):
            return json.loads(response.content)
        else:
            return json.loads(response)
    except Exception as e:
        logging.error(f"Error classifying prompt: {e}")
        return {"category": 7, "pattern": "None of the above", "error": str(e)}

# --- Helper to process a dataset ---
def process_dataset(filename, output_file, count=200):
    """Process prompts from a JSON file and save classification results."""
    with open(filename) as f:
        strings = json.load(f)

    results = []
    for prompt in tqdm(strings[:count], desc=f"Processing {filename}"):
        r = classify_pattern(prompt)
        r["prompt"] = prompt
        results.append(r)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # Count category frequencies
    base = {i: 0 for i in range(1, 8)}
    categories_frequency = sorted(
        (base | Counter([r.get("category", 7) for r in results])).items(),
        key=lambda x: x[0],
    )

    return categories_frequency

# --- Run for each dataset ---
datasets = {
    "strings_1k.json": "classification_orig.json",
    "strings_devgpt_1k.json": "classification_dg.json",
    "strings_plus_1k.json": "classification_plus.json",
}

frequencies = {}
for infile, outfile in datasets.items():
    frequencies[infile] = process_dataset(infile, outfile)

# --- Plotting ---
categories_str = [f"Category {i}" for i in range(1, 8)]
bar_width = 0.25
index = np.arange(len(categories_str))

fig, ax = plt.subplots(figsize=(10, 7))

# Unpack frequencies
orig = [freq for _, freq in frequencies["strings_1k.json"]]
dg = [freq for _, freq in frequencies["strings_devgpt_1k.json"]]
plus = [freq for _, freq in frequencies["strings_plus_1k.json"]]

ax.bar(index, orig, bar_width, label="Set a-d", color="skyblue", edgecolor="black")
ax.bar(index + bar_width, dg, bar_width, label="DevGPT", color="salmon", edgecolor="black")
ax.bar(index + 2 * bar_width, plus, bar_width, label="Set e", color="palegreen", edgecolor="black")

ax.set_xlabel("Category", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_xticks(index + bar_width)
ax.set_xticklabels(categories_str)
ax.legend()

plt.tight_layout()
plt.show()
