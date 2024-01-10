from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import openai

# Dataset
categories = ["rec.sport.baseball", "rec.sport.hockey"]
sports_dataset = fetch_20newsgroups(
    subset="train", shuffle=True, random_state=42, categories=categories
)

# EDA
## Total examples: 1197, Baseball examples: 597, Hockey examples: 600
len_all, len_baseball, len_hockey = (
    len(sports_dataset.data),
    len([e for e in sports_dataset.target if e == 0]),
    len([e for e in sports_dataset.target if e == 1]),
)
print(
    f"Total examples: {len_all}, Baseball examples: {len_baseball}, Hockey examples: {len_hockey}"
)

# Data Preperation
labels = [
    sports_dataset.target_names[x].split(".")[-1] for x in sports_dataset["target"]
]
texts = [text.strip() for text in sports_dataset["data"]]
df = pd.DataFrame(zip(texts, labels), columns=["prompt", "completion"])  # [:300]

# Save the data
df.to_json("data/sport2.jsonl", orient="records", lines=True)
