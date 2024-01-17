import json
import os
from argparse import ArgumentParser
from datasets import Dataset

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--run_id", type=int, required=True)
    args = argparser.parse_args()

    data = {
        "date_collected": [],
        "repo_name": [],
        "file_name": [],
        "file_contents": [],
        "prompts": [],
    }

    with open(f"data/grouped-data-{args.run_id:03d}.json") as f:
        raw_data = json.load(f)

        absent = 0
        total = 0
        num_prompts = 0

        for key, prompts in raw_data.items():
            repo_name = "/".join(key.split("/")[-2].split("~"))
            file_name = key.split("/")[-1]
            file_path = key.replace("/scraping/", "/scraping-2.0/")

            if not os.path.exists(file_path):
                absent += 1
                continue

            total += 1
            num_prompts += len(prompts)
            with open(file_path, "r", encoding="utf8") as file:
                data["date_collected"].append("2024-01-10")
                data["repo_name"].append(repo_name)
                data["file_name"].append(file_name)
                data["file_contents"].append(file.read())
                data["prompts"].append(prompts)

        print(f"Absent: {absent}")
        print(f"Total: {total}")
        print(f"Prompts: {num_prompts}")

    ds = Dataset.from_dict(data)
    ds.push_to_hub("pisterlabs/promptset", create_pr=True)
