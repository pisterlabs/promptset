import os
from dotenv import load_dotenv
import openai

load_dotenv()

openai.organization = "org-D2FBgBhwLFkKAOsgtSp86b4i"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Creates a fine-tuned model for search

remote_files = openai.File.list()["data"]
fine_tuning_files = filter(lambda f: "fine_tune.jsonl" in f["filename"], remote_files)
validation_files = filter(lambda f: "fine_tune_validation.jsonl" in f["filename"], remote_files)

latest_fine_tuning_file = max(fine_tuning_files, key=lambda x: x["created_at"])
latest_validation_file = max(validation_files, key=lambda x: x["created_at"])

openai.FineTune.create(
    training_file=latest_fine_tuning_file["id"],
    validation_file=latest_validation_file["id"],
    model="ada",
    n_epochs=4,
    batch_size=4,
    learning_rate_multiplier=0.1,
    prompt_loss_weight=0.1
)
