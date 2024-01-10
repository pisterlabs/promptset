import io
import os

import openai
from datasets import load_dataset


def load_and_prepare_data():
    """Load and preprocess the training data."""
    dataset = load_dataset("sublime-security/babbelphish")
    df = dataset.data["train"].to_pandas()
    df["prompt"] = df["prompt"].apply(lambda x: f"{x} ->")
    df["completion"] = df["completion"].apply(lambda x: f"{x} END")
    return df[["prompt", "completion"]]


def dataframe_to_jsonl(df):
    """Convert a DataFrame to JSONL format."""
    jsonl_string = df.to_json(orient="records", lines=True)
    return io.BytesIO(jsonl_string.encode())


def upload_to_openai(jsonl_io):
    """Upload the JSONL data to OpenAI."""
    return openai.File.create(file=jsonl_io, purpose="fine-tune").id


def fine_tune_model(file_id):
    """Fine-tune the specified model using the uploaded file."""
    return openai.FineTune.create(training_file=file_id, model="curie")


def list_fine_tune_events(fine_tune_response):
    """List fine-tuning events for the specified fine-tuning process."""
    return openai.FineTune.list_events(id=fine_tune_response.id)


if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    df_selected = load_and_prepare_data()
    jsonl_bytes_io = dataframe_to_jsonl(df_selected)
    file_id = upload_to_openai(jsonl_bytes_io)
    fine_tune_response = fine_tune_model(file_id)
    fine_tune_events = list_fine_tune_events(fine_tune_response)
