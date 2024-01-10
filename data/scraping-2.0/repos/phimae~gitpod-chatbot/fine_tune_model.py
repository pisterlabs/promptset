#!/usr/bin/env python3

import openai
import os
openai.api_key = os.environ["OPENAI_API_KEY"]


from openai import Model



# Set the API key using the OPENAI_API_KEY environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the fine_tune_model function
def fine_tune_model(
    model_name: str,
    training_data_file_id: str,
    prompt_file_id: str,
    fine_tuned_model_name: str,
    max_steps: int = 1000,
    learning_rate: float = 1e-4,
    **kwargs
):

    """
    Fine-tune the OpenAI model with the specified parameters and return the resulting model ID.
    """
    model = None
    models = openai.Model.list()
    for m in models['data']:
        if m['id'] == model_name:
            model = m
            break

    prompt = f"train:file:{training_data_file_id}\n\neval:file:{prompt_file_id}\n"

    fine_tuned_model = model.finetune(
        prompt=prompt,
        max_steps=max_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        **kwargs
    )

    return fine_tuned_model.id



if __name__ == "__main__":
    # Set OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Set the model name and other hyperparameters
    MODEL_NAME = "text-davinci-002"
    MAX_STEPS = 1000
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 2
    FINE_TUNED_MODEL_NAME = "my-fine-tuned-model"

    # Set the file IDs for the training data and prompt
    TRAINING_DATA_FILE_ID = "file-FiHciDb2peCMmGSTniZEkhTG"
    PROMPT_FILE_ID = "file-sfTtou8akP4arq3HbsMc1EwA"

    # Fine-tune the model
    fine_tuned_model_id = fine_tune_model(
        model_name=MODEL_NAME,
        training_data_file_id=TRAINING_DATA_FILE_ID,
        prompt_file_id=PROMPT_FILE_ID,
        fine_tuned_model_name=FINE_TUNED_MODEL_NAME,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE
    )

    print(f"Fine-tuned model ID: {fine_tuned_model_id}")
