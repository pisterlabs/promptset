import os
from openai import OpenAI
from config.openai_api_key import OPENAI_API_KEY
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
import pandas as pd

# Define the file path to the CSV data
file_path = os.path.join('data', 'human_labeled_semmed.csv')


def load_model(model_type, use_icl):
    """
    Load a language model for inference.

    Args:
        model_type (str): The type of the model to load ('llama', 'gpt_3_5_turbo', or 'gpt_4_0').
        use_icl (bool): Whether to use Integrated Conversation Learning (ICL) for OpenAI models.

    Returns:
        tuple or object: A tuple containing the client and model for OpenAI models, or just the model for Llama models.

    Raises:
        ValueError: If an unknown model type is provided.

    """
    if model_type == 'llama':
        # Load a Llama model
        model_name = "TheBloke/Llama-2-13B-chat-GGML"
        model_path = hf_hub_download(repo_id=model_name, filename="llama-2-13b-chat.ggmlv3.q5_1.bin")
        model = Llama(model_path=model_path, n_threads=2, n_batch=512, n_gpu_layers=32)
        if use_icl:
            return prepare_icl(model, model_type)
        return model
    else:
        # Load an OpenAI model
        client = OpenAI(api_key=OPENAI_API_KEY)
        if model_type == 'gpt_3_5_turbo':
            model = 'gpt-3.5-turbo'
        elif model_type == 'gpt_4_0':
            model = 'gpt-4'
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        if use_icl:
            return client, prepare_icl(model, model_type)
        return client, model


def prepare_icl(model, model_type):
    """
    Prepare context for models using Integrated Conversation Learning (ICL).

    Args:
        model (object): The language model (OpenAI model).
        model_type (str): The type of the model being used.

    Returns:
        tuple: A tuple containing the model and context for OpenAI models.

    """
    df = pd.read_csv(file_path)
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    test_df.to_csv(os.path.join('data', f'test_df_{model_type}_icl.csv'), index=False)
    context_entries = train_df.sample(n=10)

    context = context_entries.apply(
        lambda row: f"{row['Question']} Answer: {'Yes' if row['Label'] else 'No'}\n",
        axis=1).str.cat()
    return model, context
