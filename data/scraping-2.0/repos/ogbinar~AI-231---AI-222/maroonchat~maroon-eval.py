import pandas as pd
import time
import re
import os
import nltk
from nltk.util import ngrams
from nltk.metrics import jaccard_distance
from llama_cpp import Llama
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.llms import HuggingFaceHub
from langchain import HuggingFacePipeline
from langchain.chains import LLMChain
import numpy as np
from sentence_transformers import SentenceTransformer
import jellyfish

embedding_model_name = "BAAI/bge-small-en-v1.5"
embedding_model = SentenceTransformer(embedding_model_name)
# Set HuggingFaceHub API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


def calculate_similarity(reference, candidate):
    """
    Calculates text similarity between a reference and a candidate using both sentence embeddings and n-gram matching.

    Args:
        reference (str): The reference text string.
        candidate (str): The candidate text string.

    Returns:
        tuple: A tuple containing the exact match score and the n-gram matching score.
    """

    # Get embeddings for both texts
    embedding1 = embedding_model.encode(str(reference), convert_to_tensor=True)
    embedding2 = embedding_model.encode(str(candidate), convert_to_tensor=True)
    # Move tensors to CPU
    embedding1 = embedding1.cpu()
    embedding2 = embedding2.cpu()
    # Calculate cosine similarity using sentence embeddings
    # Embeddings: Understands meaning, good for similar ideas with different words.
    similarity_embeddings = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    # Tokenize the strings into words
    # Normalized Levenshtein Score (1-0)
    # Levenshtein: Counts edits, good for word order and typos, strict on word choice.
    levenshtein_score = 1 - (jellyfish.levenshtein_distance(str(reference), str(candidate)) / max(len(str(reference)), len(str(candidate))))

    # Normalized Jaro-Winkler Score (1-0)
    # Jaro-Winkler: Forgives some errors, good for names or addresses, balances exactness and flexibility.
    jaro_winkler_score = jellyfish.jaro_winkler_similarity(str(reference), str(candidate))

    return similarity_embeddings, levenshtein_score, jaro_winkler_score


# Read the Excel file
df = pd.read_excel("MAROON CHAT Q&A.xlsx", sheet_name="Public")

# Define model path
#model_path = "combined_merged_model/ggml-model-Q4_0.gguf"
#model_path = "combined_merged_model/ggml-model-q4_0.gguf"
# Initialize LlamaCpp
#llm = LlamaCpp(
#    model_path=model_path,
#    temperature=0.1,
#    max_tokens=256,
#    n_gpu_layers=10000,
#    top_p=1,
#    n_ctx=4096,
#    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#    verbose=True,
#)
 
import torch
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
device_map = {"":0}

# indicate where your model and tokenizers are 
model_dir = "/home/michel/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/b70aa86578567ba3301b21c8a27bea4e8f6d6d61"
tokenizer_dir = "combined_merged_model"
model_4bit = AutoModelForCausalLM.from_pretrained( model_dir, device_map=device_map,quantization_config=quantization_config, )
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
pipeline = pipeline(
        "text-generation",
        model=model_4bit,
        tokenizer=tokenizer,
        use_cache=True,
        device_map=device_map,
        max_length=256,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
) 
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=pipeline)


# Define HuggingFaceHub repo ID
repo_id = "HuggingFaceH4/zephyr-7b-beta"

# Initialize HuggingFaceHub
llm_judge = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature": 0.1, "max_length": 256}
)

# Define scoring prompt
scoring_prompt = PromptTemplate.from_template(
    """<|system|> Provide a single floating-point value between 0.0 and 1.0, representing the accuracy and completeness of the chatbot answer compared to the groundtruth. No explanations or additional text are necessary.
    <|user|>
Chatbot Answer:
{chatbot_answer}

Groundtruth Answer:
{groundtruth_answer}

Request:
Rate the chatbot answer between 0.0 and 1.0
<|assistant|>"""
)

# Define system message
system_message = """Your role as a chatbot assistant is to answer questions related to the University of the Philippines. \
    Politely decline if asked questions without relevant context."""

# Define question prompt
question_prompt = PromptTemplate.from_template(
    """[INST]
    {system_prompt}
Question:
{question}
[/INST]"""
)

# Define judge runnable
judge_runnable = scoring_prompt | llm_judge | StrOutputParser()

# Define chatbot runnable
cb_runnable = question_prompt | llm | StrOutputParser()

# Measure program duration
start_time = time.time()

# Process each row in the dataframe
answer_pairs = []

for index, row in df.iterrows():
    question = row['question ']
    gt = row['answer']

    # Invoke chatbot runnable
    cb_answer = cb_runnable.invoke({"system_prompt": system_message, "question": question})

    # Invoke judge runnable
    rating = judge_runnable.invoke({"chatbot_answer": cb_answer, "groundtruth_answer": gt})

    match = re.search(r"\d+\.\d+", rating)

    float_value = 0.0
    if match:
        float_value = float(match.group())
        print("Float value:", float_value)  # Output: Float value: 1.0
    else:
        print("No float value found in the text.")

    similarity_embeddings, levenshtein_score, jaro_winkler_score = calculate_similarity(gt, cb_answer)

    """
    Embeddings: Understands meaning, good for similar ideas with different words.
    Levenshtein: Counts edits, good for word order and typos, strict on word choice.
    Jaro-Winkler: Forgives some errors, good for names or addresses, balances exactness and flexibility.
    """

    # Append answer pair to the list
    answer_pairs.append({
        "question": question,
        "groundtruth": gt,
        "chatbot": cb_answer,
        "zephyr_score": float_value,
        "levenshtein_score": levenshtein_score,
        "jaro-wingkler_score": jaro_winkler_score,
        "embeddings_score": similarity_embeddings
    })

# Create dataframe from answer pairs
df_answer_pairs = pd.DataFrame(answer_pairs)

# Write dataframe to Excel file
df_answer_pairs.to_excel("mistral_rating.xlsx", index=False)

# Calculate program duration
duration = time.time() - start_time
print("Program duration:", duration, "seconds")
