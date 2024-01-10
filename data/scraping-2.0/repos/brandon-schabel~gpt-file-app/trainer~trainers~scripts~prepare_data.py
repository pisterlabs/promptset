from ..scripts.setup import OPEN_AI_KEY
import openai
import json
import numpy as np
import re
from collections import defaultdict
from loguru import logger
from tiktoken import get_encoding

openai.api_key = OPEN_AI_KEY

# Initialize the tokenizer
encoding = get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count the number of tokens in a given text using tiktoken."""
    encoding = get_encoding("cl100k_base")
    return len(encoding.encode(text))


instruction = "Reduce the text while ensuring you keep the important details. If there are code examples, add more variations. Ignore performance metrics and useless/unneeded information."
instruction_tokens = count_tokens(instruction)


def clean_markdown(content: str) -> str:
    """Clean the markdown content."""
    markdown_patterns = [
        (r'#+ ', ''),
        (r'\[(.*?)\]\(.*?\)', r'\1'),
        (r'!\[.*?\]\(.*?\)', ''),
        (r'\*+', ''),
        (r'`.*?`', ''),
        (r'```.*?```', '', re.DOTALL),
        (r'^>.*$', '', re.MULTILINE),
        (r'^[-*] ', '', re.MULTILINE),
        (r'^\d+\. ', '', re.MULTILINE),
        (r'\n+', '\n')
    ]
    for pattern, replacement, *args in markdown_patterns:
        content = re.sub(pattern, replacement, content, *args)
    return content.strip()


def split_content_by_sentence(content: str, max_length: int = 2500 - instruction_tokens) -> list:
    """Splits the content into chunks based on token count but prioritizes sentence boundaries."""
    # Split by sentence
    sentences = re.split(r'(?<=[.!?]) +', content)

    chunks, current_chunk, current_length = [], "", 0

    for sentence in sentences:
        if current_length + count_tokens(sentence) > max_length:
            chunks.append(current_chunk.strip())
            current_chunk, current_length = "", 0
            
        current_chunk += sentence + ' '
        current_length += count_tokens(sentence)

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def validate_and_clean_data(data):
    """Validate and clean data."""
    for entry in data:
        messages = entry.get("messages", [])
        user_msg = next(
            (msg for msg in messages if msg["role"] == "user"), None)
        assistant_msg = next(
            (msg for msg in messages if msg["role"] == "assistant"), None)

        if not user_msg or not assistant_msg:
            raise ValueError(
                "Each entry should have a user and an assistant message.")
        user_msg['content'] = clean_markdown(user_msg['content'])
        assistant_msg['content'] = clean_markdown(assistant_msg['content'])
    return data


def summarize_completion(prompt, max_tokens=3000):
    """
    Obtain a summary for the given prompt.
    """
    total_summary = []

    chunks = split_content_by_sentence(
        prompt, max_length=max_tokens - instruction_tokens)

    for chunk in chunks:
        # Calculate the total tokens for chunk and instruction
        total_tokens = count_tokens(chunk) + instruction_tokens

        # Ensure that the chunk and instruction combined don't exceed max_tokens
        if total_tokens > max_tokens:
            logger.warning(
                f"Chunk too long: {total_tokens} tokens")
            continue

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": instruction
                },
                {
                    "role": "user",
                    "content": chunk
                }
            ],
            max_tokens=max_tokens - total_tokens  # Adjusting for the instruction tokens
        )

        assistant_message = response['choices'][0]['message']['content']

        logger.info(f"Assistant message: {assistant_message}")

        total_summary.append(assistant_message)

    return "\n".join(total_summary)


def prepare_data(input_path: str, output_path: str, summarize: bool = False, save_interval: int = 1):
    # Step 1: Load the data from the input_path
    # Step 1: Load the data from the input_path
    with open(input_path, "r") as f:
        dataset = [json.loads(line) for line in f]

    # Step 2: Clean the Markdown content
    dataset = validate_and_clean_data(dataset)

    # Step 3: Process and save data in intervals
    with open(output_path, "w") as f:
        for index, entry in enumerate(dataset):
            if summarize:
                for message in entry["messages"]:
                    message["content"] = summarize_completion(
                        message["content"])
            f.write(json.dumps(entry) + "\n")

            # Log progress
            if (index + 1) % save_interval == 0:
                logger.info(f"Processed and saved {index + 1} entries.")

    # Step 4: Load the cleaned data for validation
    with open(output_path) as f:
        dataset = [json.loads(line) for line in f]

    if summarize:
        for entry in dataset:
            for message in entry["messages"]:
                message["content"] = summarize_completion(message["content"])

    # Logging dataset stats
    logger.info(f"Num examples: {len(dataset)}")
    logger.info("First example:")
    for message in dataset[0]["messages"]:
        logger.info(message)

    # Check the formatting
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(k not in ("role", "content", "name") for k in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in ("system", "user", "assistant"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            if not content or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        logger.warning("Found errors:")
        for k, v in format_errors.items():
            logger.warning(f"{k}: {v}")
    else:
        logger.info("No errors found")

    # Token counting
    def num_tokens_from_messages(messages: list[str], tokens_per_message=3, tokens_per_name=1):
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def num_assistant_tokens_from_messages(messages):
        num_tokens = 0
        for message in messages:
            if message["role"] == "assistant":
                num_tokens += len(encoding.encode(message["content"]))
        return num_tokens

    # Warnings and token counts
    n_missing_system, n_missing_user = 0, 0
    n_messages, convo_lens, assistant_message_lens = [], [], []

    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(
            num_assistant_tokens_from_messages(messages))

    logger.info(f"Num examples missing system message: {n_missing_system}")
    logger.info(f"Num examples missing user message: {n_missing_user}")

    n_too_long = sum(l > 4096 for l in convo_lens)
    logger.warning(
        f"{n_too_long} examples may be over the 4096 token limit, they will be truncated during fine-tuning")

    # Pricing and default n_epochs estimate
    MAX_TOKENS_PER_EXAMPLE = 4096

    MIN_TARGET_EXAMPLES = 100
    MAX_TARGET_EXAMPLES = 25000
    TARGET_EPOCHS = 3
    MIN_EPOCHS = 1
    MAX_EPOCHS = 25

    n_epochs = TARGET_EPOCHS
    n_train_examples = len(dataset)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    n_billing_tokens_in_dataset = sum(
        min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
    logger.info(
        f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
    logger.info(
        f"By default, you'll train for {n_epochs} epochs on this dataset")
    logger.info(
        f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")
