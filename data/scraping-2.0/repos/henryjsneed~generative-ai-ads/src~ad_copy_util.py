import openai
import torch
import json

with open('../api.key', 'r') as file:
    content = file.read()

openai.api_key = content

def load_ad_copies(filename):
    with open(filename, "r") as file:
        ad_copies = json.load(file)
    return ad_copies

def generate_text_embeddings(ad_texts, model, tokenizer, device, batch_size=32):
    model.eval()
    all_embeddings = []

    for batch_start in range(0, len(ad_texts), batch_size):
        batch_texts = ad_texts[batch_start:batch_start + batch_size]

        # Tokenize the batch
        inputs = tokenizer(
            batch_texts,
            padding='longest',
            truncation=True,
            return_tensors="pt"
        ).to(device) 

        with torch.no_grad():
            outputs = model(**inputs)
            # The last hidden-state is the first element of the output tuple
            last_hidden_states = outputs.last_hidden_state
            cls_embeddings = last_hidden_states[:, 0, :]
            all_embeddings.extend(cls_embeddings.cpu().numpy())

    return all_embeddings


def generate_ad_copy_options(prompt, max_items=5, max_tokens=300, temperature=1):
    """
    Send a prompt to OpenAI API and get a list of items based on the response.

    Args:
    prompt (str): The prompt to send to OpenAI API.
    max_items (int): The maximum number of items expected in the response.

    Returns:
    list: A list of items received in response to the prompt.
    """

    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=temperature
        )

        items = response.choices[0].text.strip().split('\n')

        return items[:max_items]

    except openai.error.OpenAIError as e:
        print(f"Received an error: {e}")
        return []
