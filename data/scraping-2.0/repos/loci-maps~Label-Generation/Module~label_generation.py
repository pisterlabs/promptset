import time
import openai
from tqdm import tqdm
import configparser
from scipy.cluster.hierarchy import linkage

def get_api_key(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config.get('open_ai', 'api_key')

def generate_reverse_embeddings(embeddings):
    reverse_embeddings = {}
    for i in range(len(embeddings.keys())):
        reverse_embeddings[tuple(list(embeddings.values())[i])] = list(embeddings.keys())[i]
    return reverse_embeddings

def summary_from_embedding(vector, reverse_embeddings, summaries):
    file = reverse_embeddings[tuple(vector)]
    return [summaries[file]]

def determine_label(topics, model_engine="gpt-3.5-turbo", prompt = "In a minimum of 1 word and a maximum of 3 words find the most specific commonality between the following topics, try to be as broad as possible:"):
    main_text = "{} {}".format(prompt, " ".join(topics))
    prompt = {"role":"system", "content": main_text}
    try:
        response = openai.ChatCompletion.create(model = model_engine, messages = [prompt])["choices"][0]["message"]["content"]
    except Exception as e:
        print("Exception: {}".format(e))
        print("Error with rate limit waiting 60 seconds")
        time.sleep(60)
        response = openai.ChatCompletion.create(model = model_engine, messages = [prompt])["choices"][0]["message"]["content"]
    return response

def generate_labels(embeddings, summaries):
    embedding_vals = list(embeddings.values())
    links = [list(x) for x in linkage(embedding_vals)]
    original_size = len(embedding_vals)
    reverse_embeddings = generate_reverse_embeddings(embeddings)

    for i in tqdm(range(len(links))):
        cluster = links[i]
        sub_cluster_a = cluster[0]
        sub_cluster_b = cluster[1]
        cluster.append(None)
        # Get Summary for leaf 
        if sub_cluster_a < original_size:
            summary1 = summary_from_embedding(embedding_vals[int(sub_cluster_a)], reverse_embeddings, summaries)
        # Generate the label of the child
        else:
            i = int(sub_cluster_a - original_size)
            summary1 = [links[i][4]]
        # Get label for child
        if sub_cluster_b < original_size:
            summary2 = summary_from_embedding(embedding_vals[int(sub_cluster_b)], reverse_embeddings, summaries)
        # Generate Label for child
        else:
            i = int(sub_cluster_b - original_size)
            summary2 = [links[i][4]]
        cluster.append(summary1 + summary2)
        cluster[4] = determine_label(cluster[5])
        links[i] = cluster
    return links