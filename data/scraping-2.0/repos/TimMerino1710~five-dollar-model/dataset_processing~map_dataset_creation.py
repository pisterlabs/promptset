import os
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
# from mysql import connector
import textwrap
import numpy as np
import cv2
import random
# from keras_nlp.models import BertBackbone, BertTokenizer, BertPreprocessor
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sentence_transformers import SentenceTransformer, util
from mysql import connector
import openai
import tiktoken
import time
import re
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# sentence embedding model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
model = AutoModel.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

openai.organization = ''
openai.api_key='' # enter open ai API key here :)

prompt = """ Take each string in the list provided, and write an alternate label for each one. These strings describe an image of a pixel video game map. 
            These alternate labels should describe the same image as the original label, but use different words and a different sentence structure. Use simple or common words when writing the alternate labels. Assume you have the vocabulary of a 10 year old. 
            Your output should have the same number of strings as the input list."""

messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt + " Write your output in the form of a list in this style: [a,b,c]. ONLY OUTPUT THE LIST. Here is the list of labels:"}
         ]

# Get prompt length
def num_tokens_from_messages(messages):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

# =================================================
#
#                   EMBEDDING DATA
#
# =================================================


#Mean Pooling - Take average of all tokens
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

#Encode text
def encode(texts, tokenizer, model, max_length=0):
    # Tokenize sentences
    if max_length > 0:
       encoded_input = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    else:
      encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)

    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return model_output, embeddings


def get_sent_word_embeddings(model, tokenizer, labels, max_len):
    model_out, mean_pool_embs = encode(labels, tokenizer, model, max_length=max_len)
    word_embeddings = model_out['last_hidden_state'].detach().cpu().numpy()
    sent_embeddings = mean_pool_embs.detach().cpu().numpy()

    return sent_embeddings, word_embeddings

# segment the data into sublists to not exceed api limits
def segment_labels(labels, threshold):
    encoding = tiktoken.get_encoding("cl100k_base")

    arr = []
    total = []

    for i, label in enumerate(labels):
        arr.append(label)
        if len(encoding.encode(f"{arr}")) > threshold:
            hold = arr.pop()
            if (size := len(encoding.encode(f"{[hold]}"))) > threshold:
                raise Exception(f"Element at position *{i}*, *'{[hold]}'* is too big: *{size} tokens* for this threshold: *{threshold} tokens*")
            total.append(arr)
            arr = [hold] 
    total.append(arr)
    return total


def do_gpt_call(labels):
    result = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant with excellent attention to detail. You only output python lists of strings according to the instructions you are given. Output the list on a single line, without any newlines. Make sure every list is closed properly"},
            {"role": "user", "content": prompt + " Here is the list of labels: " + f"{labels}"},

            ]
        )
    return result
    

def get_gpt_alt_labels(labels, num_tries=5):

    start = time.time()
    size = num_tokens_from_messages(messages)
    label_lists = segment_labels(labels, 1500 - size)
    print(f"split time = {time.time() - start}")
    alt_labels, ans_arr = [], []
    print("Number of loops: ", len(label_lists))

    for i, label_list in enumerate(label_lists):
        tries = 0
        success = False
        start = time.time()
        print(f"Loop {i} running through array of size {len(label_list)}")

        while tries < num_tries and not success:
            result = do_gpt_call(label_list)
            answer = result.choices[0]['message']['content']
            print(answer)
            apostrophe_pattern = r"(?<=\w)'(?=[^,\]])|'(?=\w+?'\s)"
            answer = re.sub(apostrophe_pattern, '', answer)
            
            alt_label_list = eval(answer[answer.find("["):answer.find(']')+1])
            if len(label_list) == len(alt_label_list):
                success = True
            else:
                print("FAILED. Trying again...")
                print("Try number: ", tries + 1)
                tries += 1


        if success:
            ans_arr.append(answer)
            alt_labels += alt_label_list
        else:
            print("failed completely.")
            return
        print(f"api call time = {time.time() - start}")

    return alt_labels, ans_arr

# Function to extract tiles from the 'pokemon.png' image file
def extract_tiles(image_file, tile_size=8, num_tiles=16):
    image = Image.open(image_file)
    tiles = []
    rows = cols = int(np.sqrt(num_tiles))

    for i in range(rows):
        for j in range(cols):
            tile = image.crop((j * tile_size, i * tile_size, (j + 1) * tile_size, (i + 1) * tile_size))
            tiles.append(tile)
    return tiles

TILES = extract_tiles('map_tileset/pokemon_tileset.png')

# Use tiles to construct image of map
def map_to_image(oh_map, tiles=TILES, tile_size=8):
    ascii_map =  np.argmax(oh_map, axis=-1)
    rows, cols = ascii_map.shape
    image = Image.new('RGB', (cols * tile_size, rows * tile_size))

    for i in range(rows):
        for j in range(cols):
            tile_index = ascii_map[i, j]
            tile = tiles[tile_index]
            image.paste(tile, (j * tile_size, i * tile_size))

    return image


# Use tiles to construct image of map
def mixup_map_to_image(mixup_oh_map, tiles=TILES, tile_size=8):
    rows, cols, _ = mixup_oh_map.shape
    image = Image.new('RGB', (cols * tile_size, rows * tile_size))

    for i in range(rows):
        for j in range(cols):
            # Find the indices of non-zero elements in the vector
            non_zero_indices = np.nonzero(mixup_oh_map[i, j])[0]

            # Initialize an empty transparent layer to merge the tiles
            merged_tile = Image.new('RGBA', (tile_size, tile_size), (0, 0, 0, 0))

            for tile_index in non_zero_indices:
                tile = tiles[tile_index].convert('RGBA')
                opacity = int(255 * mixup_oh_map[i, j, tile_index])

                # Set the opacity of the tile
                alpha = Image.new('L', tile.size, opacity)
                tile.putalpha(alpha)

                # Merge the tile with the existing merged_tile
                merged_tile = Image.alpha_composite(merged_tile, tile)

            # Convert the merged_tile back to RGB mode and paste it on the image
            merged_tile = merged_tile.convert('RGB')
            image.paste(merged_tile, (j * tile_size, i * tile_size))

    return image

def render_maps(maps, labels, embeddings, fig_label=''):
    images = [map_to_image(ascii_map) for ascii_map in maps]

    fig, axes = plt.subplots(2, len(images), figsize=(3 * len(images), 6))
    fig.suptitle(fig_label, fontsize=16)

    for i, ax in enumerate(axes[0]):
        ax.imshow(images[i])
        ax.set_title("\n".join(textwrap.wrap(labels[i], width=30)), fontsize=8)
        ax.axis('off')

    # Calculate dimensions for the square heatmap
    heatmap_size = int(np.ceil(np.sqrt(384)))

    for i, ax in enumerate(axes[1]):
        embedding = embeddings[i]
        # Pad the embedding with zeros to create a square shape
        padded_embedding = np.pad(embedding, (0, heatmap_size**2 - len(embedding)), mode='constant')
        # Reshape the padded embedding to create a heatmap
        heatmap = padded_embedding.reshape(heatmap_size, heatmap_size)
        im = ax.imshow(heatmap, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax.axis('off')

    plt.show()

def render_mixup_maps(maps, labels, embeddings, fig_label=''):
    images = [mixup_map_to_image(ascii_map) for ascii_map in maps]

    fig, axes = plt.subplots(2, len(images), figsize=(3 * len(images), 6))
    fig.suptitle(fig_label, fontsize=16)

    for i, ax in enumerate(axes[0]):
        ax.imshow(images[i])
        ax.set_title("\n".join(textwrap.wrap(labels[i], width=30)), fontsize=8)
        ax.axis('off')

    # Calculate dimensions for the square heatmap
    heatmap_size = int(np.ceil(np.sqrt(384)))

    for i, ax in enumerate(axes[1]):
        embedding = embeddings[i]
        # Pad the embedding with zeros to create a square shape
        padded_embedding = np.pad(embedding, (0, heatmap_size**2 - len(embedding)), mode='constant')
        # Reshape the padded embedding to create a heatmap
        heatmap = padded_embedding.reshape(heatmap_size, heatmap_size)
        im = ax.imshow(heatmap, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax.axis('off')

    plt.show()


# convert ascii string with newline row delimeters to a 2d array of integers
def ascii_string_to_int_array(ascii_string):
    # raw database string
    ascii_string.replace('\r', '')
    rows = ascii_string.split("\n")
    level = []
    expected_size = 10
    for r in rows:
        int_row = [int(x, 16) for x in r]
        if int_row != []:
            level.append(int_row)
    if len(level) != expected_size:
        print("size mismatch when creating int map representation. ")
        print("expected ", expected_size, " got ", len(level))
        print(ascii_string)
        print(level)
    return np.array(level)

# get data from database
def get_db_data(cursor):
    ann_ids, maps, annotations, authors = [], [], [], []
    query = "SELECT ANNOTATION_ID, ASCII_MAP, ANNOTATION, ANN_AUTHOR FROM user_levels_annotated"

    cursor.execute(query)
    results = cursor.fetchall()

    for result in results:
        ann_ids.append(result[0])
        maps.append(ascii_string_to_int_array(result[1]))
        annotations.append(result[2])
        authors.append(result[3])
    
    # one hot encode
    maps = np.eye(16)[maps]
    return ann_ids, maps, annotations, authors

# sort by ann_id
def sort_by_annid(ann_ids, maps, annotations):
    # Create a list of tuples containing elements from all four lists at the same index
    combined = list(zip(ann_ids, maps, annotations))

    # Sort the combined list based on ann_ids (the first element in each tuple)
    sorted_combined = sorted(combined, key=lambda x: x[0])

    # Separate the sorted tuples back into individual lists
    sorted_ann_ids = [item[0] for item in sorted_combined]
    sorted_maps = [item[1] for item in sorted_combined]
    sorted_annotations = [item[2] for item in sorted_combined]

    return sorted_ann_ids, sorted_maps, sorted_annotations

# get embedding for level annotations
def embed_labels(labels, embedding_model):
    labels = np.array(labels)
    embeddings = embedding_model.encode(labels)
    return embeddings


# creates num_augmentations - 1 copies of each map, where the copies have gaussian noise applied (multiplicatively) to the embedding
def noise_augmentation(ann_ids, maps, labels, embeddings, num_augmentations, noise_std_dev=0.01):
    augmented_annids, augmented_maps, augmented_labels, augmented_embeddings = [], [], [], []
    # Add Gaussian noise to the embeddings
    for ann_id, level, label, embedding in zip(ann_ids, maps, labels, embeddings):
        for _ in range(num_augmentations):
            # mean of 1, because this is multiplicative, since we want values that are 0 (or close) to stay 0
            noise = np.random.normal(1, noise_std_dev, embedding.shape)
            augmented_embedding = embedding * noise

            augmented_embeddings.append(augmented_embedding)
            augmented_maps.append(level)
            augmented_labels.append(label)
            augmented_annids.append(ann_id)
    
    # only return augmented values
    return augmented_annids, augmented_maps, augmented_labels, augmented_embeddings

def gpt_augmentation(ann_ids, maps, labels, embeddings, authors, embedding_model):
    augmented_annids, augmented_maps, augmented_labels, augmented_embeddings, augmented_authors = [], [], [], [], []
    alt_labels, _ = get_gpt_alt_labels(labels)
    print(len(alt_labels))
    print(len(labels))

    assert len(alt_labels) == len(labels)

    alt_label_embeddings = embed_labels(alt_labels, embedding_model)

    for i, (alt_label, alt_label_embedding) in enumerate(zip(alt_labels, alt_label_embeddings)):
        # append original ann_id twice
        augmented_annids.append(ann_ids[i])
        augmented_annids.append(ann_ids[i])

        augmented_authors.append(authors[i])
        augmented_authors.append(authors[i])

        # append original map twice
        augmented_maps.append(maps[i])
        augmented_maps.append(maps[i])

        # append the original label and the alt label
        augmented_labels.append(labels[i])
        augmented_labels.append(alt_label)

        # append the original embedding and the alt labels embedding
        augmented_embeddings.append(embeddings[i])
        augmented_embeddings.append(alt_label_embedding)



    return augmented_annids, augmented_maps, augmented_labels, augmented_embeddings, augmented_authors

def export_data(ann_ids, maps, labels, embeddings, filename):
    data = {
        'annotation_ids': ann_ids,
        'images': maps,
        'labels': labels,
        'embeddings': embeddings
    }
    np.save(filename, data, allow_pickle=True)

def export_data_with_z(ann_ids, maps, labels, embeddings, z_vectors, filename):
    data = {
        'annotation_ids': ann_ids,
        'images': maps,
        'labels': labels,
        'embeddings': embeddings,
        'z_vectors': z_vectors
    }
    np.save(filename, data, allow_pickle=True)

def filter_out_authors(ann_ids, maps, annotations, authors, banned_authors):
    filtered_ann_ids, filtered_maps, filtered_annotations, filtered_authors = [],[],[],[]
    for (ann_id, level, anno, auth) in zip(ann_ids, maps, annotations, authors):
        if auth not in banned_authors:
            filtered_ann_ids.append(ann_id)
            filtered_maps.append(level)
            filtered_annotations.append(anno)
            filtered_authors.append(auth)

    return filtered_ann_ids, filtered_maps, filtered_annotations, filtered_authors



# do mixup, interplating by lambda between two points
def mixup_aug(ann_ids, maps, annotations, embeddings, n_mixups=1, lamb=-0.5):
    augmented_annids, augmented_maps, augmented_labels, augmented_embeddings = [],[],[],[]
    
    for i, (ann_id, level, label, embedding) in enumerate(zip(ann_ids, maps, annotations, embeddings)):
        # Randomly select n_mixups indices without replacement
        mixup_indices = np.random.choice(len(ann_ids), n_mixups, replace=False)

        for idx in mixup_indices:
            # Get the corresponding levels, labels, and embeddings
            mix_level = maps[idx]
            mix_label = annotations[idx]
            mix_embedding = embeddings[idx]
            mix_annid = ann_ids[idx]
            
            # Interpolate the levels, labels, and embeddings
            # for mixup_level, mixup_label, mixup_embedding in zip(mixup_levels, mixup_labels, mixup_embeddings):
            new_level = lamb * level + (1 - lamb) * mix_level
            new_label = label + " + " + mix_label
            new_annid = str(ann_id) + " + " + str(mix_annid)
            new_embedding = lamb * embedding + (1 - lamb) * mix_embedding
            
            # Append the new data to the augmented lists
            augmented_annids.append(new_annid)
            augmented_maps.append(new_level)
            augmented_labels.append(new_label)
            augmented_embeddings.append(new_embedding)

    # only return the augmented values
    return augmented_annids, augmented_maps, augmented_labels, augmented_embeddings

# interpolate n times between a label and its altlabel (MUST BE CALLED RIGHT AFTER GPT AUG)
def altlabel_interp_aug(ann_ids, maps, annotations, embeddings, n_steps=1):
    augmented_annids, augmented_maps, augmented_labels, augmented_embeddings = [],[],[],[]
    
    for i in range(0, len(maps), 2):
        orig_idx = i
        alt_idx = i + 1

        alpha_values = np.linspace(0, 1, n_steps + 2)[1:-1]  # Exclude the 0 and 1 values
        interpolated_embeddings = []
        for alpha in alpha_values:
            # ann_id and map will be the same regardless
            augmented_annids.append(ann_ids[orig_idx])
            augmented_maps.append(maps[orig_idx])

            # randomly choose which label to append, the original or the alt
            label_idxs = [orig_idx, alt_idx]
            rand_label_idx = random.choice(label_idxs)
            augmented_labels.append(annotations[rand_label_idx])

            interpolated_embedding = embeddings[orig_idx] * (1 - alpha) + embeddings[alt_idx] * alpha
            interpolated_embeddings.append(interpolated_embedding)
        # add the interp embeddings to the embedding list
        augmented_embeddings = augmented_embeddings + interpolated_embeddings


    # only return the augmented samples
    return augmented_annids, augmented_maps, augmented_labels, augmented_embeddings


#Mean Pooling - Take average of all tokens
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def add_and_export_aug(orig_annids, orig_maps, orig_anns, orig_embs, aug_annids, aug_maps, aug_anns, aug_embs, fname, z_vectors=None):
    print("exporting ", fname)
    print("number of augmented samples: ", len(aug_embs))
    
    ann_ids_exp = np.array(orig_annids + aug_annids)
    maps_exp = np.array(orig_maps + aug_maps)
    annotations_exp = np.array(orig_anns + aug_anns)
    embeddings_exp = np.array(orig_embs + aug_embs)
    
    print("size of total augmented dataset: ", len(embeddings_exp))
    assert len(maps_exp) == len(embeddings_exp) and len(maps_exp) == len(annotations_exp)

    if z_vectors is None:
        export_data(ann_ids_exp, maps_exp, annotations_exp, embeddings_exp, fname)
    else:
        export_data_with_z(ann_ids_exp, maps_exp, annotations_exp, embeddings_exp, z_vectors, fname)


#Encode text
def encode(texts, tokenizer, model, max_length=0):
    # Tokenize sentences
    if max_length > 0:
       encoded_input = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    else:
      encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)

    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return model_output, embeddings


LOAD = True


if LOAD == False:
    # load data, filter it, and generate the embeddings
    ann_ids, maps, annotations, authors = get_db_data(cursor=mycursor)
    print("Number of levels in db: ", len(ann_ids))

    ann_ids, maps, annotations = sort_by_annid(ann_ids, maps, annotations)
    sent_embeddings, word_embeddings = get_sent_word_embeddings(model, tokenizer, annotations, max_len=25)
    data = {
            'ann_ids' : ann_ids,
            'images': maps,
            'labels': annotations,
            'embeddings': sent_embeddings,
            'word_embeddings': word_embeddings,
        }
    np.save('datasets/Map Data/maps_noaug.npy', data, allow_pickle=True)


    # do gpt alt label augmentation
    ann_ids, maps, annotations, embeddings, authors = gpt_augmentation(ann_ids, maps, annotations, sent_embeddings, authors, model)
    print("Number of levels before filtering: ", len(ann_ids))

    ann_ids_exp = np.array(ann_ids)
    maps_exp = np.array(maps)
    annotations_exp = np.array(annotations)
    embeddings_exp = np.array(embeddings)


    export_data(ann_ids_exp, maps_exp, annotations_exp, embeddings_exp, 'datasets/maps_gpt4_aug.npy')

# Do this if you don't want to rerun all the GPT augmentation, which takes time and money
else:
    dataset_file = 'maps_gpt4_aug_genexp.npy'
    data = np.load(dataset_file, allow_pickle=True).item()
    ann_ids = list(data['annotation_ids'])
    maps = list(data['images'])
    labels = list(data['labels'])
    embeddings = list(data['embeddings'])

    sent_embeddings, word_embeddings = get_sent_word_embeddings(model, tokenizer, labels, max_len=35)
    data = {
            'ann_ids' : ann_ids,
            'images': maps,
            'labels': labels,
            'embeddings': sent_embeddings,
            'word_embeddings': word_embeddings,
        }
    np.save('datasets/maps_gpt4_aug_wordemb_maxlen35.npy', data, allow_pickle=True)
    







