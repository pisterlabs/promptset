import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, OrdinalEncoder
import torch
from pathlib import Path
import csv
from sklearn import preprocessing
from robustness.tools.breeds_helpers import ClassHierarchy, BreedsDatasetGenerator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sklearn.cluster import KMeans, FeatureAgglomeration
import torch.nn.functional as F
from wilds import get_dataset
import os
from dataclasses import dataclass
from typing import Any, Optional, Callable
import glob
import PIL
from PIL import Image
import hashlib
import torchvision.transforms as transforms
from wilds.datasets.wilds_dataset import WILDSSubset
from transformers import AutoTokenizer, LlamaForCausalLM
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
import vec2text
from skimage.segmentation import slic
from skimage.measure import label, regionprops
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from src.optimal_transport import group_mean_shift_transport, group_kmeans_shift_transport, transform_samples_kmeans, transform_samples, group_feature_transport, group_feature_transport2, group_kmeans_shift_transport2
from src.distance import group_percent_explained, W2_dist
from src.training import regular_training, dro_training, dro_training2
from src.logistic_regression import PTLogisticRegression, PTNN, FFNetwork, PTNNSimple
from src.training import regular_training
from src.cf_transport import get_dice_transformed, get_closest_target
from src.counterfactual_img import counterfactual_img2img, resnet50_embed, resnet50_classify, modify_text, img2text, ddim_cf_generate
from src.featurization.vector_lda.vector_lda_main import obtain_lda_feat
from src.featurization.vector_lda.embedded_topic_model.models.etm import ETM

# import matplotlib.pyplot as plt
from sklearn import preprocessing
# import tikzplotlib

IMAGENET_DIR = "../data/imagenet/"

def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, list):
        return [_to_device(xi, device) for xi in x]

def _batch_inference(model, dataset, batch_size=128, resize=None, processor=None, device='cuda') -> torch.Tensor:
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    results = []
    with torch.no_grad():
        # for batch in tqdm(loader):
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i+batch_size]
            if processor is not None:
                batch = processor(batch)
            x = _to_device(batch, device)

            if resize:
                x = torch.nn.functional.interpolate(x, size=resize, mode='bilinear', align_corners=False)

            results.append(model(x).cpu())

    results = torch.cat(results)
    return results

class CfGenerator:
    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names

    def generate(self, raw_source: np.ndarray, source: np.ndarray, diff: np.ndarray):
        raise NotImplementedError()

class CfIdentityGenerator(CfGenerator):
    def generate(self, raw_source: np.ndarray, source: np.ndarray, diff: np.ndarray):
        return source + diff

class CfPixelGenerator(CfGenerator):
    def generate(self, raw_source: np.ndarray, source: np.ndarray, diff: np.ndarray):
        return np.clip(source + diff, 0, 255)


class CfNLPTopicGenerator(CfGenerator):
    def __init__(self, feature_names: list[str], topic_model:ETM):
        super().__init__(feature_names)
        self.topic_model = topic_model
    
    def generate(self, raw_source, source: np.ndarray, diff: np.ndarray):
        transformed_source = source + diff
        transformed_source[transformed_source < 0] =1e-8
        return self.topic_model.generate_counterfactual(torch.from_numpy(transformed_source))

class CfNLPTopicGenerator_wordcount(CfGenerator):
    def __init__(self, feature_names: list[str], ngram_vectorizer):
        super().__init__(feature_names)
        self.ngram_vectorizer = ngram_vectorizer
    
    def generate(self, source: np.ndarray, diff: np.ndarray):
        transformed_source = source + diff
        generated_sentences = []
        transformed_source_str_ls = self.ngram_vectorizer.inverse_transform(transformed_source)
        for transformed_src_str in transformed_source_str_ls:
            trans_src_full_str = ""
            for k in range(len(transformed_src_str)):
                if k >= 1:
                    trans_src_full_str += " "
                trans_src_full_str += transformed_src_str[k]
            generated_sentences.append(trans_src_full_str)
        # for k in range(len(transformed_source)):
        #     curr_sentence = ""
        #     for j in range(len(transformed_source[k])):
        #         if j >= 1:
        #             curr_sentence += " "    
        #         curr_sentence += self.ngram_vectorizer.inverse_transform(transformed_source[k][j])

        #     generated_sentences.append(curr_sentence)
        
        return np.array(generated_sentences)
    
class CfDiffusionGenerator(CfGenerator):
    def __init__(self, orig_prompts: list[str], feature_names: list[str], finetuned=False):
        self.orig_prompts = orig_prompts
        self.feature_names = feature_names
        self.finetuned = finetuned

    def generate(self, raw_source: np.ndarray, source: np.ndarray, diff: np.ndarray):
        diff_hash = hashlib.sha1(diff).hexdigest()
        if os.path.exists(f"../data/base/cf/cf_imgs_{self.finetuned}_{diff_hash}.pkl"):
            print("Found cached cf images")
            cf_imgs = pickle.load(open(f"../data/base/cf/cf_imgs_{self.finetuned}_{diff_hash}.pkl", "rb"))
        else:
            cf_imgs = counterfactual_img2img(self.orig_prompts, self.feature_names, diff, finetuned=self.finetuned)[0]
            if not os.path.exists("../data/base/cf"):
                os.makedirs("../data/base/cf")
            pickle.dump(cf_imgs, open(f"../data/base/cf/cf_imgs_{self.finetuned}_{diff_hash}.pkl", "wb"))
        return cf_imgs


class CfInvertedDiffusionGenerator(CfGenerator):
    def __init__(self, orig_prompts: list[str], feature_names: list[str]):
        self.orig_prompts = orig_prompts
        self.feature_names = feature_names

    def generate(self, raw_source: np.ndarray, source: np.ndarray, diff: np.ndarray):
        diff_hash = hashlib.sha1(diff).hexdigest()
        if os.path.exists(f"../data/base/cf/cf_imgs_ddim_{diff_hash}.pkl"):
            print("Found cached cf images")
            cf_imgs = pickle.load(open(f"../data/base/cf/cf_imgs_ddim_{diff_hash}.pkl", "rb"))
            cf_prompts = []
            for i, prompt in enumerate(self.orig_prompts):
                cf_prompts.append(modify_text(prompt, self.feature_names, diff[i, :]))
            pickle.dump(cf_prompts, open(f"../data/base/cf/cf_prompts_ddim_{diff_hash}.pkl", "wb"))
            print("Output cf prompts")
        else:
            cf_imgs = ddim_cf_generate(raw_source, self.orig_prompts, self.feature_names, diff)[0]
            if not os.path.exists("../data/base/cf"):
                os.makedirs("../data/base/cf")
            pickle.dump(cf_imgs, open(f"../data/base/cf/cf_imgs_ddim_{diff_hash}.pkl", "wb"))
        return cf_imgs


class CfTextGenerator(CfGenerator):
    def __init__(self, orig_prompts: list[str], feature_names: list[str]):
        self.orig_prompts = orig_prompts
        self.feature_names = feature_names

    def generate(self, raw_source, source: np.ndarray, diff: np.ndarray):
        return np.array([modify_text(self.orig_prompts[i], self.feature_names, diff[i]) for i in range(len(self.orig_prompts))])


@dataclass
class ShiftDataset:
    raw_source: np.ndarray
    raw_target: np.ndarray
    source: np.ndarray
    target: np.ndarray
    source_groups: np.ndarray
    target_groups: np.ndarray
    feature_names: list[str]
    feature_types: list[str]
    scaler: Any
    group_fn: Callable[[np.ndarray, np.ndarray], int]
    cf_generator: CfGenerator
    embed_fn: Callable[[np.ndarray], np.ndarray]

    def subset(self, indices):
        # check if cf_generator has orig_prompts
        if hasattr(self.cf_generator, 'orig_prompts'):
            print("Subsetting cf_generator prompts")
            self.cf_generator.orig_prompts = [self.cf_generator.orig_prompts[i] for i in indices]

        return ShiftDataset(
            self.raw_source[indices] if self.raw_source is not None else None,
            self.raw_target[indices] if self.raw_target is not None else None,
            self.source[indices],
            self.target[indices],
            self.source_groups[indices],
            self.target_groups[indices],
            self.feature_names,
            self.feature_types,
            self.scaler,
            self.group_fn,
            self.cf_generator,
            self.embed_fn,
        )

    def copy(self):
        return ShiftDataset(
            self.raw_source.copy() if self.raw_source is not None else None,
            self.raw_target.copy() if self.raw_target is not None else None,
            self.source.copy(),
            self.target.copy(),
            self.source_groups.copy(),
            self.target_groups.copy(),
            self.feature_names,
            self.feature_types,
            self.scaler,
            self.group_fn,
            self.cf_generator,
            self.embed_fn,
        )

@dataclass
class Params:
    lr: float
    iters: int
    tol : float=1e-5
    n_clusters: Optional[int] = 10
    wreg: Optional[float] = None
    blur: float = 0.05


def load_breeds_rawpixels():
    import glob
    from PIL import Image

    np.random.seed(0)

    df = pd.read_json("../data/imagenetx/imagenet_x_val_multi_factor.jsonl", lines=True)
    hier = ClassHierarchy("../data/breeds/")
    level = 3 # Could be any number smaller than max level
    superclasses = hier.get_nodes_at_level(level)
    
    DG = BreedsDatasetGenerator("../data/breeds/")
    ret = DG.get_superclasses(level=4,
        Nsubclasses=6,
        split="bad",
        ancestor="n01861778",
        balanced=True)
    superclasses, subclass_split, label_map = ret
    def flatlist(l):
        return [item for sublist in l for item in sublist]


    source_classes = []
    target_classes = []
    for i in flatlist(subclass_split[0][1:3]):
        source_classes.append(i)
    for i in flatlist(subclass_split[1][1:3]):
        target_classes.append(i)

    source_df = df[df["class"].isin(source_classes)]
    target_df = df[df["class"].isin(target_classes)]
    source_files = source_df["file_name"].to_list()
    target_files = target_df["file_name"].to_list()
    source_labels = source_df["class"].isin(subclass_split[0][1]).to_numpy().astype(float)
    target_labels = target_df["class"].isin(subclass_split[1][1]).to_numpy().astype(float)

    source_img = []
    target_img = []
    all_img_files = list(glob.glob(f"{IMAGENET_DIR}/*/*.JPEG"))
    SIZE = 32
    for i, filename in enumerate(source_files):
        full_filename = next(f for f in all_img_files if filename in f)
        img = Image.open(full_filename).convert("RGB").resize((SIZE, SIZE))
        source_img.append(img)
    for i, filename in enumerate(target_files):
        full_filename = next(f for f in all_img_files if filename in f)
        img = Image.open(full_filename).convert("RGB").resize((SIZE, SIZE))
        target_img.append(img)
    source_img = np.stack([np.array(img) for img in source_img], axis=0)
    target_img = np.stack([np.array(img) for img in target_img], axis=0)

    source = source_img.reshape(source_img.shape[0], -1)[:298].astype(float)
    target = target_img.reshape(target_img.shape[0], -1)[:298].astype(float)
    features = [f"pixel_{i}" for i in range(source.shape[1])]

    source_groups = np.concatenate([
        (source_labels == 1)[:, np.newaxis], # pattern
        (source_labels == 0)[:, np.newaxis], # pattern
    ], axis=1).copy()
    target_groups = np.concatenate([
        (target_labels == 1)[:, np.newaxis], # background
        (target_labels == 0)[:, np.newaxis], # background
    ], axis=1).copy()

    def group_fn(raw_transformed_source, transformed_source):
        neighs = KNeighborsClassifier(n_neighbors=1).fit(target, np.arange(target.shape[0]))
        return target_groups[neighs.predict(transformed_source)]

    scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)
    print(source.shape[0], target.shape[0])
    print(source[0])

    dataset =  ShiftDataset(
        source,
        target,
        source,
        target,
        source_groups,
        target_groups,
        features,
        ["int"] * source.shape[1],
        scaler,
        group_fn,
        CfPixelGenerator(features),
        lambda x: x)
    params = {
        "kmeans": Params(lr=0.25, iters=500, n_clusters=20),
        "ot": Params(lr=1.0, iters=500),
        "dice": Params(lr=0.005, iters=200, wreg=1e-4),
    }
    return dataset, params


def load_breeds_superpixels():
    np.random.seed(0)

    df = pd.read_json("../data/imagenetx/imagenet_x_val_multi_factor.jsonl", lines=True)
    hier = ClassHierarchy("../data/breeds/")
    level = 3 # Could be any number smaller than max level
    superclasses = hier.get_nodes_at_level(level)
    
    DG = BreedsDatasetGenerator("../data/breeds/")
    ret = DG.get_superclasses(level=4,
        Nsubclasses=6,
        split="bad",
        ancestor="n01861778",
        balanced=True)
    superclasses, subclass_split, label_map = ret
    def flatlist(l):
        return [item for sublist in l for item in sublist]


    source_classes = []
    target_classes = []
    for i in flatlist(subclass_split[0][1:3]):
        source_classes.append(i)
    for i in flatlist(subclass_split[1][1:3]):
        target_classes.append(i)

    source_df = df[df["class"].isin(source_classes)]
    target_df = df[df["class"].isin(target_classes)]
    source_files = source_df["file_name"].to_list()
    target_files = target_df["file_name"].to_list()
    source_labels = source_df["class"].isin(subclass_split[0][1]).to_numpy().astype(float)
    target_labels = target_df["class"].isin(subclass_split[1][1]).to_numpy().astype(float)

    source_groups = np.concatenate([
        (source_labels == 1)[:, np.newaxis], # pattern
        (source_labels == 0)[:, np.newaxis], # pattern
    ], axis=1).copy()
    target_groups = np.concatenate([
        (target_labels == 1)[:, np.newaxis], # background
        (target_labels == 0)[:, np.newaxis], # background
    ], axis=1).copy()

    base_path = ""
    superpixels = pickle.load(open(f"{base_path}/concepts/superpixels.pkl", "rb"))
    masks = pickle.load(open(f"{base_path}/concepts/masks.pkl", "rb"))
    img_to_pixelids = pickle.load(open(f"{base_path}/concepts/img_to_pixelids.pkl", "rb"))
    superpixels_pca = pickle.load(open(f"{base_path}/concepts/superpixels_pca.pkl", "rb"))

    kmeans = KMeans(n_clusters=200, random_state=0).fit(superpixels_pca)
    pixelid_to_cls = kmeans.labels_

    def featurize(pixel_ids, pixelid_to_cls):
        features = np.zeros((len(pixel_ids), 200))
        for i, pixel_ids_img in enumerate(pixel_ids):
            for pixel_id in pixel_ids_img:
                features[i, pixelid_to_cls[pixel_id]] += 1
        return features

    source = featurize(img_to_pixelids[:len(source_labels)], pixelid_to_cls)
    target = featurize(img_to_pixelids[len(source_labels):], pixelid_to_cls)
    features = [f"cluster_{i}" for i in range(1, 201)]

    X = np.concatenate([source, target], axis=0)
    sel = SelectKBest(chi2, k=50).fit(X, np.concatenate([np.zeros(source.shape[0]), np.ones(target.shape[0])], axis=0))
    print(features[sel.get_support()])
    features = features[sel.get_support()]
    source = source[:298, sel.get_support()]
    target = target[:298, sel.get_support()]

    id_scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)
    print(source.shape[0], target.shape[0])

    return ShiftDataset(None, None, source, target, source_groups, target_groups, features, ["int"] * source.shape[1], id_scaler, None)

def language_embed(raw_samples: list[str]):
    # Retry up to 6 times with exponential backoff, starting at 1 second and maxing out at 20 seconds delay
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
        return openai.Embedding.create(input=[text], model=model, encoding_format="float")["data"][0]["embedding"]

    embeddings = []
    for sample in tqdm(raw_samples, desc="Embedding"):
        embedding = get_embedding(sample, model="text-embedding-ada-002")
        embeddings.append(torch.tensor(embedding))
    return torch.stack(embeddings, dim=0)

def decode_embeddings(embeddings: list[torch.Tensor]):
    text = []
    corrector = vec2text.load_corrector("text-embedding-ada-002")
    for embedding in tqdm(embeddings, desc="Decoding"):
        output = vec2text.invert_embeddings(
            embeddings=embedding.to("cuda"),
            corrector=corrector,
            num_steps=20,
            sequence_beam_width=4,
        )[0]
        text.append(output)
    return text

def get_demographic_counts(demo1, demo2, demographics_data,demographics_groups,
                           ngram, max_features, equalize_sizes=False, featurization="default", lda_epochs=200, lda_lr=0.001):
    # splitting between demographic attribute 1 and 2
    og_X0 = demographics_data[demo1]
    og_X0_group = demographics_groups[demo1]
    og_X1 = demographics_data[demo2]
    og_X1_group = demographics_groups[demo2]
    if equalize_sizes:
        print('Equalizing sizes.')
        print('OG sizes:', (og_X0.shape, og_X1.shape))
        if og_X0.shape[0] < og_X1.shape[0]:
            rng = np.random.RandomState(42)
            # if not os.path.exists(os.path.join(save_dir, "subsample_idxs")):
            subsample_idxs = rng.choice(len(og_X1), replace=False, size=len(og_X0))
            # else:
            #     subsample_idxs  = load_objs(os.path.join(save_dir, "subsample_idxs"))            
            og_X1 = og_X1[subsample_idxs]  # subsampling to equalize sizes
            og_X1_group = og_X1_group[subsample_idxs]
        elif og_X0.shape[0] > og_X1.shape[0]:
            rng = np.random.RandomState(42)
            # if not os.path.exists(os.path.join(save_dir, "subsample_idxs")):
            subsample_idxs = rng.choice(len(og_X0), replace=False, size=len(og_X1))
            # else:
            #     subsample_idxs  = load_objs(os.path.join(save_dir, "subsample_idxs"))  
            # subsample_idxs = rng.choice(len(og_X0), replace=False, size=len(og_X1))
            og_X0 = og_X0[subsample_idxs]
            og_X0_group = og_X0_group[subsample_idxs]
        print('New sizes:', (og_X0.shape, og_X1.shape))
    og_X_both = np.concatenate((og_X0, og_X1))
    
    # og_X_both = np.concatenate((og_X0, og_X1))
    
    
    if featurization == "default":
        ngram_vectorizer = CountVectorizer(ngram_range=(ngram,ngram),
                                        stop_words='english') #, max_features=max_features)
        vectorized_data = ngram_vectorizer.fit_transform(og_X_both).toarray()

        feature_name_ls = ngram_vectorizer.get_feature_names_out()

    elif featurization == "lda":
        vectorized_data_raw, feature_name_ls, non_empty_ids, etm_instance = obtain_lda_feat(og_X_both.tolist(), num_topics=max_features, epochs=lda_epochs, lr=lda_lr)
        min_max_scaler = preprocessing.MinMaxScaler()
        vectorized_data = min_max_scaler.fit_transform(vectorized_data_raw.cpu().numpy())
        og_X_both = np.array([og_X_both[idx] for idx in non_empty_ids]).astype(object)
        vectorized_data = vectorized_data[0:int(og_X_both.shape[0]/2)*2]
        og_X_both = og_X_both[0:int(og_X_both.shape[0]/2)*2]
        og_X0 = og_X_both[0:int(og_X_both.shape[0]/2)]
        og_X1 = og_X_both[int(og_X_both.shape[0]/2):]
        # og_X0 = np.array([og_X0[idx] for idx in non_empty_ids if idx < og_X0.shape[0]]).astype(object)
        # og_X1 = np.array([og_X1[idx-og_X0.shape[0]] for idx in non_empty_ids if idx >= og_X0.shape[0]]).astype(object)
        
        # vectorized_data = vectorized_data.cpu().numpy()
        feature_name_ls = np.array(feature_name_ls)
    elif featurization == "emb":
        vectorized_data = language_embed(og_X_both.tolist()).numpy()
        feature_name_ls = np.array(["emb_"+str(i) for i in range(vectorized_data.shape[1])])

    X0, X1 = vectorized_data[:og_X0.shape[0]].astype(float), vectorized_data[og_X0.shape[0]:].astype(float)
    
    if featurization == "default":
        X_combined = np.concatenate([X0, X1], axis=0)
        sel = SelectKBest(chi2, k=50).fit(X_combined, np.concatenate([np.zeros(X0.shape[0]), np.ones(X1.shape[0])], axis=0))
        feature_name_ls = list(feature_name_ls[sel.get_support()])
        X0 = X0[:, sel.get_support()]
        X1 = X1[:, sel.get_support()]
    if featurization == "default":
        return X0, X1, feature_name_ls, og_X0, og_X1, og_X0_group, og_X1_group, ngram_vectorizer
    else:
        return X0, X1, feature_name_ls, og_X0, og_X1, og_X0_group, og_X1_group, None

def featurize_nlp(data):
    all_source = []
    all_target = []
    for d in data:
        if d[1].item() == 0:
            all_source.append(d[0])
        else:
            all_target.append(d[0])
    all_source = np.array(all_source)
    all_target = np.array(all_target)

    # equalize sizes
    rng = np.random.RandomState(42)
    subsample_idxs = rng.choice(len(all_target), replace=False, size=1000)
    all_target = all_target[subsample_idxs]
    subsample_idxs = rng.choice(len(all_source), replace=False, size=1000)
    all_source = all_source[subsample_idxs]

    ngram_vectorizer = CountVectorizer(ngram_range=(1,1),
                                       stop_words='english')
    all_data = np.concatenate((all_source, all_target))
    vectorized_data = ngram_vectorizer.fit_transform(all_data).toarray()

    X0, X1 = vectorized_data[:len(all_source)].astype(float), vectorized_data[len(all_source):].astype(float)
    feature_name_ls = ngram_vectorizer.get_feature_names_out()

    X_combined = np.concatenate([X0, X1], axis=0)
    # sel = VarianceThreshold(threshold=(.9 * (1 - .9))).fit(X_combined)
    # feature_name_ls = list(feature_name_ls[sel.get_support()])
    # print("Resulting features:", len(feature_name_ls))
    sel = SelectKBest(chi2, k=50).fit(X_combined, np.concatenate([np.zeros(X0.shape[0]), np.ones(X1.shape[0])], axis=0))
    feature_name_ls = list(feature_name_ls[sel.get_support()])
    X0 = X0[:, sel.get_support()]
    X1 = X1[:, sel.get_support()]
    
    return X0, X1, feature_name_ls, all_source, all_target


def save_objs(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_objs(filename):
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj

def load_nlp(args, sample_count=4437):
    np.random.seed(0)
    data_dir = Path('../data/nlp')
    demo_base = 'male'
    demo1 = 'nontoxic'  # nontoxic
    demo2 = 'toxic'     # toxic
    group_names = np.array(['male',
            'female',
            'LGBTQ',
            'christian',
            'muslim',
            'other_religions',
            'black',
            'white',
            'identity_any',
            'severe_toxicity',
            'obscene',
            'threat',
            'insult',
            'identity_attack',
            'sexual_explicit'])

    dataset = get_dataset(dataset='civilcomments', unlabeled=False, root_dir=str(data_dir), download=True)
    train = dataset.get_subset('train', frac=1)

    demographics_data = {'male':[], 'female':[], 'lgbtq':[], 'christian':[], 'muslim':[],
                        'other religions':[], 'white':[], 'black':[]}
    demographics_groups = {'male':[], 'female':[], 'lgbtq':[], 'christian':[], 'muslim':[],
                        'other religions':[], 'white':[], 'black':[]}
    demographics_label = {'male':[], 'female':[], 'lgbtq':[], 'christian':[], 'muslim':[],
                        'other religions':[], 'white':[], 'black':[]}


    # splitting the training data up into the demographic groups
    all_data_arr = np.array(train, dtype=object)  # moving to np since indexing can be quirky in pytorch
    for metadata_col, corresponding_key in enumerate(demographics_data):
        rows_in_demographic = train.metadata_array[:, metadata_col] == 1
        demographics_data[corresponding_key] = all_data_arr[rows_in_demographic][:, 0]
        demographics_groups[corresponding_key] = train.metadata_array[rows_in_demographic]
        demographics_label[corresponding_key] = np.array([item.item() for item in all_data_arr[rows_in_demographic][:, 1]])

    
    
    # adding base_nontoxic and base_toxic to the demographics data and labels
    # demographics_data[demo1] = #demographics_data[demo_base][demographics_label[demo_base] == 0]
    # demographics_groups[demo1] = demographics_groups[demo_base][demographics_label[demo_base] == 0,0].view(-1,1)
    # demographics_label[demo1] = demographics_label[demo_base][demographics_label[demo_base] == 0]

    # demographics_data[demo2] = demographics_data[demo_base][demographics_label[demo_base] == 1]
    # demographics_groups[demo2] = demographics_groups[demo_base][demographics_label[demo_base] == 1,0].view(-1,1)
    # demographics_label[demo2] = demographics_label[demo_base][demographics_label[demo_base] == 1]

    if (not args.train) and os.path.exists(os.path.join(data_dir, "civil", args.featurization)):
        cached_folder = os.path.join(data_dir, "civil", args.featurization)
        bow_source = load_objs(os.path.join(cached_folder, "bow_source"))
        bow_target = load_objs(os.path.join(cached_folder, "bow_target"))
        feature_names = load_objs(os.path.join(cached_folder, "feature_names"))
        source = load_objs(os.path.join(cached_folder, "source"))
        target = load_objs(os.path.join(cached_folder, "target"))
        source_group = load_objs(os.path.join(cached_folder, "source_group"))
        target_group = load_objs(os.path.join(cached_folder, "target_group"))
        if args.featurization == "lda":
            etm_model = load_objs(os.path.join(cached_folder, "etm_model"))    
    else:
        all_labels = torch.tensor(all_data_arr[:,1].tolist())
        neg_ids = torch.nonzero(all_labels == 0).view(-1)
        neg_id_ids = torch.randperm(len(neg_ids))[0:sample_count]
        neg_ids = neg_ids[neg_id_ids]
        
        demographics_data[demo1] = all_data_arr[neg_ids][:, 0]
        demographics_groups[demo1] = train.metadata_array[neg_ids, 0:1]
        demographics_groups[demo1] = torch.cat([demographics_groups[demo1], 1 - demographics_groups[demo1]], dim=1)
        demographics_label[demo1] = all_labels[neg_ids].numpy()
        
        
        pos_ids = torch.nonzero(all_labels == 1).view(-1)
        pos_id_ids = torch.randperm(len(pos_ids))[0:sample_count]
        pos_ids = pos_ids[pos_id_ids]
        demographics_data[demo2] = all_data_arr[pos_ids][:, 0]
        demographics_groups[demo2] = train.metadata_array[pos_ids, 0:1]
        demographics_groups[demo2] = torch.cat([demographics_groups[demo2], 1 - demographics_groups[demo2]], dim=1)
        demographics_label[demo2] = all_labels[pos_ids].numpy()
        cached_folder = os.path.join(data_dir, "civil", args.featurization)
        os.makedirs(cached_folder, exist_ok=True)
        if args.featurization == "default":
            bow_source, bow_target, feature_names, source, target, source_group, target_group, _ = get_demographic_counts(demo1, demo2, demographics_data,demographics_groups,
                                                    1, args.max_feat_count, equalize_sizes=True, featurization=args.featurization)
        elif args.featurization == "lda":
            bow_source, bow_target, feature_names, source, target, source_group, target_group, etm_model = get_demographic_counts(demo1, demo2, demographics_data,demographics_groups,
                                                    1, args.max_feat_count, equalize_sizes=True, featurization=args.featurization)
            save_objs(etm_model, os.path.join(cached_folder, "etm_model"))    
        
        
        save_objs(bow_source, os.path.join(cached_folder, "bow_source"))
        save_objs(bow_target, os.path.join(cached_folder, "bow_target"))
        save_objs(feature_names, os.path.join(cached_folder, "feature_names"))
        save_objs(source, os.path.join(cached_folder, "source"))
        save_objs(target, os.path.join(cached_folder, "target"))
        save_objs(source_group, os.path.join(cached_folder, "source_group"))
        save_objs(target_group, os.path.join(cached_folder, "target_group"))
        

    id_scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)

    # Determine the groups of transformed samples by the closest target sample
    def group_fn(raw_transformed_source, transformed_source):
        neighs = KNeighborsClassifier(n_neighbors=1).fit(bow_target, np.arange(bow_target.shape[0]))
        bow_target_tensor = torch.from_numpy(bow_target)
        transformed_source_tensor = torch.from_numpy(transformed_source)
        pair_distance = torch.norm(bow_target_tensor.unsqueeze(0) - transformed_source_tensor.unsqueeze(1), dim=-1)
        closest_target = torch.argmin(pair_distance, dim=-1)
        return target_group.numpy()[closest_target.numpy()]

    def embed_fn(raw_samples):
        return extract_sentence_embeddings(raw_samples.tolist()).numpy()
    if args.featurization == "default":
        params = {
            "kmeans": Params(lr=0.1, iters=500, n_clusters=20),
            "ot": Params(lr=0.5, iters=500),
            "dice": Params(lr=0.5, iters=200, n_clusters=20, wreg=5e-5),
        }
        dataset = ShiftDataset(
            source,
            target,
            bow_source,
            bow_target,
            source_group.numpy(),
            target_group.numpy(),
            feature_names,
            ["int"] * bow_source.shape[1],
            id_scaler,
            group_fn,
            CfTextGenerator(source, feature_names),
            embed_fn)
    elif args.featurization== "lda":
        params = {
            "kmeans": Params(lr=0.05, iters=1000, n_clusters=5, wreg=5e-5, tol=1e-9),
            "ot": Params(lr=0.05, iters=1000, tol=1e-6),
            "dice": Params(lr=0.5, iters=4000, n_clusters=20, wreg=5e-5),
        }
        
        dataset = ShiftDataset(
            source,
            target,
            bow_source,
            bow_target,
            source_group.numpy(),
            target_group.numpy(),
            feature_names,
            ["float"] * bow_source.shape[1],
            id_scaler,
            group_fn,
            CfNLPTopicGenerator(feature_names, etm_model),
            embed_fn)
        
    

    # dataset = ShiftDataset(
    #     source,
    #     target,
    #     bow_source,
    #     bow_target,
    #     source_group.numpy(),
    #     target_group.numpy(),
    #     feature_names,
    #     ["int"] * bow_source.shape[1],
    #     id_scaler,
    #     group_fn,
    #     CfGenerator(feature_names),
    #     embed_fn)
    # params = {
    #     "kmeans": Params(lr=20.0, iters=100),
    #     "ot": Params(lr=0.1, iters=100),
    #     "dice": Params(lr=0.5, iters=200, n_clusters=20, wreg=5e-5),
    # }
        # return extract_sentence_embeddings(raw_samples.tolist()).numpy()

    # dataset = ShiftDataset(
    #     source,
    #     target,
    #     bow_source,
    #     bow_target,
    #     source_group.numpy(),
    #     target_group.numpy(),
    #     feature_names,
    #     ["int"] * bow_source.shape[1],
    #     id_scaler,
    #     group_fn,
    #     CfTextGenerator(source, feature_names),
    #     embed_fn)
    # params = {
    #     "kmeans": Params(lr=20.0, iters=100, n_clusters=20),
    #     "ot": Params(lr=0.1, iters=100),
    #     "dice": Params(lr=0.5, iters=200, n_clusters=20, wreg=5e-5),
    # }
    return dataset, params

def load_nlp_emb(args, sample_count=4437):
    np.random.seed(0)
    data_dir = Path('../data/nlp')
    demo_base = 'male'
    demo1 = 'nontoxic'  # nontoxic
    demo2 = 'toxic'     # toxic
    group_names = np.array(['male',
            'female',
            'LGBTQ',
            'christian',
            'muslim',
            'other_religions',
            'black',
            'white',
            'identity_any',
            'severe_toxicity',
            'obscene',
            'threat',
            'insult',
            'identity_attack',
            'sexual_explicit'])

    dataset = get_dataset(dataset='civilcomments', unlabeled=False, root_dir=str(data_dir), download=True)
    train = dataset.get_subset('train', frac=1)

    demographics_data = {'male':[], 'female':[], 'lgbtq':[], 'christian':[], 'muslim':[],
                        'other religions':[], 'white':[], 'black':[]}
    demographics_groups = {'male':[], 'female':[], 'lgbtq':[], 'christian':[], 'muslim':[],
                        'other religions':[], 'white':[], 'black':[]}
    demographics_label = {'male':[], 'female':[], 'lgbtq':[], 'christian':[], 'muslim':[],
                        'other religions':[], 'white':[], 'black':[]}


    # splitting the training data up into the demographic groups
    all_data_arr = np.array(train, dtype=object)  # moving to np since indexing can be quirky in pytorch
    for metadata_col, corresponding_key in enumerate(demographics_data):
        rows_in_demographic = train.metadata_array[:, metadata_col] == 1
        demographics_data[corresponding_key] = all_data_arr[rows_in_demographic][:, 0]
        demographics_groups[corresponding_key] = train.metadata_array[rows_in_demographic]
        demographics_label[corresponding_key] = np.array([item.item() for item in all_data_arr[rows_in_demographic][:, 1]])

    if os.path.exists(os.path.join(data_dir, "civil_emb", args.featurization)):
        cached_folder = os.path.join(data_dir, "civil_emb", args.featurization)
        source_emb = load_objs(os.path.join(cached_folder, "source_emb"))
        target_emb = load_objs(os.path.join(cached_folder, "target_emb"))
        feature_names = load_objs(os.path.join(cached_folder, "feature_names"))
        source = load_objs(os.path.join(cached_folder, "source"))
        target = load_objs(os.path.join(cached_folder, "target"))
        source_group = load_objs(os.path.join(cached_folder, "source_group"))
        target_group = load_objs(os.path.join(cached_folder, "target_group"))
    else:
        all_labels = torch.tensor(all_data_arr[:,1].tolist())
        neg_ids = torch.nonzero(all_labels == 0).view(-1)
        neg_id_ids = torch.randperm(len(neg_ids))[0:sample_count]
        neg_ids = neg_ids[neg_id_ids]
        
        demographics_data[demo1] = all_data_arr[neg_ids][:, 0]
        demographics_groups[demo1] = train.metadata_array[neg_ids, 0:1]
        demographics_groups[demo1] = torch.cat([demographics_groups[demo1], 1 - demographics_groups[demo1]], dim=1)
        demographics_label[demo1] = all_labels[neg_ids].numpy()
        
        pos_ids = torch.nonzero(all_labels == 1).view(-1)
        pos_id_ids = torch.randperm(len(pos_ids))[0:sample_count]
        pos_ids = pos_ids[pos_id_ids]
        demographics_data[demo2] = all_data_arr[pos_ids][:, 0]
        demographics_groups[demo2] = train.metadata_array[pos_ids, 0:1]
        demographics_groups[demo2] = torch.cat([demographics_groups[demo2], 1 - demographics_groups[demo2]], dim=1)
        demographics_label[demo2] = all_labels[pos_ids].numpy()
        cached_folder = os.path.join(data_dir, "civil_emb", args.featurization)
        os.makedirs(cached_folder, exist_ok=True)

        source_emb, target_emb, feature_names, source, target, source_group, target_group, _ = get_demographic_counts(
            demo1, demo2, demographics_data,demographics_groups,
            1, args.max_feat_count, equalize_sizes=True, featurization=args.featurization)

        save_objs(source_emb, os.path.join(cached_folder, "source_emb"))
        save_objs(target_emb, os.path.join(cached_folder, "target_emb"))
        save_objs(feature_names, os.path.join(cached_folder, "feature_names"))
        save_objs(source, os.path.join(cached_folder, "source"))
        save_objs(target, os.path.join(cached_folder, "target"))
        save_objs(source_group, os.path.join(cached_folder, "source_group"))
        save_objs(target_group, os.path.join(cached_folder, "target_group"))
        
    id_scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)
    # id_scaler = StandardScaler().fit(np.concatenate([source_emb, target_emb], axis=0))

    # Determine the groups of transformed samples by the closest target sample
    def group_fn(raw_transformed_source, transformed_source):
        bow_target_tensor = torch.from_numpy(target_emb)
        transformed_source_tensor = torch.from_numpy(transformed_source)
        pair_distance = torch.norm(bow_target_tensor.unsqueeze(0) - transformed_source_tensor.unsqueeze(1), dim=-1)
        closest_target = torch.argmin(pair_distance, dim=-1)
        return target_group.numpy()[closest_target.numpy()]

    def embed_fn(raw_samples):
        return raw_samples
        # return extract_sentence_embeddings(raw_samples.tolist()).numpy()

    params = {
        "kmeans": Params(lr=0.5, iters=200, n_clusters=50, tol=1e-9, blur=0.00001),
        "ot": Params(lr=0.1, iters=500, tol=1e-9, blur=0.00001),
        "dice": Params(lr=1e-12, iters=200, n_clusters=20, wreg=0),
    }
    dataset = ShiftDataset(
        None,
        None,
        source_emb,
        target_emb,
        source_group.numpy(),
        target_group.numpy(),
        feature_names,
        ["any"] * source_emb.shape[1],
        id_scaler,
        group_fn,
        CfIdentityGenerator(feature_names),
        # CfTextGenerator(source, feature_names),
        embed_fn)
        
    return dataset, params

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def extract_sentence_embeddings(dataset):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    # Tokenize sentences
    encoded_input = tokenizer(dataset, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

def cluster_embeddings(embeddings):
    kmeans = KMeans(n_clusters=10, random_state=0).fit(embeddings)
    groups = []
    for c in range(len(np.unique(kmeans.labels_))):
        groups.append(kmeans.labels_ == c)
        print(f"Samples in cluster {c}: {np.sum(kmeans.labels_ == c)}")

    return np.stack(groups, axis=1)


def load_fmow():
    np.random.seed(0)
    base_path = ""
    data_dir = Path(f'{base_path}datasets/fmow')
    dataset = get_dataset(dataset='fmow', unlabeled=False, root_dir=str(data_dir), download=True)

    val_data = dataset.get_subset(
        "val",
        transform=transforms.Compose(
            [transforms.Resize((32, 32))]
        ),
    )

    print(val_data.metadata_fields)
    print(torch.unique(val_data.metadata_array[:, 0]))
    print(torch.unique(val_data.metadata_array[:, 1]))
    print(torch.unique(val_data.metadata_array[:, 2]))
    source_mask = (torch.logical_and(val_data.metadata_array[:, 1] <= 12, val_data.metadata_array[:, 2] <= 2)).flatten()
    target_mask = (torch.logical_and(val_data.metadata_array[:, 1] >= 13, val_data.metadata_array[:, 2] <= 2)).flatten()
    print(torch.sum(source_mask))
    print(torch.sum(target_mask))
    source_groups = F.one_hot(torch.tensor(val_data.metadata_array[source_mask][:, 0]).long()).numpy()
    target_groups = F.one_hot(torch.tensor(val_data.metadata_array[target_mask][:, 0]).long()).numpy()
    source_data = WILDSSubset(val_data, torch.nonzero(source_mask).flatten(), transform=None)
    target_data = WILDSSubset(val_data, torch.nonzero(target_mask).flatten(), transform=None)
    print("size of source:", len(source_data))
    print("size of target:", len(target_data))
    source_imgs = [img for img, _, _ in source_data]
    target_imgs = [img for img, _, _ in target_data]
    source_img = np.stack([np.array(img) for img in source_imgs], axis=0)
    target_img = np.stack([np.array(img) for img in target_imgs], axis=0)

    num = min(len(source_img), len(target_img))
    source = source_img.reshape(source_img.shape[0], -1)[:num].astype(float)
    target = target_img.reshape(target_img.shape[0], -1)[:num].astype(float)
    features = [f"pixel_{i}" for i in range(source.shape[1])]

    scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)

    def group_fn(raw_transformed_source, transformed_source):
        neighs = KNeighborsClassifier(n_neighbors=1).fit(target, np.arange(target.shape[0]))
        return target_groups[neighs.predict(transformed_source)]

    dataset =  ShiftDataset(
        source,
        target,
        source,
        target,
        source_groups[:num],
        target_groups[:num],
        features,
        ["int"] * source.shape[1],
        scaler,
        group_fn,
        CfPixelGenerator(features),
        lambda x: x)
    params = {
        "kmeans": Params(lr=0.01, iters=500, n_clusters=20),
        "ot": Params(lr=0.5, iters=500),
        "dice": Params(lr=0.005, iters=1000, wreg=1e-4),
    }
    return dataset, params


def load_iwildcam():
    np.random.seed(0)
    base_path = ""
    data_dir = Path(f'{base_path}datasets/iwildcam')
    dataset = get_dataset(dataset='iwildcam', unlabeled=False, root_dir=str(data_dir), download=True)

    val_data = dataset.get_subset(
        "val",
        transform=transforms.Compose(
            [transforms.Resize((512, 512))]
        ),
    )
    source_data = WILDSSubset(val_data, torch.nonzero(val_data.metadata_array[:, 0] == 154).flatten(), transform=None)
    target_data = WILDSSubset(val_data, torch.nonzero(val_data.metadata_array[:, 0] == 275).flatten(), transform=None)
    print(len(source_data))
    print(len(target_data))
    source_imgs = [img for img, _, _ in source_data]
    target_imgs = [img for img, _, _ in target_data]
    print(torch.unique(source_data.metadata_array[:, 5]))
    print(torch.unique(target_data.metadata_array[:, 5]))
    source_img = np.stack([np.array(img) for img in source_imgs], axis=0)
    target_img = np.stack([np.array(img) for img in target_imgs], axis=0)

    source_groups = F.one_hot(torch.logical_or(source_data.metadata_array[:, 5] >= 20, source_data.metadata_array[:, 5] <= 8).long()).numpy()
    target_groups = F.one_hot(torch.logical_or(target_data.metadata_array[:, 5] >= 20, target_data.metadata_array[:, 5] <= 8).long()).numpy()
    print(np.sum(source_groups, axis=0))
    print(np.sum(target_groups, axis=0))

    if not os.path.exists(data_dir / "source_captions.pkl"):
        source_captions = img2text(source_imgs)
        target_captions = img2text(target_imgs)
        with open(data_dir / "source_captions.pkl", "wb") as f:
            pickle.dump(source_captions, f)
        with open(data_dir / "target_captions.pkl", "wb") as f:
            pickle.dump(target_captions, f)
    else:
        source_captions = pickle.load(open(data_dir / "source_captions.pkl", "rb"))
        target_captions = pickle.load(open(data_dir / "target_captions.pkl", "rb"))

    ngram_vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words='english') #, max_features=2000)
    vectorized_data = ngram_vectorizer.fit_transform(source_captions + target_captions).toarray()

    source = vectorized_data[:len(source_captions), :].astype(float)
    target = vectorized_data[len(source_captions):, :].astype(float)
    features = ngram_vectorizer.get_feature_names_out()

    X = np.concatenate([source, target], axis=0)
    sel = SelectKBest(chi2, k=50).fit(X, np.concatenate([np.zeros(source.shape[0]), np.ones(target.shape[0])], axis=0))
    print(features[sel.get_support()])
    features = features[sel.get_support()]

    num = min(len(source), len(target))
    source = source[:num, sel.get_support()]
    target = target[:num, sel.get_support()]
    scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)
    print("Sizes of the source and target:", source.shape[0], target.shape[0])

    def group_fn(raw_transformed_source, transformed_source):
        # How can we determine night vs day?
        neighs = KNeighborsClassifier(n_neighbors=1).fit(target, np.arange(target.shape[0]))
        return target_groups[neighs.predict(transformed_source)]

    def embed_fn(raw_samples):
        return resnet50_embed(raw_samples)

    dataset = ShiftDataset(
        source_img[:num],
        target_img[:num],
        source,
        target,
        source_groups[:num],
        target_groups[:num],
        features,
        ["int"] * source.shape[1],
        scaler,
        group_fn,
        CfDiffusionGenerator(source_captions[:num], features, finetuned=True),
        embed_fn)
    params = {
        "kmeans": Params(lr=1.0, iters=500, n_clusters=20),
        "ot": Params(lr=1.0, iters=500),
        "dice": Params(lr=0.5, iters=200, n_clusters=20, wreg=1e-3),
    }
    return dataset, params

def load_imagenet_concepts():
    np.random.seed(0)

    df = pd.read_json("../data/imagenetx/imagenet_x_val_multi_factor.jsonl", lines=True)
    hier = ClassHierarchy("../data/breeds/")
    level = 3 # Could be any number smaller than max level
    superclasses = hier.get_nodes_at_level(level)
    
    DG = BreedsDatasetGenerator("../data/breeds/")
    ret = DG.get_superclasses(level=4,
        Nsubclasses=6,
        split="bad",
        ancestor="n01861778",
        balanced=True)
    superclasses, subclass_split, label_map = ret
    def flatlist(l):
        return [item for sublist in l for item in sublist]


    source_classes = []
    target_classes = []
    for i in flatlist(subclass_split[0][1:3]):
        source_classes.append(i)
    for i in flatlist(subclass_split[1][1:3]):
        target_classes.append(i)

    source_df = df[df["class"].isin(source_classes)]
    target_df = df[df["class"].isin(target_classes)]
    source_files = source_df["file_name"].to_list()
    target_files = target_df["file_name"].to_list()
    source_labels = source_df["class"].isin(subclass_split[0][1]).to_numpy().astype(float)
    target_labels = target_df["class"].isin(subclass_split[1][1]).to_numpy().astype(float)

    source_groups = np.concatenate([
        (source_labels == 1)[:, np.newaxis], # pattern
        (source_labels == 0)[:, np.newaxis], # pattern
    ], axis=1).copy()
    target_groups = np.concatenate([
        (target_labels == 1)[:, np.newaxis], # background
        (target_labels == 0)[:, np.newaxis], # background
    ], axis=1).copy()

    source_img = []
    target_img = []
    all_img_files = list(glob.glob(f"{IMAGENET_DIR}/*/*.JPEG"))
    for i, filename in enumerate(source_files):
        full_filename = next(f for f in all_img_files if filename in f)
        img = Image.open(full_filename).convert("RGB").resize((512, 512))
        source_img.append(img)
    for i, filename in enumerate(target_files):
        full_filename = next(f for f in all_img_files if filename in f)
        img = Image.open(full_filename).convert("RGB").resize((512, 512))
        target_img.append(img)

    if not os.path.exists("../data/imagenet_emb/concepts"):
        X = source_img + target_img
        all_labels = []
        all_bboxes = []
        rand_subset = np.random.choice(len(X), 200, replace=False)
        for i in tqdm(rand_subset):
            segments = slic(np.array(X[i]), n_segments=8*8, compactness=50, sigma=1, start_label=1)
            all_labels.append(segments)

        for img_mask in all_labels:
            bboxes = []
            props = regionprops(img_mask)
            for prop in props:
                x1 = prop.bbox[1]
                y1 = prop.bbox[0]

                x2 = prop.bbox[3]
                y2 = prop.bbox[2]

                bboxes.append([x1, y1, x2, y2])
            all_bboxes.append(bboxes)

        patches = []
        for i, bboxes_img, labels_img in zip(rand_subset, all_bboxes, all_labels):
            for j, bbox in enumerate(bboxes_img):
                img = Image.fromarray(X[i] * (labels_img == j + 1)[:, :, None])
                img = PIL.ImageOps.pad(img.crop(bbox), (224, 224))
                patches.append(img)
        print("Number of patches:", len(patches))
        save_objs(patches, "../data/imagenet_emb/patches_img")

        # Encode all the patches
        # patches = np.stack([np.moveaxis(np.array(img), -1, 0) for img in patches], axis=0)
        model = timm.create_model('vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k', pretrained=True, num_classes=0).to("cuda")
        model.eval()
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        def transform_batch(batch):
            return torch.stack([transform(img) for img in batch])
        patches_emb = _batch_inference(model, patches, batch_size=64, processor=transform_batch)
        patches_emb = patches_emb / np.linalg.norm(patches_emb, axis=1, keepdims=True)

        # Cluster to get the concepts
        kmeans = KMeans(n_clusters=100, random_state=0).fit(patches_emb.numpy())
        concepts = torch.tensor(kmeans.cluster_centers_ / np.linalg.norm(kmeans.cluster_centers_, axis=1, keepdims=True))

        # Encode all samples and extract concepts from them as features
        X = np.stack([np.moveaxis(np.array(img.resize((224, 224))), -1, 0) for img in X], axis=0)
        X_emb = _batch_inference(model, torch.tensor(X).float(), batch_size=64)

        source_img = X[:len(source_img)]
        source_emb = X_emb[:len(source_img)]
        target_img = X[len(source_img):]
        target_emb = X_emb[len(source_img):]

        Path(f"../data/imagenet_emb").mkdir(parents=True, exist_ok=True)
        save_objs(patches_emb.cpu(), "../data/imagenet_emb/patches_emb")
        save_objs(concepts, "../data/imagenet_emb/concepts")
        save_objs(source_img, "../data/imagenet_emb/source_img")
        save_objs(source_emb, "../data/imagenet_emb/source_emb")
        save_objs(target_img, "../data/imagenet_emb/target_img")
        save_objs(target_emb, "../data/imagenet_emb/target_emb")
        # print(features)
    else:
        concepts = load_objs("../data/imagenet_emb/concepts")
        source_img = load_objs("../data/imagenet_emb/source_img")
        source_emb = load_objs("../data/imagenet_emb/source_emb")
        target_img = load_objs("../data/imagenet_emb/target_img")
        target_emb = load_objs("../data/imagenet_emb/target_emb")

    source = source_emb #@ concepts.T
    target = target_emb #@ concepts.T

    source = source[:298]
    target = target[:298]
    scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)
    print(source.shape[0], target.shape[0])

    def vit_classify_emb(embeddings):
        model = timm.create_model('vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k', pretrained=True).to("cuda")
        model.eval()
        print("Embeddings shape:", embeddings.shape)
        def forward_emb(emb):
            x = model.fc_norm(emb)
            x = model.head_drop(x)
            x = model.head(x)
            return x
        preds = _batch_inference(forward_emb, torch.tensor(embeddings).float(), batch_size=64).numpy()
        return np.argmax(preds, axis=1)

    def group_fn(raw_transformed_source, transformed_source):
        # classes = vit_classify_emb(transformed_source)
        # cf_group1 = np.isin(classes, subclass_split[1][1]).astype(int)
        # cf_group2 = np.isin(classes, subclass_split[1][2]).astype(int)
        # new_groups = np.stack([cf_group1, cf_group2], axis=1)
        # return new_groups
        target_tensor = target
        transformed_source_tensor = torch.from_numpy(transformed_source)
        pair_distance = torch.norm(target_tensor.unsqueeze(0) - transformed_source_tensor.unsqueeze(1), dim=-1)
        closest_target = torch.argmin(pair_distance, dim=-1)
        return target_groups[closest_target.numpy()]

    def embed_fn(raw_samples):
        return raw_samples
        # return resnet50_embed(raw_samples)

    dataset = ShiftDataset(
        None, #source_img[:298],
        None, #target_img[:298],
        source.numpy(),
        target.numpy(),
        source_groups[:298],
        target_groups[:298],
        [f"emb_{i}" for i in range(source.shape[1])],
        ["any"] * source.shape[1],
        scaler,
        group_fn,
        CfIdentityGenerator([f"emb_{i}" for i in range(source.shape[1])]),
        # CfInvertedDiffusionGenerator(source_captions[:298], features),
        # CfDiffusionGenerator(source_captions[:298], features, finetuned=False),
        embed_fn)
    params = {
        "kmeans": Params(lr=0.5, iters=500, n_clusters=30),
        "ot": Params(lr=0.1, iters=200),
        "dice": Params(lr=0.5, iters=200, n_clusters=20, wreg=1e-3),
    }
    return dataset, params

def load_imagenet():
    np.random.seed(0)

    df = pd.read_json("../data/imagenetx/imagenet_x_val_multi_factor.jsonl", lines=True)
    hier = ClassHierarchy("../data/breeds/")
    level = 3 # Could be any number smaller than max level
    superclasses = hier.get_nodes_at_level(level)
    
    DG = BreedsDatasetGenerator("../data/breeds/")
    ret = DG.get_superclasses(level=4,
        Nsubclasses=6,
        split="bad",
        ancestor="n01861778",
        balanced=True)
    superclasses, subclass_split, label_map = ret
    def flatlist(l):
        return [item for sublist in l for item in sublist]


    source_classes = []
    target_classes = []
    for i in flatlist(subclass_split[0][1:3]):
        source_classes.append(i)
    for i in flatlist(subclass_split[1][1:3]):
        target_classes.append(i)

    source_df = df[df["class"].isin(source_classes)]
    target_df = df[df["class"].isin(target_classes)]
    source_files = source_df["file_name"].to_list()
    target_files = target_df["file_name"].to_list()
    source_labels = source_df["class"].isin(subclass_split[0][1]).to_numpy().astype(float)
    target_labels = target_df["class"].isin(subclass_split[1][1]).to_numpy().astype(float)

    source_groups = np.concatenate([
        (source_labels == 1)[:, np.newaxis], # pattern
        (source_labels == 0)[:, np.newaxis], # pattern
    ], axis=1).copy()
    target_groups = np.concatenate([
        (target_labels == 1)[:, np.newaxis], # background
        (target_labels == 0)[:, np.newaxis], # background
    ], axis=1).copy()

    # source_captions = pickle.load(open("../data/imagenetx/source_captions.pkl", "rb"))
    # target_captions = pickle.load(open("../data/imagenetx/target_captions.pkl", "rb"))
    base_path = ""
    dataset = pickle.load(open(f"{base_path}/imagenet/imagenet_captions.pkl", "rb"))
    source_captions = [dataset[i][1] for i in range(len(source_labels))]
    target_captions = [dataset[i][1] for i in range(len(source_labels), len(source_labels) + len(target_labels))]

    source_captions = [", ".join(cap.split(", ")) for cap in source_captions]
    target_captions = [", ".join(cap.split(", ")) for cap in target_captions]

    ngram_vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words='english') #, max_features=2000)
    vectorized_data = ngram_vectorizer.fit_transform(source_captions + target_captions).toarray()

    source = vectorized_data[:len(source_captions), :].astype(float)
    target = vectorized_data[len(source_captions):, :].astype(float)
    features = ngram_vectorizer.get_feature_names_out()

    source_img = []
    target_img = []
    all_img_files = list(glob.glob(f"{IMAGENET_DIR}/*/*.JPEG"))
    for i, filename in enumerate(source_files):
        full_filename = next(f for f in all_img_files if filename in f)
        img = Image.open(full_filename).convert("RGB").resize((512, 512))
        source_img.append(img)
    for i, filename in enumerate(target_files):
        full_filename = next(f for f in all_img_files if filename in f)
        img = Image.open(full_filename).convert("RGB").resize((512, 512))
        target_img.append(img)
    source_img = np.stack([np.array(img) for img in source_img], axis=0)
    target_img = np.stack([np.array(img) for img in target_img], axis=0)
    # print(features)

    X = np.concatenate([source, target], axis=0)
    sel = SelectKBest(chi2, k=50).fit(X, np.concatenate([np.zeros(source.shape[0]), np.ones(target.shape[0])], axis=0))
    print(features[sel.get_support()])
    features = features[sel.get_support()]
    source = source[:298, sel.get_support()]
    target = target[:298, sel.get_support()]
    scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)
    print(source.shape[0], target.shape[0])

    def group_fn(raw_transformed_source, transformed_source):
        classes = resnet50_classify(raw_transformed_source)
        cf_group1 = np.isin(classes, subclass_split[1][1]).astype(int)
        cf_group2 = np.isin(classes, subclass_split[1][2]).astype(int)
        new_groups = np.stack([cf_group1, cf_group2], axis=1)
        return new_groups

    def embed_fn(raw_samples):
        return resnet50_embed(raw_samples)

    dataset = ShiftDataset(
        source_img[:298],
        target_img[:298],
        source,
        target,
        source_groups[:298],
        target_groups[:298],
        features,
        ["int"] * source.shape[1],
        scaler,
        group_fn,
        CfInvertedDiffusionGenerator(source_captions[:298], features),
        # CfInvertedDiffusionGenerator(source_captions[:298], features),
        # CfDiffusionGenerator(source_captions[:298], features, finetuned=False),
        embed_fn)
    params = {
        "kmeans": Params(lr=0.5, iters=200, n_clusters=20),
        "ot": Params(lr=1.0, iters=200),
        "dice": Params(lr=0.5, iters=200, n_clusters=20, wreg=1e-3),
    }
    return dataset, params


def load_breast():
    np.random.seed(0)

    COLUMN_NAMES = [
        "diagnosis", "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean"
    ]
    raw_data = pd.read_csv(
        "../data/breast_cancer/data.csv",
        index_col=0)
    raw_data.drop(raw_data[(raw_data == '?').sum(axis=1) != 0].index, inplace=True)
    raw_data = raw_data[COLUMN_NAMES].dropna()

    bc_source_df = raw_data.query('diagnosis == "B"').sample(212)
    bc_target_df = raw_data.query('diagnosis == "M"').sample(212)
    bc_source = bc_source_df.drop(columns='diagnosis').to_numpy().astype(float)
    bc_target = bc_target_df.drop(columns='diagnosis').to_numpy().astype(float)

    # bc_scaler = preprocessing.StandardScaler().fit(bc_source)
    bc_scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)
    # bc_source = bc_scaler.transform(bc_source)
    # bc_target = bc_scaler.transform(bc_target)

    bc_source_labels = np.concatenate([
        (bc_source[:, i] >= np.percentile(bc_source[:, i], 75)).astype(int)[:, np.newaxis] for i in range(1, bc_source.shape[1])],
        axis=1)
    bc_target_labels = np.concatenate([
        (bc_target[:, i] >= np.percentile(bc_target[:, i], 75)).astype(int)[:, np.newaxis] for i in range(1, bc_source.shape[1])],
        axis=1)
    bc_feature_names = COLUMN_NAMES[1:]
    bc_feasible_names = ["radius**2 / area above third quartile", "radius**2 / area between first and third quartile", "radius**2 / area below first quartile",]
    bc_source_feasible_groups = np.concatenate([
        (((bc_source[:, 0]**2) / bc_source[:, 3]) > np.percentile(((bc_source[:, 0]**2) / bc_source[:, 3]), 75)).astype(int)[:, np.newaxis],
        (
            (((bc_source[:, 0]**2) / bc_source[:, 3]) >= np.percentile(((bc_source[:, 0]**2) / bc_source[:, 3]), 25)) &
            (((bc_source[:, 0]**2) / bc_source[:, 3]) <= np.percentile(((bc_source[:, 0]**2) / bc_source[:, 3]), 75))
        ).astype(int)[:, np.newaxis],
        (((bc_source[:, 0]**2) / bc_source[:, 3]) < np.percentile(((bc_source[:, 0]**2) / bc_source[:, 3]), 25)).astype(int)[:, np.newaxis],
    ], axis=1)
    bc_target_feasible_groups = np.concatenate([
        (((bc_target[:, 0]**2) / bc_target[:, 3]) > np.percentile(((bc_target[:, 0]**2) / bc_target[:, 3]), 75)).astype(int)[:, np.newaxis],
        (
            (((bc_target[:, 0]**2) / bc_target[:, 3]) >= np.percentile(((bc_target[:, 0]**2) / bc_target[:, 3]), 25)) &
            (((bc_target[:, 0]**2) / bc_target[:, 3]) <= np.percentile(((bc_target[:, 0]**2) / bc_target[:, 3]), 75))
        ).astype(int)[:, np.newaxis],
        (((bc_target[:, 0]**2) / bc_target[:, 3]) < np.percentile(((bc_target[:, 0]**2) / bc_target[:, 3]), 25)).astype(int)[:, np.newaxis],
    ], axis=1)

    def get_groups(raw_transformed_source, transformed_source):
        return np.concatenate([
            (((transformed_source[:, 0]**2) / transformed_source[:, 3]) > np.percentile(((bc_target[:, 0]**2) / bc_target[:, 3]), 75)).astype(int)[:, np.newaxis],
            (
                (((transformed_source[:, 0]**2) / transformed_source[:, 3]) >= np.percentile(((bc_target[:, 0]**2) / bc_target[:, 3]), 25)) &
                (((transformed_source[:, 0]**2) / transformed_source[:, 3]) <= np.percentile(((bc_target[:, 0]**2) / bc_target[:, 3]), 75))
            ).astype(int)[:, np.newaxis],
            (((transformed_source[:, 0]**2) / transformed_source[:, 3]) < np.percentile(((bc_target[:, 0]**2) / bc_target[:, 3]), 25)).astype(int)[:, np.newaxis],
        ], axis=1)
    
    types = ["float"] * bc_source.shape[1]
    dataset = ShiftDataset(None, None, bc_source, bc_target, bc_source_feasible_groups, bc_target_feasible_groups, bc_feature_names, types, bc_scaler, get_groups, CfIdentityGenerator(bc_feature_names), lambda x: x)
    params = {
        "kmeans": Params(lr=0.1, iters=500, n_clusters=20),
        "ot": Params(lr=5, iters=500),
        "dice": Params(lr=0.5, iters=200, n_clusters=20, wreg=1e-3),
    }
    return dataset, params

def load_adult():
    np.random.seed(0)
    COLUMN_NAMES = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income']
    raw_data = pd.read_csv("../data/adult/adult.data", names=COLUMN_NAMES, skipinitialspace=True)
    subset = ["age", "education-num", "race", "sex", "income", "marital-status", "occupation", "workclass"]
    raw_data = raw_data[subset]
    raw_data = pd.get_dummies(raw_data, columns=[
        "workclass", "marital-status", "occupation", "race"
    ])
    # print(np.unique(raw_data["race"]))
    # for r in np.unique(raw_data["race"]):
    #     print(np.sum(raw_data["race"] == r))
    binarizer = LabelBinarizer(neg_label=0, pos_label=1)
    print(np.unique(raw_data["income"]))
    raw_data["income"] = 1 - binarizer.fit_transform(raw_data["income"])
    raw_data["sex"] = binarizer.fit_transform(raw_data["sex"])
    # raw_data["race"] = binarizer.fit_transform(raw_data["race"])
    # raw_data["race"] = raw_data["race"].map(lambda v: 1 if v == "White" else 0)
    labels = raw_data["income"]
    raw_data = raw_data.fillna(0)
    adult_raw_data = raw_data
    adult_targets = raw_data["income"]
    print(raw_data.columns)

    adult_source_data = raw_data[raw_data["income"] == 1]
    adult_target_data = raw_data[raw_data["income"] == 0]
    samples = min(adult_source_data.shape[0], adult_target_data.shape[0])
    adult_source_data = adult_source_data.sample(samples, replace=False)
    adult_target_data = adult_target_data.sample(samples, replace=False)
    index_of_income = adult_source_data.columns.get_loc("income")
    adult_source = adult_source_data.to_numpy().astype(float)[:, [i for i in range(adult_source_data.shape[1]) if i != index_of_income]]
    adult_target = adult_target_data.to_numpy().astype(float)[:, [i for i in range(adult_source_data.shape[1]) if i != index_of_income]]
    adult_feature_names = [adult_source_data.columns[i] for i in range(adult_source_data.shape[1]) if i != index_of_income]
    # adult_source_labels = adult_source_data.to_numpy().astype(int)[:, 4:]
    # adult_target_labels = adult_target_data.to_numpy().astype(int)[:, 4:]
    white_index = adult_source_data.columns.get_loc("race_White") - 1
    black_index = adult_source_data.columns.get_loc("race_Black") - 1
    asian_index = adult_source_data.columns.get_loc("race_Asian-Pac-Islander") - 1
    amindian_index = adult_source_data.columns.get_loc("race_Amer-Indian-Eskimo") - 1
    other_index = adult_source_data.columns.get_loc("race_Other") - 1
    adult_source_feasible_groups = np.concatenate([
        (adult_source[:, white_index] == 1).astype(int)[:, np.newaxis],
        (adult_source[:, black_index] == 1).astype(int)[:, np.newaxis],
        # (adult_source[:, 0] < 30).astype(int)[:, np.newaxis],
        # (adult_source[:, 0] >= 30).astype(int)[:, np.newaxis],
    ], axis=1)
    adult_target_feasible_groups = np.concatenate([
        (adult_target[:, white_index] == 1).astype(int)[:, np.newaxis],
        (adult_target[:, black_index] == 1).astype(int)[:, np.newaxis],
        # (adult_target[:, 0] < 30).astype(int)[:, np.newaxis],
        # (adult_target[:, 0] >= 30).astype(int)[:, np.newaxis],
    ], axis=1)
    # adult_labels = np.concatenate([adult_source_labels, adult_target_labels])
    # adult_scaler = preprocessing.StandardScaler().fit(np.concatenate([adult_source, adult_target]))
    adult_scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)
    # adult_scaler = MaxAbsScaler().fit(np.concatenate([adult_source, adult_target]))
    # adult_source = adult_scaler.transform(adult_source)
    # adult_target = adult_scaler.transform(adult_target)
    print("average age:", np.mean(adult_source[:, 0]))
    print("Percent below 55:", np.sum(adult_source[:, 0] < 55) / adult_source.shape[0])
    # print("Percent between 25 and 55:", np.sum((adult_source[:, 0] >= 25) & (adult_source[:, 0] < 55)) / adult_source.shape[0])
    print("Percent above 55:", np.sum(adult_source[:, 0] >= 55) / adult_source.shape[0])

    def get_groups(raw_transformed_source, transformed_source):
        return np.concatenate([
            (transformed_source[:, white_index] == 1).astype(int)[:, np.newaxis],
            (transformed_source[:, black_index] == 1).astype(int)[:, np.newaxis],
            # (transformed_source[:, 0] < 30).astype(int)[:, np.newaxis],
            # (transformed_source[:, 0] >= 30).astype(int)[:, np.newaxis],
        ], axis=1)

    types = ["int", "int"] + ["binary"] * (adult_source.shape[1] - 2)

    dataset =  ShiftDataset(
        None,
        None,
        adult_source,
        adult_target,
        adult_source_feasible_groups,
        adult_target_feasible_groups,
        adult_feature_names,
        types,
        adult_scaler,
        get_groups,
        CfIdentityGenerator(adult_feature_names),
        lambda x: x)
    params = {
        "kmeans": Params(lr=0.1, iters=200, n_clusters=10, blur=0.1),
        "ot": Params(lr=0.5, iters=500),
        "dice": Params(lr=0.1, iters=500, n_clusters=20, wreg=1e-4),
    }
    return dataset, params

def perturb_features(data, types, rs=[1]):
    print("rs:", rs)
    rand = np.random.uniform(size=data.shape)
    nonzero_feats = np.nonzero(np.sum(data, axis=0) > 0)[0]
    perturb_feats = np.random.choice(nonzero_feats, size=int(1 * nonzero_feats.shape[0] / 2), replace=False) 
    for r in rs: #, 2, 4, 8, 16]:
        perturbation = np.zeros(data.shape)
        for perturb_feat in perturb_feats:
            if types[perturb_feat] == "binary":
                # print(perturb_feat, "binary")
                # one_mask = data[:, perturb_feat] == 1
                # zero_mask = data[:, perturb_feat] == 0
                perturb_mask = (rand[:, perturb_feat] < (r / 100)).astype(bool)
                pert = np.random.choice([0, 1], size=np.sum(perturb_mask))
                perturbation[:, perturb_feat][perturb_mask] = -data[:, perturb_feat][perturb_mask] + pert
                # perturbation[:, perturb_feat][zero_mask & perturb_mask] = 1

                # if np.random.choice([-1, 1]) == 1: #np.sum(data[:, perturb_feat]) > data.shape[0] / 2:
                #     one_mask = data[:, perturb_feat] == 1
                #     perturb_mask = (rand[:, perturb_feat] < (r / 500)).astype(bool)
                #     perturbation[:, perturb_feat][one_mask & perturb_mask] = -1
                #     # print(np.sum(one_mask & perturb_mask))
                # else:
                #     zero_mask = data[:, perturb_feat] == 0
                #     perturb_mask = (rand[:, perturb_feat] < (r / 500)).astype(bool)
                #     perturbation[:, perturb_feat][zero_mask & perturb_mask] = 1
                #     # print(np.sum(zero_mask & perturb_mask))
            elif types[perturb_feat] == "int":
                print("int type")
                # print(perturb_feat, "int")
                low = np.min(data[:, perturb_feat]) / 4
                high = np.max(data[:, perturb_feat]) / 4
                perturb_mask = (rand[:, perturb_feat] < (r / 100)).astype(bool)
                rand_vals = np.random.uniform(size=np.sum(perturb_mask), low=low, high=high).round(0).astype(int)
                perturbation[:, perturb_feat][perturb_mask] = -data[:, perturb_feat][perturb_mask] + rand_vals
                # if np.random.choice([-1, 1]) == 1:
                #     stdev = 1
                #     perturb_mask = (rand[:, perturb_feat] < (r / 500)).astype(bool)
                #     perturbation[:, perturb_feat][perturb_mask] = stdev
                # else:
                #     stdev = 1
                #     perturb_mask = (rand[:, perturb_feat] < (r / 500)).astype(bool)
                #     pos_mask = data[:, perturb_feat] >= stdev
                #     perturbation[:, perturb_feat][pos_mask & perturb_mask] = -stdev
            else:
                # stdev = 0.1 * np.std(data[:, perturb_feat])
                mean = np.mean(data[:, perturb_feat])
                std = 0.5
                # print(perturb_feat, "float", stdev)
                perturb_mask = (rand[:, perturb_feat] < ((r) / 100)).astype(bool)
                pert = np.random.normal(size=np.sum(perturb_mask), loc=data[:, perturb_feat][perturb_mask], scale=std)
                # pos_mask = data[:, perturb_feat] >= stdev
                # rand_vals = np.random.uniform(size=np.sum(perturb_mask), low=low, high=high)
                print(-data[:, perturb_feat][perturb_mask] + pert)
                perturbation[:, perturb_feat][perturb_mask] = -data[:, perturb_feat][perturb_mask] + pert
        yield r, perturbation.astype(float)


def train_method(dataset_name, method, shift_dataset: ShiftDataset, params: dict[str, Params], use_groups=True, init_source_t=None):
    output_dict = {}
    no_group_s = np.ones((shift_dataset.source.shape[0], 1))
    no_group_t = np.ones((shift_dataset.target.shape[0], 1))
    if use_groups:
        shift_dataset.source_groups = np.concatenate([shift_dataset.source_groups, no_group_s], axis=1)
        shift_dataset.target_groups = np.concatenate([shift_dataset.target_groups, no_group_t], axis=1)
    if method == "kmeans":
        x_s, centroids, shifts = group_kmeans_shift_transport(
            shift_dataset.scaler.transform(shift_dataset.source),
            shift_dataset.scaler.transform(shift_dataset.target),
            shift_dataset.source_groups if use_groups else no_group_s,
            shift_dataset.target_groups if use_groups else no_group_t,
            shift_dataset.source.shape[1], params[method].n_clusters,
            params[method].lr, params[method].iters, init_clusters=init_source_t, tol=params[method].tol, blur=params[method].blur,
            loss_type="sum")
        x_s = shift_dataset.scaler.inverse_transform(x_s)
        output_dict['centroids'] = centroids
        output_dict['shift'] = shifts
    elif method == "ot":
        x_s = group_feature_transport(
            shift_dataset.scaler.transform(shift_dataset.source),
            shift_dataset.scaler.transform(shift_dataset.target),
            shift_dataset.source_groups if use_groups else no_group_s,
            shift_dataset.target_groups if use_groups else no_group_t,
            shift_dataset.source.shape[1],
            params[method].lr, params[method].iters,
            init_x_s=shift_dataset.scaler.transform(init_source_t) if init_source_t is not None else None,
            tol=params[method].tol, blur=params[method].blur,
            loss_type="max")
        x_s = shift_dataset.scaler.inverse_transform(x_s)
        # shifts = x_s - source
    elif method == "dice":
        np.random.seed(42)
        source_selection = np.random.choice(np.arange(shift_dataset.source.shape[0]), size=min(300, shift_dataset.source.shape[0]), replace=False)

        full_data = np.concatenate([shift_dataset.source, shift_dataset.target], axis=0)
        full_df_dict = {}
        # continuous_features = []
        # categorical_features = []
        for idx in range(shift_dataset.source.shape[1]):
            feature_col = full_data[:,idx]
            full_df_dict[shift_dataset.feature_names[idx]] = feature_col
            # if shift_dataset.feature_types[idx] == "float":
            #     continuous_features.append(shift_dataset.feature_names[idx])
            # else:
            #     categorical_features.append(shift_dataset.feature_names[idx])
    
        labels = np.concatenate([np.ones(shift_dataset.source.shape[0]), np.zeros(shift_dataset.target.shape[0])])
        print(labels.shape, full_data.shape)
        full_df_dict["_label"] = labels
        full_df = pd.DataFrame(full_df_dict, dtype=float)\
            .astype({"_label": int})
            # .astype({col: float for col in continuous_features})\
            # .astype({col: int for col in categorical_features})\
        # print(full_df / full_df.max())
        source_df = full_df.iloc[0:shift_dataset.source.shape[0]]
        target_df = full_df.iloc[shift_dataset.source.shape[0]:]

        if dataset_name == "adult" or dataset_name == "breast":
            net = FFNetwork(shift_dataset.source.shape[1])
        # elif dataset_name == "nlp-amazon":
        #     net = PTNN(shift_dataset.source.shape[1], 1)
        # elif dataset_name == "nlp" and args.featurization == "lda":
        #     net = FFNetwork(shift_dataset.source.shape[1])
        else:
            net = PTLogisticRegression(shift_dataset.source.shape[1], 1)
    
        x_train = torch.from_numpy(np.concatenate([shift_dataset.source, shift_dataset.target], axis=0)).float()
        x_train = x_train / torch.max(x_train, dim=0)[0]
        x_train[torch.isnan(x_train)] = 0
        group_mat = np.concatenate(
            [shift_dataset.source_groups if use_groups else no_group_s,
             shift_dataset.target_groups if use_groups else no_group_t,
            ], axis=0)
        
        no_group = 1 - (np.sum(group_mat, axis=1) > 0)[:, np.newaxis]
        if np.sum(no_group) > 0:
            group_mat = torch.from_numpy(np.concatenate([group_mat, no_group], axis=1)).float()
        else:
            group_mat = torch.from_numpy(group_mat).float()
        y = torch.from_numpy(np.concatenate([np.ones(shift_dataset.source.shape[0]), np.zeros(shift_dataset.target.shape[0])], axis=0)).float()[:, None]

        dro_training(
            net, 
            x_train,
            y,
            group_mat,
            params[method].iters,
            params[method].lr,
            params[method].wreg,
            loss_type="dro")
    
        x_s = get_dice_transformed(
                net,
                full_df / full_df.max(),
                # source_df.drop(columns=['_label']).iloc[source_selection],
                (source_df / full_df.max()).drop(columns=['_label']).iloc[source_selection],
                "_label",
                0).to_numpy()
        x_s = x_s * full_df.drop(columns=['_label']).max().to_numpy()
        # x_s = x_s
        print(x_s)
        # shifts = x_s - source[source_selection]
    else:
        raise ValueError("Invalid method")


    # inverse_x_s = scaler.inverse_transform(x_s)
    # inverse_shifts = scaler.inverse_transform(shifts)
    for type, i in zip(shift_dataset.feature_types, range(shift_dataset.source.shape[1])):
        if type == "int":
            x_s[:, i] = np.round(x_s[:, i])
        if type == "binary":
            # x_s[:, i] = np.round(x_s[:, i])
            x_s[:, i] = np.clip(x_s[:, i], 0, 1)
        if type == "float":
            x_s[:,i] = (x_s[:,i] > 0).reshape(-1).astype(float)*x_s[:,i]
    # x_s_round = scaler.transform(inverse_x_s)
    if method == "dice":
        output_dict['shifts'] = x_s - shift_dataset.source[source_selection]
    else:
        output_dict['shifts'] = x_s - shift_dataset.source
    return output_dict


def train(dataset, method, shift_dataset: ShiftDataset, params: dict[str, Params]):
    if args.no_groups:
        X0_embeddings = extract_sentence_embeddings(list(shift_dataset.raw_source))
        X1_embeddings = extract_sentence_embeddings(list(shift_dataset.raw_target))
        all_groups = cluster_embeddings(torch.concat([X0_embeddings, X1_embeddings]))
        source_groups = all_groups[:len(X0_embeddings)]
        target_groups = all_groups[len(X0_embeddings):]

    print(shift_dataset.source.shape)
    # if os.path.exists(f"../data/base/{method}/{dataset}_shifts_0.pkl"):
    #     ceds = pickle.load(open(f"../data/base/{method}/{dataset}_shifts_0.pkl", "rb"))
    #     grceds = pickle.load(open(f"../data/base/{method}/{dataset}_shifts_g_0.pkl", "rb"))
    # else:
    for iter in range(3):
        np.random.seed(iter)
        if os.path.exists(f"../data/base/{method}/{dataset}_shifts_lr{params[method].lr}_{iter}.pkl"):
            ceds = pickle.load(open(f"../data/base/{method}/{dataset}_shifts_lr{params[method].lr}_{iter}.pkl", "rb"))
        else:
            out = train_method(dataset, method, shift_dataset, params, use_groups=False)
            for name, val in out.items():
                with open(f"../data/base/{method}/{dataset}_{name}_lr{params[method].lr}_{iter}.pkl", "wb") as f:
                    pickle.dump(val, f)
            ceds = out['shifts']

        if os.path.exists(f"../data/base/{method}/{dataset}_shifts_g_lr{params[method].lr}_{iter}.pkl"):
            grceds = pickle.load(open(f"../data/base/{method}/{dataset}_shifts_g_lr{params[method].lr}_{iter}.pkl", "rb"))
        else:
            out = train_method(dataset, method, shift_dataset, params)
            for name, val in out.items():
                with open(f"../data/base/{method}/{dataset}_{name}_g_lr{params[method].lr}_{iter}.pkl", "wb") as f:
                    pickle.dump(val, f)
            grceds = out['shifts']

    if args.robustness:
        for iter in range(3):
            np.random.seed(iter)
            for r, perturbation in perturb_features(shift_dataset.source, shift_dataset.feature_types, rs=[0.5]):
                print("Norm of perturbation:", np.linalg.norm(perturbation, ord=2))
                print("Norm of source:", np.linalg.norm(shift_dataset.source, ord=2))

                # params[method].lr = 0.01
                shift_dataset_pert = shift_dataset.copy()
                shift_dataset_pert.source = shift_dataset.source + perturbation
                print(shift_dataset.source.dtype, shift_dataset.target.dtype)
                ceds_out = train_method(dataset, method, shift_dataset_pert, params, use_groups=False)
                print("CEDS diff:", np.linalg.norm(ceds_out['shifts'], ord=2) / np.linalg.norm(perturbation, ord=2))
                for name, val in ceds_out.items():
                    with open(f"../data/base/{method}/{dataset}_{name}_perturb_lr{params[method].lr}_{iter}.pkl", "wb") as f:
                        pickle.dump(val, f)

                # assert np.all(shift_dataset_pert.source == shift_dataset.source + perturbation)
                shift_dataset_pert = shift_dataset.copy()
                shift_dataset_pert.source = shift_dataset.source + perturbation
                out = train_method(dataset, method, shift_dataset_pert, params)
                print("GRCEDS diff:", np.linalg.norm(out['shifts'], ord=2) / np.linalg.norm(perturbation, ord=2))
                for name, val in out.items():
                    with open(f"../data/base/{method}/{dataset}_{name}_perturb_g_lr{params[method].lr}_{iter}.pkl", "wb") as f:
                        pickle.dump(val, f)

                with open(f"../data/base/{method}/{dataset}_perturbation_lr{params[method].lr}_{iter}.pkl", "wb") as f:
                    pickle.dump(perturbation, f)

def load_nlp_amazon(args):
    data_dir = Path('../data/nlp')
    dataset = get_dataset(dataset="amazon", download=True, root_dir=str(data_dir))
    # Get the training set
    train_data = dataset.get_subset(
        "train",
        frac=0.02
        # transform=transforms.Compose(
        #     [transforms.Resize((448, 448)), transforms.ToTensor()]
        # ),
    )
    
    dataset = get_dataset(dataset="amazon", download=True, unlabeled=True, root_dir=str(data_dir))
    
    test_data = dataset.get_subset(
        "test_unlabeled",
        frac=0.02
        # transform=transforms.Compose(
        #     [transforms.Resize((448, 448)), transforms.ToTensor()]
        # ),
    )

    print("Initial train and test sizes:", len(train_data), len(test_data))
    
    train_data_arr = np.array(train_data, dtype=object)  # moving to np since indexing can be quirky in pytorch
    test_data_arr = np.array(test_data, dtype=object)  # moving to np since indexing can be quirky in pytorch
    demographics_data = {}
    demographics_groups = {}
    demographics_label = {}
    demo1 = "train"
    demo2 = "test"
    
    if (not args.train) and os.path.exists(os.path.join(data_dir, "amazon", args.featurization)):
        cached_folder = os.path.join(data_dir, "amazon", args.featurization)
        bow_source = load_objs(os.path.join(cached_folder, "bow_source"))
        bow_target = load_objs(os.path.join(cached_folder, "bow_target"))
        feature_names = load_objs(os.path.join(cached_folder, "feature_names"))
        source = load_objs(os.path.join(cached_folder, "source"))
        target = load_objs(os.path.join(cached_folder, "target"))
        source_group = load_objs(os.path.join(cached_folder, "source_group"))
        target_group = load_objs(os.path.join(cached_folder, "target_group"))
        if args.featurization == "lda":
            etm_model = load_objs(os.path.join(cached_folder, "etm_model"))    
        else:
            ngram_vectorizer = load_objs(os.path.join(cached_folder, "ngram_vectorizer"))    
    else:
        # all_labels = torch.tensor(train_data_arr[:,1].tolist())
        # neg_ids = torch.nonzero(all_labels == 0).view(-1)
        # neg_id_ids = torch.randperm(len(neg_ids))#[0:sample_count]
        # neg_ids = neg_ids[neg_id_ids]
        
        demographics_data[demo1] = train_data_arr[:, 0]
        # demographics_groups[demo1] = train_data.metadata_array[:,-2].unsqueeze(1)#[neg_ids, 0:1]
        demographics_groups[demo1] = train_data.metadata_array[:,-2]#[pos_ids, 0:1]
        demographics_groups[demo1] = (demographics_groups[demo1] <= 2).long().unsqueeze(-1)
        demographics_groups[demo1] = torch.cat([demographics_groups[demo1], 1 - demographics_groups[demo1]], dim=1)
        # demographics_label[demo1] = all_labels[neg_ids].numpy()
        
        
        # pos_ids = torch.nonzero(all_labels == 1).view(-1)
        # pos_id_ids = torch.randperm(len(pos_ids))[0:sample_count]
        # pos_ids = pos_ids[pos_id_ids]
        demographics_data[demo2] = test_data_arr[:, 0]
        demographics_groups[demo2] = test_data.metadata_array[:,-2]#[pos_ids, 0:1]
        print(test_data.metadata_fields)
        demographics_groups[demo2] = (demographics_groups[demo2] <= 2).long().unsqueeze(-1)
        demographics_groups[demo2] = torch.cat([demographics_groups[demo2], 1 - demographics_groups[demo2]], dim=1)
        # demographics_label[demo2] = all_labels[pos_ids].numpy()
        cached_folder = os.path.join(data_dir, "amazon", args.featurization)
        os.makedirs(cached_folder, exist_ok=True)
        if args.featurization == "default":
            bow_source, bow_target, feature_names, source, target, source_group, target_group, ngram_vectorizer = get_demographic_counts(demo1, demo2, demographics_data,demographics_groups,
                                                    1, args.max_feat_count, equalize_sizes=True, featurization=args.featurization, lda_epochs=500, lda_lr=0.05)
            save_objs(ngram_vectorizer, os.path.join(cached_folder, "ngram_vectorizer"))    
        elif args.featurization == "lda":
            bow_source, bow_target, feature_names, source, target, source_group, target_group, etm_model = get_demographic_counts(demo1, demo2, demographics_data,demographics_groups,
                                                    1, args.max_feat_count, equalize_sizes=True, featurization=args.featurization, lda_epochs=500, lda_lr=0.05)
            save_objs(etm_model, os.path.join(cached_folder, "etm_model"))    
        
        
        save_objs(bow_source, os.path.join(cached_folder, "bow_source"))
        save_objs(bow_target, os.path.join(cached_folder, "bow_target"))
        save_objs(feature_names, os.path.join(cached_folder, "feature_names"))
        save_objs(source, os.path.join(cached_folder, "source"))
        save_objs(target, os.path.join(cached_folder, "target"))
        save_objs(source_group, os.path.join(cached_folder, "source_group"))
        save_objs(target_group, os.path.join(cached_folder, "target_group"))
        

    id_scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)

    # Determine the groups of transformed samples by the closest target sample
    def group_fn(raw_transformed_source, transformed_source):
        neighs = KNeighborsClassifier(n_neighbors=1).fit(bow_target, np.arange(bow_target.shape[0]))
        return target_group.numpy()[neighs.predict(transformed_source)]

    def embed_fn(raw_samples):
        return extract_sentence_embeddings(raw_samples.tolist()).numpy()
    if args.featurization == "default":
        params = {
            "kmeans": Params(lr=0.5, iters=500, n_clusters=3),
            "ot": Params(lr=0.5, iters=500),
            "dice": Params(lr=0.5, iters=500, n_clusters=20, wreg=5e-5),
        }
        print("source size:", len(source))
        print("target size:", len(target))
        # print size of groups in source and target
        for i in range(source_group.shape[1]):
            print("source group", i, "size:", torch.sum(source_group[:, i]))
        for i in range(target_group.shape[1]):
            print("target group", i, "size:", torch.sum(target_group[:, i]))
        dataset = ShiftDataset(
            source,
            target,
            bow_source,
            bow_target,
            source_group.numpy(),
            target_group.numpy(),
            feature_names,
            ["int"] * bow_source.shape[1],
            id_scaler,
            group_fn,
            CfTextGenerator(source, feature_names),
            embed_fn)
    elif args.featurization== "lda":
        params = {
            "kmeans": Params(lr=2.0, iters=500, n_clusters=3, tol=1e-6),
            "ot": Params(lr=0.02, iters=500, tol=1e-6),
            "dice": Params(lr=0.01, iters=500, n_clusters=20, wreg=5e-5),
        }
        
        dataset = ShiftDataset(
            source,
            target,
            bow_source,
            bow_target,
            source_group.numpy(),
            target_group.numpy(),
            feature_names,
            ["float"] * bow_source.shape[1],
            id_scaler,
            group_fn,
            CfNLPTopicGenerator(feature_names, etm_model),
            embed_fn)
    return dataset, params




def train_adv(dataset_name, method, shift_dataset: ShiftDataset):
    if args.no_groups:
        X0_embeddings = extract_sentence_embeddings(list(shift_dataset.raw_source))
        X1_embeddings = extract_sentence_embeddings(list(shift_dataset.raw_target))
        all_groups = cluster_embeddings(torch.concat([X0_embeddings, X1_embeddings]))
        source_groups = all_groups[:len(X0_embeddings)]
        target_groups = all_groups[len(X0_embeddings):]

    print(shift_dataset.source.shape)

    adv_ceds = 0
    adv_grceds = 0
    source_save = shift_dataset.source.copy()

    if os.path.exists(f"../data/base/{method}/{dataset_name}_shifts.pkl"):
        ceds = pickle.load(open(f"../data/base/{method}/{dataset_name}_shifts.pkl", "rb"))
        grceds = pickle.load(open(f"../data/base/{method}/{dataset_name}_shifts_g.pkl", "rb"))
    else:
        out = train_method(dataset_name, method, shift_dataset, use_groups=False)
        ceds = out['shifts']
        with open(f"../data/base/{method}/{dataset_name}_shifts.pkl", "wb") as f:
            pickle.dump(ceds, f)
        if method == "kmeans":
            init = out['shift']
        elif method == "ot":
            init = out['shifts']
        else:
            init = None

        out = train_method(dataset_name, method, shift_dataset)
        grceds = out['shifts']
        with open(f"../data/base/{method}/{dataset_name}_shifts_g.pkl", "wb") as f:
            pickle.dump(grceds, f)
        if method == "kmeans":
            init_g = out['shift']
        elif method == "ot":
            init_g = out['shifts']
        else:
            init_g = None

    for i in range(3):
        np.random.seed(i)
        for iter in range(25):
            source = source_save.copy()
            for r, perturbation in perturb_features(source, types, rs=[1]):
                print("Norm of perturbation:", np.linalg.norm(perturbation, ord=2))
                print("Norm of source:", np.linalg.norm(source, ord=2))
                shift_dataset_pert = shift_dataset.copy()
                shift_dataset_pert.source = source + perturbation
                out = train_method(dataset_name, method, shift_dataset_pert, use_groups=False)
                ceds_pert = out['shifts']

                out = train_method(dataset_name, method, shift_dataset_pert)
                grceds_pert = out['shifts']

                diff_ceds = np.linalg.norm(ceds - ceds_pert, ord=2) / np.linalg.norm(perturbation, ord=2)
                diff_grceds = np.linalg.norm(grceds - grceds_pert, ord=2) / np.linalg.norm(perturbation, ord=2)
                print("Diff in explanations:", diff_ceds, diff_grceds)
                if diff_ceds != np.nan and diff_ceds > adv_ceds:
                    adv_ceds = diff_ceds
                    with open(f"../data/base/{method}/{dataset_name}_best_ceds_perturb_{i}.pkl", "wb") as f:
                        pickle.dump(perturbation, f)
                if diff_grceds != np.nan and diff_grceds > adv_grceds:
                    adv_grceds = diff_grceds
                    with open(f"../data/base/{method}/{dataset_name}_best_grceds_perturb_{i}.pkl", "wb") as f:
                        pickle.dump(perturbation, f)
                print("Max diff:", adv_ceds, adv_grceds)
        with open(f"../data/base/{method}/{dataset_name}_adv_{i}.pkl", "wb") as f:
            pickle.dump([adv_ceds, adv_grceds], f)

            # with open(f"../data/robustness/{method}/{dataset}_perturbation_{r}_{iter}.pkl", "wb") as f:
            #     pickle.dump(perturbation, f)

def eval_adv(method, dataset):
    all_ceds, all_grceds = [], []
    for i in range(3):
        ceds, grceds = pickle.load(open(f"../data/robustness/{method}/{dataset}_adv_{i}.pkl", "rb"))
        all_ceds.append(ceds)
        all_grceds.append(grceds)
    print("Average CEDS:", np.mean(all_ceds), np.std(all_ceds))
    print("Average GR-CEDS:", np.mean(all_grceds), np.std(all_grceds))

def eval_all(args, dataset, method, shift_dataset: ShiftDataset, params):
    def feasible(orig, new):
        return 100 * (np.sum(np.all(orig == new, axis=1)) / orig.shape[0])

    r = 1
    reg_tot_pe = []
    reg_group_pe = []
    reg_worst_pe = []
    reg_feas = []
    reg_robustness = []
    gse_tot_pe = []
    gse_group_pe = []
    gse_worst_pe = []
    gse_feas = []
    gse_robustness = []
    reg_pert_feas = []
    last_reg_shift = None
    last_gse_shift = None
    for iter in range(3):
        reg_shift = pickle.load(open(f"../data/base/{method}/{dataset}_shifts_lr{params[method].lr}_{iter}.pkl", "rb"))
        gse_shift = pickle.load(open(f"../data/base/{method}/{dataset}_shifts_g_lr{params[method].lr}_{iter}.pkl", "rb"))

        if last_reg_shift is not None:
            print("Norm of reg shift:", np.linalg.norm(reg_shift - last_reg_shift, ord=2))
            print("Norm of gse shift:", np.linalg.norm(gse_shift - last_gse_shift, ord=2))
        last_reg_shift = reg_shift
        last_gse_shift = gse_shift

        # Perform counterfactual modifications
        cf_reg_source = shift_dataset.cf_generator.generate(shift_dataset.raw_source, shift_dataset.source, reg_shift)
        cf_gse_source = shift_dataset.cf_generator.generate(shift_dataset.raw_target, shift_dataset.source, gse_shift)

        reg_new_groups = shift_dataset.group_fn(cf_reg_source, shift_dataset.source + reg_shift)
        gse_new_groups = shift_dataset.group_fn(cf_gse_source, shift_dataset.source + gse_shift)

        source_emb = shift_dataset.embed_fn(shift_dataset.raw_source if shift_dataset.raw_source is not None else shift_dataset.source)
        target_emb = shift_dataset.embed_fn(shift_dataset.raw_target if shift_dataset.raw_target is not None else shift_dataset.target)
        reg_shift_emb = shift_dataset.embed_fn(cf_reg_source)
        gse_shift_emb = shift_dataset.embed_fn(cf_gse_source)
        print("Source shape:", source_emb.shape, "Reg shift shape:", reg_shift_emb.shape, "GSE shift shape:", gse_shift_emb.shape, "Target shape:", target_emb.shape)
        total, worst, group_pes = group_percent_explained(
            source_emb,
            reg_shift_emb,
            target_emb,
            shift_dataset.source_groups,
            shift_dataset.target_groups,
            shift_dataset.feature_names)
        reg_tot_pe.append(total)
        reg_group_pe.append(group_pes)
        reg_worst_pe.append(worst)
        reg_feas.append(feasible(shift_dataset.source_groups, reg_new_groups))

        total, worst, group_pes = group_percent_explained(
            source_emb,
            gse_shift_emb,
            target_emb,
            shift_dataset.source_groups,
            shift_dataset.target_groups,
            shift_dataset.feature_names)
        gse_tot_pe.append(total)
        gse_group_pe.append(group_pes)
        gse_worst_pe.append(worst)
        gse_feas.append(feasible(shift_dataset.source_groups, gse_new_groups))

    if args.robustness:
        reg_shift = pickle.load(open(f"../data/base/{method}/{dataset}_shifts_lr{params[method].lr}_2.pkl", "rb"))
        gse_shift = pickle.load(open(f"../data/base/{method}/{dataset}_shifts_g_lr{params[method].lr}_2.pkl", "rb"))
        iter = 0
        while os.path.exists(f"../data/base/{method}/{dataset}_shifts_perturb_lr{params[method].lr}_{iter}.pkl"):
            base_perturb_exp = pickle.load(open(f"../data/base/{method}/{dataset}_shifts_perturb_lr{params[method].lr}_{iter}.pkl", "rb"))
            group_perturb_exp = pickle.load(open(f"../data/base/{method}/{dataset}_shifts_perturb_g_lr{params[method].lr}_{iter}.pkl", "rb"))
            perturbation = pickle.load(open(f"../data/base/{method}/{dataset}_perturbation_lr{params[method].lr}_{iter}.pkl", "rb"))
            print("Norm of perturbation:", np.linalg.norm(perturbation, ord=2))
            print("Norm of source:", np.linalg.norm(shift_dataset.source, ord=2))

            new_groups = shift_dataset.group_fn(shift_dataset.source + perturbation, shift_dataset.source + perturbation + base_perturb_exp)
            feas = feasible(shift_dataset.source_groups, new_groups)
            reg_pert_feas.append(feas)

            base_delta = np.linalg.norm(reg_shift - base_perturb_exp, ord=2) / np.linalg.norm(perturbation, ord=2)
            group_delta = np.linalg.norm(gse_shift - group_perturb_exp, ord=2) / np.linalg.norm(perturbation, ord=2)
            reg_robustness.append(base_delta)
            gse_robustness.append(group_delta)
            iter += 1

    if args.latex:
        method_name = "$K$-cluster"
        if method == "ot":
            method_name = "OT"
        elif method == "dice":
            method_name = "\dice"

        print(f"{method_name} & {np.mean(reg_tot_pe):.2f}$\pm${np.std(reg_tot_pe):.2f} & {np.mean(reg_worst_pe):.2f}$\pm${np.std(reg_worst_pe):.2f} & {np.mean(reg_feas):.2f}$\pm${np.std(reg_feas):.2f} & {np.mean(reg_robustness):.2f}$\pm${np.std(reg_robustness):.2f} & {np.max(reg_robustness):.2f}\\\\")
        print(f"\\ourmethod{{}} {method_name} & {np.mean(gse_tot_pe):.2f}$\pm${np.std(gse_tot_pe):.2f} & {np.mean(gse_worst_pe):.2f}$\pm${np.std(gse_worst_pe):.2f} & {np.mean(gse_feas):.2f}$\pm${np.std(gse_feas):.2f} & {np.mean(gse_robustness):.2f}$\pm${np.std(gse_robustness):.2f} & {np.max(gse_robustness):.2f}\\\\")
        return
    else:
        print("PE", np.mean(reg_tot_pe), np.std(reg_tot_pe), np.mean(gse_tot_pe), np.std(gse_tot_pe))
        print("WG-PE", np.mean(reg_worst_pe), np.std(reg_worst_pe), np.mean(gse_worst_pe), np.std(gse_worst_pe))
        print("Feas", np.mean(reg_feas), np.std(reg_feas), np.mean(gse_feas), np.std(gse_feas))
        reg_group_pe = np.stack(reg_group_pe)
        gse_group_pe = np.stack(gse_group_pe)
        print("group PE", np.mean(reg_group_pe, axis=0), np.std(reg_group_pe, axis=0), np.mean(gse_group_pe, axis=0), np.std(gse_group_pe, axis=0))
        if args.robustness:
            print("Robustness", np.mean(reg_robustness), np.std(reg_robustness), np.mean(gse_robustness), np.std(gse_robustness))
            print(f"Adv. Robustness ({len(reg_robustness)})", np.max(reg_robustness), np.max(gse_robustness))
            n = np.argmin(reg_feas)
            print(f"Perturbed feasibility:", reg_feas[n], reg_pert_feas[n])


    if dataset == "adult" and method == "kmeans":
        iter = 0
        reg_d = pickle.load(open(f"../data/base/{method}/{dataset}_shift_lr{params[method].lr}_{iter}.pkl", "rb")).round(0)
        gse_d = pickle.load(open(f"../data/base/{method}/{dataset}_shift_g_lr{params[method].lr}_{iter}.pkl", "rb")).round(0)
        reg_shift = reg_shift.round(0)
        gse_shift = gse_shift.round(0)
        sort_features = np.argsort(-np.abs(reg_shift - gse_shift), axis=1)
        sort_samples = np.argsort(-np.abs(reg_shift[:, -3] - gse_shift[:, -3]))
        # sort_samples = np.argsort(-np.abs(reg_shift[:, 0] - gse_shift[:, 0]))
        reg_centroids = pickle.load(open(f"../data/base/{method}/{dataset}_centroids_lr{params[method].lr}_{iter}.pkl", "rb")).round(0)
        gse_centroids = pickle.load(open(f"../data/base/{method}/{dataset}_centroids_g_lr{params[method].lr}_{iter}.pkl", "rb"))
        for j, centroid in enumerate(shift_dataset.scaler.inverse_transform(reg_centroids).round(2)):
            print([f"{shift_dataset.feature_names[sort_features[j][i]]}: {s:.2f}" for i, s in enumerate(centroid[sort_features[j]]) if s != 0])
        print()
        for j, shift in enumerate(shift_dataset.scaler.inverse_transform(reg_shift[sort_samples][:25])):
            print([f"{shift_dataset.feature_names[sort_features[j][i]]}: {s:.2f}" for i, s in enumerate(shift[sort_features[j]].round(2)) if s.round(2) != 0])
        print()
        for j, shift in enumerate(shift_dataset.scaler.inverse_transform(gse_shift[sort_samples][:25])):
            print([f"{shift_dataset.feature_names[sort_features[j][i]]}: {s:.2f}" for i, s in enumerate(shift[sort_features[j]].round(2)) if s.round(2) != 0])
        print()
        for j, shift in enumerate(shift_dataset.scaler.inverse_transform(shift_dataset.source[sort_samples][:25])):
            print([f"{shift_dataset.feature_names[sort_features[j][i]]}: {s:.2f}" for i, s in enumerate(shift[sort_features[j]].round(2)) if s.round(2) != 0])

        print()
        print()
        reg_shift = reg_d
        gse_shift = gse_d
        base_perturb_exp = pickle.load(open(f"../data/base/{method}/{dataset}_shift_perturb_lr{params[method].lr}_{iter}.pkl", "rb")).round(0)
        sort_samples = np.argsort(-(reg_shift[:, -3] - base_perturb_exp[:, -3]))
        print("Original centroids:")
        for j, shift in enumerate(shift_dataset.scaler.inverse_transform(reg_centroids[sort_samples][:5])):
            print([f"{shift_dataset.feature_names[sort_features[j][i]]}: {s:.2f}" for i, s in enumerate(shift[sort_features[j]].round(2)) if s.round(2) != 0])
        print()
        print("Perturbed explanations:")
        for j, shift in enumerate(shift_dataset.scaler.inverse_transform(base_perturb_exp[sort_samples][:5])):
            print([f"{shift_dataset.feature_names[sort_features[j][i]]}: {s:.2f}" for i, s in enumerate(shift[sort_features[j]].round(2)) if s.round(2) != 0])
        print()
        print("Original explanations:")
        for j, shift in enumerate(shift_dataset.scaler.inverse_transform(reg_shift[sort_samples][:5])):
            print([f"{shift_dataset.feature_names[sort_features[j][i]]}: {s:.2f}" for i, s in enumerate(shift[sort_features[j]].round(2)) if s.round(2) != 0])
    if dataset == "imagenet-concepts" and method == "kmeans":
        iter = 2
        concepts = load_objs("../data/imagenet_emb/concepts").numpy()
        reg_shift = pickle.load(open(f"../data/base/{method}/{dataset}_shift_lr{params[method].lr}_{iter}.pkl", "rb"))
        gse_shift = pickle.load(open(f"../data/base/{method}/{dataset}_shift_g_lr{params[method].lr}_{iter}.pkl", "rb"))
        reg_centroids = pickle.load(open(f"../data/base/{method}/{dataset}_centroids_lr{params[method].lr}_{iter}.pkl", "rb"))

        reg_centroids = reg_centroids @ concepts.T
        reg_shift = reg_shift @ concepts.T
        gse_shift = gse_shift @ concepts.T
        centroids_top = np.concatenate([np.argsort(-(reg_centroids), axis=1)[:, :20]], axis=1)
        reg_top = np.concatenate([np.argsort(-(reg_shift), axis=1)[:, :5], np.argsort(-(reg_shift), axis=1)[:, -5:]], axis=1)
        gse_top = np.concatenate([np.argsort(-(gse_shift), axis=1)[:, :5], np.argsort(-(gse_shift), axis=1)[:, -5:]], axis=1)
        # set top elements to 1 and bottom elements to 0 in reg_shift
        reg_centroids = np.zeros_like(reg_centroids)
        reg_centroids[np.arange(reg_centroids.shape[0])[:, None], centroids_top] = 1
        reg_shift = np.zeros_like(reg_shift)
        reg_shift[np.arange(reg_shift.shape[0])[:, None], reg_top[:, :5]] = 1
        reg_shift[np.arange(reg_shift.shape[0])[:, None], reg_top[:, 5:]] = -1
        gse_shift = np.zeros_like(gse_shift)
        gse_shift[np.arange(gse_shift.shape[0])[:, None], gse_top[:, :5]] = 1
        gse_shift[np.arange(gse_shift.shape[0])[:, None], gse_top[:, 5:]] = -1

        exp_sort = np.argsort(-np.abs(np.linalg.norm(reg_shift - gse_shift, axis=1))).flatten()
        sort_features = np.argsort(-np.abs(reg_shift - gse_shift), axis=1)[exp_sort]
        # sort_features_neg = np.argsort((reg_shift - gse_shift), axis=1)[exp_sort]
        feature_names = [f"c{i}" for i in range(len(concepts))]

        for j, shift in enumerate(shift_dataset.scaler.inverse_transform(reg_centroids)[exp_sort]):
            features = shift[sort_features[j]].round(2)
            feature_idx = sort_features[j, :]
            print([f"{feature_names[feature_idx[i]]}: {s:.2f}" for i, s in enumerate(features) if s.round(2) != 0])
        print()
        print("Vanilla Explantions")
        for j, shift in enumerate(shift_dataset.scaler.inverse_transform(reg_shift)[exp_sort]):
            features = shift[sort_features[j]].round(2)
            feature_idx = sort_features[j, :]
            print([f"{feature_names[feature_idx[i]]}: {s:.2f}" for i, s in enumerate(features) if s.round(2) != 0])
        print()
        print("GSE Explantions")
        # sort_features_pos = np.argsort(-(gse_shift), axis=1)[exp_sort]
        # sort_features_neg = np.argsort((gse_shift), axis=1)[exp_sort]
        for j, shift in enumerate(shift_dataset.scaler.inverse_transform(gse_shift)[exp_sort]):
            features = shift[sort_features[j]].round(2)
            feature_idx = sort_features[j, :]
            print([f"{feature_names[feature_idx[i]]}: {s:.2f}" for i, s in enumerate(features) if s.round(2) != 0])


def qualitative_eval(dataset, orig_source, orig_target, source, target, source_groups, target_groups, feature_names):
    # if True or not os.path.exists(f"../data/robustness/kmeans/{dataset}_groups.pkl"):
    #     X0_embeddings = extract_sentence_embeddings(list(orig_source))
    #     X1_embeddings = extract_sentence_embeddings(list(orig_target))
    #     all_groups = cluster_embeddings(torch.concat([X0_embeddings, X1_embeddings]))
    #     source_groups = all_groups[:len(X0_embeddings)]
    #     target_groups = all_groups[len(X0_embeddings):]
    #     with open(f"../data/robustness/kmeans/{dataset}_groups.pkl", "wb") as f:
    #         pickle.dump([source_groups, target_groups], f)
    # else:
    #     source_groups, target_groups = pickle.load(open(f"../data/robustness/kmeans/{dataset}_groups.pkl", "rb"))

    no_group_s = np.ones((source.shape[0], 1))
    no_group_t = np.ones((target.shape[0], 1))

    # perturbation = pickle.load(open(f"../data/robustness/kmeans/{dataset}_best_ceds_perturb.pkl", "rb"))
    # perturbation = [p for p in perturb_features(source, ["int"] * source.shape[1], rs=[0.5])][0][1]
    perturbation = pickle.load(open(f"../data/base/kmeans/{dataset}_perturbation_lr{params['kmeans'].lr}_1.pkl", "rb"))
    print("Norm of perturbation:", np.linalg.norm(perturbation, ord=2))
    print(perturbation.shape)
    # if os.path.exists(f"../data/robustness/kmeans/{dataset}_shifts.pkl"):
        # ceds = pickle.load(open(f"../data/robustness/kmeans/{dataset}_shifts.pkl", "rb")).round(0)
        # grceds = pickle.load(open(f"../data/robustness/kmeans/{dataset}_shifts_g.pkl", "rb")).round(0)
    ceds = pickle.load(open(f"../data/base/kmeans/{dataset}_shifts_lr{params['kmeans'].lr}_1.pkl", "rb"))
    grceds = pickle.load(open(f"../data/base/kmeans/{dataset}_shifts_g_lr{params['kmeans'].lr}_1.pkl", "rb"))

    ceds_shifts = pickle.load(open(f"../data/base/kmeans/{dataset}_shifts_perturb_lr{params['kmeans'].lr}_1.pkl", "rb"))
    ceds_pert = ceds_shifts + source + perturbation
    grceds_shifts = pickle.load(open(f"../data/base/kmeans/{dataset}_shifts_perturb_g_lr{params['kmeans'].lr}_1.pkl", "rb"))
    grceds_pert = grceds_shifts + source + perturbation

    # print(np.linalg.norm(ceds - ceds_pert, ord=2) / np.linalg.norm(perturbation, ord=2))
    # print(np.linalg.norm(grceds - grceds_pert, ord=2) / np.linalg.norm(perturbation, ord=2))

    def is_adult_feasible_change(orig, new):
        orig_sex = (orig[:, 2])[:, np.newaxis]
        new_sex = (new[:, 2])[:, np.newaxis]

        return (
            np.all(new_sex == orig_sex, axis=1) )#&

    def edit_text(sentence, words, diff):
        edits = {w: diff[i] for i, w in enumerate(words)}
        new_sentence = []
        for w, e in edits.items():
            if e > 0:
                new_sentence += [w] * int(e)

        for w in sentence.lower().split():
            if w in edits and edits[w] < 0:
                edits[w] += 1
                continue
            new_sentence.append(w)
        return " ".join(new_sentence)

    def is_nlp_feasible_change(orig_groups, new_groups):
        feas = np.sum(orig_groups == new_groups, axis=1) == orig_groups.shape[1]
        return feas, new_groups

    if dataset == "nlp":
        if True or not os.path.exists(f"../data/robustness/kmeans/{dataset}_new_groups.pkl"):
            neigh = KNeighborsClassifier(n_neighbors=1).fit(X1_embeddings, np.arange(len(X1_embeddings)))
            cf_source_trans = [edit_text(s, feature_names, diff) for s, diff in zip(orig_source, ceds)]
            cf_source_trans_emb = extract_sentence_embeddings(cf_source_trans)
            new_groups = target_groups[neigh.predict(cf_source_trans_emb)]
            with open(f"../data/robustness/kmeans/{dataset}_new_groups.pkl", "wb") as f:
                pickle.dump(new_groups, f)
        else:
            new_groups = pickle.load(open(f"../data/robustness/kmeans/{dataset}_new_groups.pkl", "rb"))

        feasible, new_groups = is_nlp_feasible_change(source_groups, new_groups)
        print("Original Percent feasible shift:", (np.sum(feasible) / feasible.shape[0]) * 100)

        if True or not os.path.exists(f"../data/robustness/kmeans/{dataset}_new_groups_pert.pkl"):
            neigh = KNeighborsClassifier(n_neighbors=1).fit(X1_embeddings, np.arange(len(X1_embeddings)))
            cf_source_trans = [edit_text(s, feature_names, diff) for s, diff in zip(orig_source, perturbation + ceds_shifts)]
            cf_source_trans_emb = extract_sentence_embeddings(cf_source_trans)
            new_groups_pert = target_groups[neigh.predict(cf_source_trans_emb)]
            with open(f"../data/robustness/kmeans/{dataset}_new_groups_pert.pkl", "wb") as f:
                pickle.dump(new_groups_pert, f)
        else:
            new_groups_pert = pickle.load(open(f"../data/robustness/kmeans/{dataset}_new_groups_pert.pkl", "rb"))

        if True or not os.path.exists(f"../data/robustness/kmeans/{dataset}_perturb_groups.pkl"):
            neigh_source = KNeighborsClassifier(n_neighbors=1).fit(X0_embeddings, np.arange(len(X0_embeddings)))
            cf_source_perturb = [edit_text(s, feature_names, diff) for s, diff in zip(orig_source, perturbation)]
            cf_source_perturb_emb = extract_sentence_embeddings(cf_source_perturb)
            perturb_groups = source_groups[neigh_source.predict(cf_source_perturb_emb)]
        else:
            perturb_groups = pickle.load(open(f"../data/robustness/kmeans/{dataset}_perturb_groups.pkl", "rb"))

        feasible_pert, new_groups_pert = is_nlp_feasible_change(perturb_groups, new_groups_pert)
        print("Pert Percent feasible shift:", (np.sum(feasible_pert) / feasible_pert.shape[0]) * 100)
    elif dataset == "adult":
        feasible = is_adult_feasible_change(source, ceds + source)
        print("Original Percent feasible shift:", (np.sum(feasible) / feasible.shape[0]) * 100)
        feasible_pert = is_adult_feasible_change(source + perturbation, ceds_pert)
        print("Pert Percent feasible shift:", (np.sum(feasible_pert) / feasible_pert.shape[0]) * 100)
        print()
        feasible = is_adult_feasible_change(source, grceds + source)
        print("Original Percent feasible shift:", (np.sum(feasible) / feasible.shape[0]) * 100)
        feasible_pert = is_adult_feasible_change(source + perturbation, grceds_pert)
        print("Pert Percent feasible shift:", (np.sum(feasible_pert) / feasible_pert.shape[0]) * 100)

    becomes_infeasible = np.logical_and(feasible, ~feasible_pert)


    # total, worst = group_percent_explained(
    #     source + perturbation,
    #     ceds_pert,
    #     target,
    #     source_groups,
    #     target_groups,
    #     [f"f{i}" for i in range(source_groups.shape[1])])
    # print(total, worst)
    # total, worst = group_percent_explained(
    #     source + perturbation,
    #     grceds_pert,
    #     target,
    #     source_groups,
    #     target_groups,
    #     [f"f{i}" for i in range(source_groups.shape[1])])
    # print(total, worst)

    # Print one sample for each group
    for i in range(source_groups.shape[1]):
        print(f"Group {i}")
        print(source[source_groups[:, i] == 1][0])
        print()

    # diffs = np.abs(ceds_shift - grceds_shift)
    # print(np.unique(ceds, axis=0))
    diffs = np.abs(ceds - ceds_shifts)
    sort_idx = np.argsort(-diffs, axis=1)
    sort_idx0 = np.argsort(-np.sum(np.abs(perturbation[becomes_infeasible]), axis=1)).flatten()
    features = feature_names
    j = 0
    for group, new_group, new_group_pert, feas, s_text, t_text, sample, sample_perturb, shift, shift_orig in np.array(list(zip(
        source_groups[becomes_infeasible], new_groups[becomes_infeasible], new_groups_pert[becomes_infeasible],
        feasible_pert[becomes_infeasible], source[becomes_infeasible], target[becomes_infeasible], source[becomes_infeasible],
        (perturbation)[becomes_infeasible], ceds_shifts[becomes_infeasible], ceds[becomes_infeasible])))[sort_idx0]:

        # if np.sum(np.abs(sample_perturb)) > 1:
        #     continue

        # print([f"{features[i]}: {centroid[i].round(1)}" for i in np.nonzero(centroid.round(1))[0]])
        print(f"Group {np.nonzero(group)[0]} -> {np.nonzero(new_group)[0]} -> {np.nonzero(new_group_pert)[0]}")
        print("Feasible:", feas)
        print("Source:", s_text)
        print("Source features:", [f"{features[i]}: {sample[i].round(1)}" for i in np.nonzero(sample.round(1))[0]])
        print("Orig transformed", [f"{features[i]}: {shift_orig[i].round(1)}" for i in np.nonzero(shift_orig.round(1))[0]])
        print("Pert transformed", [f"{features[i]}: {shift[i].round(1)}" for i in np.nonzero(shift.round(1))[0]])
        print("Perturbed features:", [f"{features[i]}: {sample_perturb[i].round(1)}" for i in np.nonzero(sample_perturb.round(1))[0]])
        # print(f"Sample {j}")
        # for i in sort_idx[j][:20]:
            # print(f"Shift in {features[i]} by {shift[i].round(1)}. (initial: {shift_orig[i].round(1)})")
            # print(f"Group Shift in {features[i]} by {g_shift[i].round(1)}. (initial: {g_shift_orig[i].round(1)})")
        # for i in np.argsort(-np.abs(shift_orig))[:5]:
            # print(f"Orig Shift in {features[i]} by {shift[i].round(1)}. (initial: {shift_orig[i].round(1)})")
        # j += 1
        print()

    print("----- Feasible -----")
    becomes_infeasible = np.argsort(-np.sum(abs(perturbation), axis=1)).flatten()[np.logical_and(feasible, feasible_pert)]
    for groups, feas, s_text, t_text, sample, sample_perturb, shift, shift_orig, g_shift, g_shift_orig in np.array(list(zip(
        source_groups[becomes_infeasible], feasible_pert[becomes_infeasible], source[becomes_infeasible], target[becomes_infeasible], source[becomes_infeasible],
        (source + perturbation)[becomes_infeasible], ceds_shifts[becomes_infeasible], ceds[becomes_infeasible], grceds_shifts[becomes_infeasible], grceds[becomes_infeasible])))[sort_idx0][:5]:
        print("Groups:", groups)
        print("Feasible:", feas)
        print("Source:", s_text)
        print("Source features:", [f"{features[i]}: {sample[i].round(1)}" for i in np.nonzero(sample.round(1))[0]])
        print("Orig transformed", [f"{features[i]}: {shift_orig[i].round(1)}" for i in np.nonzero(shift_orig.round(1))[0]])
        print("Pert transformed", [f"{features[i]}: {shift[i].round(1)}" for i in np.nonzero(shift.round(1))[0]])
        print("Perturbed features:", [f"{features[i]}: {sample_perturb[i].round(1)}" for i in np.nonzero(sample_perturb.round(1))[0]])



    # print()
    # diffs = np.abs(grceds - grceds_shift)
    # sort_idx = np.argsort(-diffs, axis=1)
    # j = 0
    # for centroid, shift, shift_orig in zip(grceds_centroids[:5], grceds_shift[:5], grceds[:5]):
    #     print([f"{features[i]}: {centroid[i].round(1)}" for i in np.nonzero(centroid.round(1))[0]])
    #     for i in sort_idx[j][:5]:
    #         print(f"Shift in {features[i]} by {shift[i].round(1)}. (initial: {shift_orig[i].round(1)} {diffs[j][i]})")
    #     j += 1



def eval(method, dataset, shift_dataset):
    if args.no_groups:
        X0_embeddings = extract_sentence_embeddings(list(shift_dataset.raw_source))
        X1_embeddings = extract_sentence_embeddings(list(shift_dataset.raw_target))
        all_groups = cluster_embeddings(torch.concat([X0_embeddings, X1_embeddings]))
        source_groups = all_groups[:len(X0_embeddings)]
        target_groups = all_groups[len(X0_embeddings):]

    all_base_deltas = []
    base_std = []
    all_group_deltas = []
    group_std = []

    # if method == "kmeans":
    #     centroids = pickle.load(open(f"../data/robustness/{method}/{dataset}_centroids.pkl", "rb"))
    #     centroids_g = pickle.load(open(f"../data/robustness/{method}/{dataset}_centroids_g.pkl", "rb"))
    base_exp = pickle.load(open(f"../data/base/{method}/{dataset}_shifts.pkl", "rb"))
    group_exp = pickle.load(open(f"../data/base/{method}/{dataset}_shifts_g.pkl", "rb"))
    # if method == "kmeans":
    #     source_transformed = transform_samples_kmeans(source, centroids, shifts)
    #     source_transformed_g = transform_samples_kmeans(source, centroids_g, shifts_g)
    # else:
    #     source_transformed = source + shifts
    #     source_transformed_g = source + shifts_g

    for r in [1]:
        base_deltas = []
        group_deltas = []
        for iter in range(3):
            perturbation = pickle.load(open(f"../data/robustness/{method}/{dataset}_perturbation_{r}_{iter}.pkl", "rb")) #[selection]
            # if method == "dice":
            #     perturbation = perturbation[selection]
            print("Norm of perturbation:", np.linalg.norm(perturbation, ord=2))

            # print("Total percent changed:", np.sum((np.sum(perturbation, axis=1) > 0)) / perturbation.shape[0])
            # for gid in range(source_groups.shape[1]):
            #     print(f"Percent changed in group {gid}:",
            #           np.sum((np.sum(perturbation[source_groups[:, gid] == 1], axis=1) > 0)) / np.sum(source_groups[:, gid] == 1))
            # if method == "kmeans":
            #     centroids_perturb = pickle.load(open(f"../data/robustness/{method}/{dataset}_centroids_perturb_{r}_{iter}.pkl", "rb"))
            #     centroids_g_perturb = pickle.load(open(f"../data/robustness/{method}/{dataset}_centroids_g_perturb_{r}_{iter}.pkl", "rb"))

            base_perturb_exp = pickle.load(open(f"../data/robustness/{method}/{dataset}_shifts_perturb_{r}_{iter}.pkl", "rb"))
            group_perturb_exp = pickle.load(open(f"../data/robustness/{method}/{dataset}_shifts_g_perturb_{r}_{iter}.pkl", "rb"))

            # print("shifts", shifts.shape)
            # if method == "kmeans":
            #     perturb_transformed = transform_samples_kmeans(source + perturbation, centroids_perturb, shifts_perturb)
            #     perturb_transformed_g = transform_samples_kmeans(source + perturbation, centroids_g_perturb, shifts_g_perturb)
            # else:
            #     print(source.shape, perturbation.shape)
            #     perturb_transformed = source + perturbation + shifts_perturb
            #     perturb_transformed_g = source + perturbation + shifts_g_perturb

            # base_exp = source - source_transformed
            # base_perturb_exp = source + perturbation - perturb_transformed

            # group_exp = source - source_transformed_g
            # group_perturb_exp = source + perturbation - perturb_transformed_g

            base_delta = np.linalg.norm(base_exp - base_perturb_exp, ord=2) / np.linalg.norm(perturbation, ord=2)
            group_delta = np.linalg.norm(group_exp - group_perturb_exp, ord=2) / np.linalg.norm(perturbation, ord=2)
            print(base_delta, group_delta)
            base_deltas.append(base_delta)
            group_deltas.append(group_delta)

        all_base_deltas.append(np.mean(base_deltas))
        base_std.append(np.std(base_deltas))
        all_group_deltas.append(np.mean(group_deltas))
        group_std.append(np.std(group_deltas))
        print("Base delta:", np.mean(base_deltas), np.std(base_deltas))
        print("Group delta:", np.mean(group_deltas), np.std(group_deltas))
        print()
    print(len(all_base_deltas), len(all_group_deltas))
    out_dict = [{"r": r, "base_delta": base_delta, "base_std": base_std, "group_delta": group_delta, "group_std": group_std} for r, base_delta, base_std, group_delta, group_std in zip([1, 2, 4, 8, 16], all_base_deltas, base_std, all_group_deltas, group_std)]
    with open(f"../data/robustness/{dataset}_{method}_out.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["r", "base_delta", "base_std", "group_delta", "group_std"])
        writer.writeheader()
        for row in out_dict:
            writer.writerow(row)

    print(all_base_deltas)
    print(all_group_deltas)
    # plt.style.use('seaborn-whitegrid')
    # plt.plot([1, 2, 4, 8, 16], all_base_deltas, label=f"{method}")
    # plt.plot([1, 2, 4, 8, 16], all_group_deltas, label=f"WG-{method}")
    # plt.fill_between([1, 2, 4, 8, 16], np.array(all_base_deltas) - np.array(base_std), np.array(all_base_deltas) + np.array(base_std), alpha=0.2)
    # plt.fill_between([1, 2, 4, 8, 16], np.array(all_group_deltas) - np.array(group_std), np.array(all_group_deltas) + np.array(group_std), alpha=0.2)
    # plt.xlabel("r")
    # plt.ylabel("L1 distance")
    # plt.legend()
    # tikzplotlib.save(f"../figures/{dataset}_{method}_perturb.tex")

def get_perturbation(source):
    perturbation = np.random.normal(0, 1, source.shape)
    perturbation /= np.linalg.norm(perturbation, axis=1, keepdims=True, ord=1) / 2
    return perturbation



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # a flag argument
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--method')
    parser.add_argument('--robustness', action='store_true')
    parser.add_argument('--adv', action='store_true')
    parser.add_argument('--dataset')
    parser.add_argument('--featurization', type=str, default="default", choices=["default", "lda", "emb"])
    parser.add_argument('--max_feat_count', type=int, default=100)
    parser.add_argument('--qual', action='store_true')
    parser.add_argument('--no_groups', action='store_true')
    parser.add_argument('--latex', action='store_true')
    args = parser.parse_args()

    if args.dataset == "adult":
        shift_dataset, params = load_adult()
    elif args.dataset == "breast":
        shift_dataset, params = load_breast()
    elif args.dataset == "imagenet":
        shift_dataset, params = load_imagenet()
    elif args.dataset == "imagenet-concepts":
        shift_dataset, params = load_imagenet_concepts()
    elif args.dataset == "nlp":
        shift_dataset, params = load_nlp(args)
    elif args.dataset == "nlp-emb":
        shift_dataset, params = load_nlp_emb(args)
    elif args.dataset == "nlp-amazon":
        shift_dataset, params = load_nlp_amazon(args)
    elif args.dataset == "imagenet-pixels":
        shift_dataset, params = load_breeds_rawpixels()
    elif args.dataset == "iwildcam":
        shift_dataset, params = load_iwildcam()
    elif args.dataset == "fmow":
        shift_dataset, params = load_fmow()
    else:
        raise NotImplementedError

    if args.method != "dice":
        sample_size = 1000
        np.random.seed(0)
        torch.manual_seed(0)
        subset = np.random.permutation(shift_dataset.source.shape[0])[:sample_size]
    else:
        subset = np.arange(shift_dataset.source.shape[0])

    if args.train:
        Path(f"../data/base/{args.method}").mkdir(parents=True, exist_ok=True)

        train(args.dataset, args.method, shift_dataset.subset(subset), params)
    elif args.qual:
        shift_dataset = shift_dataset.subset(subset)
        qualitative_eval(args.dataset, shift_dataset.raw_source, shift_dataset.raw_target,
                         shift_dataset.source, shift_dataset.target, shift_dataset.source_groups,
                         shift_dataset.target_groups, shift_dataset.feature_names)
    else:
        if args.method == "dice":
            np.random.seed(42)
            subsubset = np.random.choice(np.arange(subset.shape[0]), size=min(300, subset.shape[0]), replace=False)
            subset = subset[subsubset]
        else:
            subsubset = subset
        # else:
        #     sample_size = -1
        #     np.random.seed(0)
        #     subset = np.random.permutation(source.shape[0])[:sample_size]

        print(subset.shape)
        eval_all(args, args.dataset, args.method, shift_dataset.subset(subset), params)
