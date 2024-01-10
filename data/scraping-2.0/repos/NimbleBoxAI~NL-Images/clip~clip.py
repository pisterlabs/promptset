# from OpenAI CLIP sourcecode: https://github.com/openai/CLIP/blob/main/clip/clip.py
# released under: MIT License
# Modified by NimbleBox.ai

# files with a bunch of helper functions
from .daily import *

import os
import subprocess

import urllib
from tqdm import tqdm

from PIL import Image
import torch

import numpy as np
import pickle

from clip.model import build_model
from clip.tokenizer import SimpleTokenizer
from clip import utils

# fixed
_MODELS = {
  "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
  "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
  "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
  "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
}
_VOCAB_PATH = 'https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz'


# download function
def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
  os.makedirs(root, exist_ok=True)
  filename = os.path.basename(url)
  download_target = os.path.join(root, filename)

  if os.path.exists(download_target) and not os.path.isfile(download_target):
    raise RuntimeError(f"{download_target} exists and is not a regular file")

  if os.path.isfile(download_target):
      return download_target

  with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
    with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
      while True:
        buffer = source.read(8192)
        if not buffer:
          break

        output.write(buffer)
        loop.update(len(buffer))

  return download_target

# functions
class CLIP:
  def __init__(
    self,
    image_model = "RN50",
    model_cache_folder = ".cache_model",
    image_cache_folder = ".cache_images",
    jit = False,
  ):
    """CLIP model wrapper.

    Args:
      image_model (str, optional): Model name, one of `"RN50", "RN101", "RN50x4", "ViT-B/32"`.
        Defaults to "RN50".
      model_cache_folder (str, optional): folder with weights and vocab file.
        Defaults to ".cache_model".
      image_cache_folder (str, optional): folder with `latents.npy` and `image_keys.p`.
        Defaults to ".cache_images".
      jit (bool, optional): [BUG] to load the model as jit script.
        Defaults to False.
    """

    # note that this is a simple wget call to get the files, this not take into
    # account the corruption of file that can happen during the transfer by checksum
    # if you find any issue during download we recommend looking at the source code:
    # https://github.com/openai/CLIP/blob/main/clip/clip.py
    mcache = os.path.join(folder(__file__), model_cache_folder)

    if image_model in _MODELS:
        model_path = _download(_MODELS[image_model], mcache)
    elif os.path.isfile(image_model):
        model_path = image_model
    else:
        raise RuntimeError(f"Model {image_model} not found; available models = {_MODELS.keys()}")

    vocab_path = _download(_VOCAB_PATH, mcache)

    # the model was saved with CUDA and so it cannot be loaded directly on CPU
    # note: this is why we are using self.device tag for this. Moreover when loading
    # the model, if this has JIT, then there is a huge problem where the subroutines
    # still contains the torchscript code:
    # ```
    # _0 = self.visual input = torch.to(image, torch.device("cuda:0"), 5, False, False, None)
    # ```
    # note the torch.device("cuda:0"), under such a situation we will need to manually
    # build the model using build_model() method.
    self.device = torch.device("cuda:0") if (torch.cuda.is_available() and jit) else "cpu"
    model = torch.jit.load(model_path, map_location = self.device).eval()
    if not jit:
      self.model = build_model(model.state_dict()).to(self.device)
    else:
      self.model = model
    self.input_resolution = self.model.image_resolution
    self.context_length = self.model.context_length
    self.tokenizer = SimpleTokenizer(vocab_path)

    # now we check if there already exists a cache folder and if there is we will load
    # the embeddings for images as well
    icache = os.path.join(folder(__file__), image_cache_folder)
    os.makedirs(icache, exist_ok=True)
    emb_path = os.path.join(icache, "latents.npy")
    f_keys = os.path.join(icache, "image_keys.p")
    if not os.path.exists(emb_path):
      print("Embeddings path not found, upload images to create embeddings")
      emb = None
      keys = {}
    else:
      emb = np.load(emb_path)
      with open(f_keys, "rb") as f:
        keys = pickle.load(f)
      print("Loaded", emb.shape, "embeddings")
      print("Loaded", len(keys), "keys")

    self.icache = icache
    self.emb_path = emb_path
    self.f_keys = f_keys
    self.emb = emb # emb mat
    self.keys = keys # hash: idx
    self.idx_keys = {v:k for k, v in keys.items()}


  @property
  def n_images(self):
    return self.emb.shape[0] if self.emb is not None else 0


  def upate_emb(self, all_i: list, all_h: list, all_emb: list):
    """Update the embeddings, keys and cache images

    Args:
        all_i (list): list of all opened images
        all_h (list): list of all hashes for corresponding all_i[j]
        all_emb (list): list of embeddings for corresponding all_i[j]
    """
    # update the keys
    self.keys.update({k:i+len(self.keys) for i,k in enumerate(all_h)})
    self.idx_keys = {v:k for k, v in self.keys.items()}

    # cache the images -> copy from source (i) to target (t)
    for _hash, img in zip(all_h, all_i):
      # i is `UploadedFile` object thus i.name
      t = os.path.join(self.icache, _hash + ".png")
      img.save(t)

    # update the embeddings
    if self.emb is not None:
      self.emb = np.vstack([self.emb, *all_emb])
    else:
      self.emb = np.vstack(all_emb)

    # update the cached files
    with open(self.f_keys, "wb") as f:
      pickle.dump(self.keys, f)
    np.save(self.emb_path, self.emb)

  @torch.no_grad()
  def text_to_text_similarity(self, memory: list, query: str, n: int = 10):
    """Text to text similarity for comparing input query to memory.

    Args:
      memory (list): list of strings for memory
      query (str): query string
      n (int, optional): number of items to return. Defaults to 10.

    Returns:
      (list): return list of matching strings from memory
    """
    # first all the sequences
    all_sent = memory + [query]
    all_sent = self.tokenizer(all_sent, self.context_length, self.device)
    feat = self.model.encode_text(all_sent)
    feat /= feat.norm(dim=-1, keepdim=True).cpu()
    feat = feat.numpy()

    # next process the query
    memory_emb, query = feat[:-1], feat[-1]
    out_idx = np.argsort(query @ memory_emb.T)[::-1][:n]
    matches = [memory[i] for i in out_idx]

    return matches


  @torch.no_grad()
  def text_to_image_similarity(self, images: list, text: list, transpose_flag: bool):
    """This is the implementation of n-text to m-images similarity checking
    just like how CLIP was intended to be used.

    Args:
      images (list): list of image files
      text (list): list of text strings
      transpose_flag (bool): image first or text first priority boolean

    Returns:
      (plt.figure): heatmap for the similarity scores
    """
    hashes = [Hashlib.sha256(Image.open(x).tobytes()) for x in images]
    cached_emb = []; to_proc = []; to_proc_hash = []
    for i,h in zip(images, hashes):
      if h in self.keys:
        cached_emb.append(self.emb[i])
      else:
        to_proc.append(Image.open(i))
        to_proc_hash.append(h)

    # tokenize text and tensorize images and get features for each
    input_images = utils.prepare_images(to_proc, self.input_resolution, self.device)
    input_text = self.tokenizer(text, self.context_length, self.device)
    image_features = self.model.encode_image(input_images)
    text_features = self.model.encode_text(input_text)

    # normalise the features, add the cached ones and get similarity matrix
    text_features /= text_features.norm(dim=-1, keepdim=True).cpu().numpy()
    image_features /= image_features.norm(dim=-1, keepdim=True).cpu().numpy()
    stacked_features = np.vstack([image_features, *cached_emb])
    if transpose_flag:
      similarity_matrix = stacked_features @ text_features.T
    else:
      similarity_matrix = text_features @ stacked_features.T
    result = (100.0 * similarity_matrix).softmax(dim=-1)

    # get the final heatmap image and update the embeddings
    output = utils.get_similarity_heatmap(result, images, text, transpose_flag)
    self.upate_emb(to_proc, to_proc_hash, image_features)
    return output


  @torch.no_grad()
  def text_search(self, text: str, n: int) -> list:
    """search through images based on the input text

    Args:
      text (str): text string for searching
      n (int): number of results to return

    Returns:
      images (list): return list of iamge
    """
    # get the text features
    input_tokens = self.tokenizer(text, self.context_length, self.device)
    text_features = self.model.encode_text(input_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.numpy()

    # score using dot-product and load the images requires
    # note that we shift the selection by 1 because 0 is the image itself
    img_idx = np.argsort(text_features @ self.emb.T)[0][::-1][:n]
    hash_idx = [self.idx_keys[x] for x in img_idx]
    images = []
    for x in hash_idx:
      fp = os.path.join(self.icache, f"{x}.png")
      images.append(Image.open(fp))
    return images


  @torch.no_grad()
  def visual_search(self, image: str, n: int) -> list:
    """CLIP.visual encoder model gives out embeddings that can be used for visual
    similarity. Note: this does not return the perfect similarity but visual similarity.

    Args:
      image (str): path to input image
      n (int): number of results to return

    Returns:
      images (list): returns list of images
    """
    # load image and get the embeddings
    image = Image.open(image)
    _hash = Hashlib.sha256(image.tobytes())
    if _hash not in self.keys:
      out = utils.prepare_images([image], self.input_resolution, self.device)
      out = self.model.encode_image(out).cpu()
      out /= out.norm(dim=-1, keepdim=True)
      out = out.numpy()
      # this looks like a new image, store it
      self.upate_emb([image], [_hash], [out])
      shift = 0
    else:
      out = self.emb[self.keys[_hash]].reshape(1, -1)
      shift = 1

    # score using dot-product and load the images requires
    # note that we shift the selection by 1 because 0 is the image itself
    img_idx = np.argsort(out @ self.emb.T)[0][::-1][shift:n+shift]
    hash_idx = [self.idx_keys[x] for x in img_idx]
    images = []
    for x in hash_idx:
      fp = os.path.join(self.icache, f"{x}.png")
      images.append(Image.open(fp))
    return images


  @torch.no_grad()
  def upload_images(self, images: list) -> list:
    """uploading simply means processing all these images and creating embeddings
    from this that can then be saved in the

    Args:
      images (list): image to cache

    Returns:
      list: hash objects of all the files
    """
    # get the hashes for new files only
    hashes = [Hashlib.sha256(Image.open(x).tobytes()) for x in images]
    all_i = []; all_h = []
    for i,h in zip(images,hashes):
      if h in self.keys:
        continue
      all_i.append(i); all_h.append(h)

    # get the tensors after processing
    opened_images = [Image.open(i) for i in all_i]
    out = utils.prepare_images(opened_images, self.input_resolution, self.device)
    out = self.model.encode_image(out)
    out /= out.norm(dim=-1, keepdim=True)
    out = out.numpy()

    # update and store the new images and hashes
    self.upate_emb(opened_images, all_h, out)

    return all_h