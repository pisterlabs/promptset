# This script is based on https://github.com/langchain-ai/langchain/blob/master/libs/experimental/langchain_experimental/open_clip/open_clip.py
from typing import Any, List, Union, Optional, Literal
import ftfy, html, re
import torch
from langchain.pydantic_v1 import BaseModel
from langchain.schema.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, BatchFeature

DEFAULT_MODEL_NAME = "stabilityai/japanese-stable-clip-vit-l-16"

# taken from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/tokenizer.py#L65C8-L65C8
def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def tokenize(
    tokenizer,
    texts: Union[str, List[str]],
    max_seq_len: int = 77,
):
    """
    This is a function that have the original clip's code has.
    https://github.com/openai/CLIP/blob/main/clip/clip.py#L195
    """
    if isinstance(texts, str):
        texts = [texts]
    texts = [whitespace_clean(basic_clean(text)) for text in texts]

    inputs = tokenizer(
        texts,
        max_length=max_seq_len - 1,
        padding="max_length",
        truncation=True,
        add_special_tokens=False,
    )
    # add bos token at first place
    input_ids = [[tokenizer.bos_token_id] + ids for ids in inputs["input_ids"]]
    attention_mask = [[1] + am for am in inputs["attention_mask"]]
    position_ids = [list(range(0, len(input_ids[0])))] * len(texts)

    return BatchFeature(
        {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
        }
    )

class JapaneseCLIPEmbeddings(BaseModel, Embeddings):
    """Japanese Stable CLIP embedding models.

    To use, you should have the ``transformers`` python package installed.

    Example:
        .. code-block:: python

            from japanese_clip import JapaneseCLIPEmbeddings

            model_name = "stabilityai/japanese-stable-clip-vit-l-16"
            device_map = "cuda"
            embeddings = JapaneseCLIPEmbeddings(
                model_name=model_name,
                device_map=device_map,
            )
    """
    
    model: Any
    device: Any
    preprocess: Any
    tokenizer: Any
    model_name: str = DEFAULT_MODEL_NAME
    """Model name to use."""
    cache_folder: Optional[str] = None
    """Path to store models. 
    Can be also set by HF_HOME environment variable."""
    device_map: Optional[str] = None
    """Keyword arguments to pass to the model."""
    multi_process: bool = False
    """Run encode() on multiple GPUs."""
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if self.device_map is None:
            self.device_map = "cuda" if torch.cuda.is_available() \
                else "mps"  if torch.backends.mps.is_available() \
                else "cpu"
        self.model = AutoModel.from_pretrained(
            self.model_name, 
            trust_remote_code=True, 
            device_map=self.device_map,
            cache_dir=self.cache_folder,
        )
        self.device = self.model.device
        preprocess = AutoImageProcessor.from_pretrained(self.model_name, cache_dir=self.cache_folder)
        self.preprocess = lambda image: preprocess(image, return_tensors="pt").to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_folder)
        self.tokenizer = lambda text: tokenize(tokenizer, text).to(self.device)
    
    def embed_query(self, text: str, query_type: Literal["text", "image"]="text") -> List[float]:
        if query_type == "text":
            return self.embed_documents([text])[0]
        elif query_type == "image":
            return self.embed_image([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        tokenized_texts = self.tokenizer(texts)

        # Encode the text to get the embeddings
        with torch.no_grad():
            embeddings_tensor = self.model.get_text_features(**tokenized_texts)

            # Normalize the embeddings
            norm = embeddings_tensor.norm(p=2, dim=1, keepdim=True)
            normalized_embeddings_tensor = embeddings_tensor.div(norm)

            # Convert normalized tensor to list and add to the text_features list
            text_features = normalized_embeddings_tensor.tolist()

        return text_features

    def embed_image(self, uris: List[str]) -> List[List[float]]:
        try:
            from PIL import Image as _PILImage
        except ImportError:
            raise ImportError("Please install the PIL library: pip install pillow")

        # Open images directly as PIL images
        pil_images = [_PILImage.open(uri) for uri in uris]

        image_features = []
        for pil_image in pil_images:
            # Preprocess the image for the model
            preprocessed_image = self.preprocess(pil_image)

            # Encode the image to get the embeddings
            with torch.no_grad():
                embeddings_tensor = self.model.get_image_features(**preprocessed_image)

                # Normalize the embeddings tensor
                norm = embeddings_tensor.norm(p=2, dim=1, keepdim=True)
                normalized_embeddings_tensor = embeddings_tensor.div(norm)

                # Convert tensor to list and add to the image_features list
                embeddings_list = normalized_embeddings_tensor.squeeze(0).tolist()

                image_features.append(embeddings_list)

        return image_features