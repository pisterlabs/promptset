from multi_type_search.search.search_model.types.contrastive import ContrastiveModel
from multi_type_search.utils.paths import ROOT_FOLDER

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import RobertaTokenizer, RobertaModel, AutoModel, AutoConfig, AutoTokenizer

import pickle
import openai
import json
import torch
from torch import nn
from torch.nn import functional as F
import transformers
from pathlib import Path
from typing import List, Union, Dict, Tuple, Optional


def load_api_key():
    key_file: Path = ROOT_FOLDER / 'openai_key.txt'
    with open(key_file, 'r') as f:
        return f.read().strip()


def load_embeddings_from_file(file_name):
    with open(file_name, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings


class RawGPT3Encoder(ContrastiveModel):
    # TODO - support on the fly embeddings, right now everything must be cached.
    model_type: str = 'raw_gpt3_encoder'

    def __init__(
            self,
            cached_embeddings_file: str = None,
            cached_strings_file: str = None,
            allow_api_access: bool = True,
            api_end_point: str = "text-embedding-ada-002"
    ):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if cached_embeddings_file is not None:
            self.cached_embeddings_file = str(Path(ROOT_FOLDER / cached_embeddings_file).absolute())
            self.cached_strings_file = str(Path(ROOT_FOLDER / cached_strings_file).absolute())

            self.str_cache = json.load(Path(ROOT_FOLDER / cached_strings_file).open('r'))

            emb_cache = load_embeddings_from_file(self.cached_embeddings_file)
            emb_cache = torch.stack([torch.tensor(x).to(torch.float32) for x in emb_cache], dim=0).to(self.device)
            self.register_buffer("emb_cache", emb_cache)
        else:
            self.cached_embeddings_file = None
            self.cached_strings_file = None
            self.str_cache = []
            self.register_buffer("emb_cache", torch.tensor([], dtype=torch.float32).to(self.device))

            # TODO - bad... brought this over for training script to run, but there is no training in the raw variant.
        self.tmp = nn.Linear(10, 10)
        self.roberta_tokenizer = self.__tokenizer__

        self.api_end_point = api_end_point

        self.allow_api_access = allow_api_access
        if self.allow_api_access:
            key = load_api_key()
            openai.api_key = key

    def activate_key(self):
        self.allow_api_access = True
        key = load_api_key()
        openai.api_key = key

    def deactivate_key(self):
        self.allow_api_access = False
        openai.api_key = None

    def __tokenizer__(self, string, *args, **kwargs):
        return string

    def get_kwargs(self):
        return {
                'cached_embeddings_file': self.cached_embeddings_file,
                'cached_strings_file': self.cached_strings_file,
                'allow_api_access': self.allow_api_access,
                'api_end_point': self.api_end_point
            }

    def tokenize(self, exs: Union[List[str], str]):
        return exs

    def forward(self, tokens: Union[torch.Tensor, List[str]]):
        embs = []
        for string in tokens:
            try:
                idx = self.str_cache.index(string)
                embs.append(self.emb_cache[idx])
            except ValueError:
                if self.allow_api_access:
                    embs.append(self.call_api(string))
                else:
                    print("ERROR")

        # print('===')
        # print(len(tokens))
        # print(len(embs))
        return torch.stack(embs, dim=0).to(self.device)

    def call_api(self, text: str):
        with torch.no_grad():
            res = openai.Embedding.create(model=self.api_end_point, input=text)
            emb = torch.tensor(res['data'][0]['embedding']).to(self.device).requires_grad_(False)
            self.str_cache.append(text)
            self.emb_cache = torch.cat([self.emb_cache, emb.unsqueeze(dim=0)], 0)
        return emb

    def get_encodings(self, strings: List[str]) -> torch.Tensor:
        return self(strings)

    @classmethod
    def __load__(cls, data: Dict, device: str, opt) -> 'RawGPT3Encoder':
        kwargs = data.get('kwargs')
        assert kwargs is not None, f'Error loading node embedder from checkpoint: {ckpt}, no kwargs in file.'

        model = cls(**kwargs)
        model.activate_key()

        if opt:
            return model, opt
        return model


class GPT3LinearLayer(nn.Module):

    def __init__(
            self,
            input_dim: int = 1536,
            output_dim: int = 1536,
            normalization: Optional[nn.Module] = nn.LayerNorm(1536),
            activation: Optional[nn.Module] = nn.ReLU(),
            dropout: Optional[float] = 0.0
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer = nn.Linear(input_dim, output_dim)
        self.normalization = normalization
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        seq = [self.layer]
        if normalization:
            seq.append(self.normalization)
        if activation:
            seq.append(self.activation)
        if dropout is not None:
            seq.append(self.dropout)

        self.projection = nn.Sequential(*seq)

    def forward(self, x):
        return self.projection(x)


class GPT3GluLayer(nn.Module):
    def __init__(
            self,
            input_dim: int = 1536,
            output_dim: int = 1536,
            residual_connection: bool = True,
            normalization: Optional[nn.Module] = nn.LayerNorm(1536),
            dropout: Optional[float] = 0.0
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual_connection = residual_connection

        self.true_output_dim = output_dim + int(output_dim / 2) if self.residual_connection else int(output_dim / 2)
        self.layer = nn.Linear(input_dim, output_dim)
        self.normalization = normalization
        self.activation = nn.GLU(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer(x)
        if self.normalization:
            x = self.normalization(x)
        activations = self.activation(x)
        if self.residual_connection:
            activations = torch.cat([x, activations], dim=1)
        if self.dropout:
            activations = self.dropout(activations)
        return activations


class ProjectedGPT3Encoder(RawGPT3Encoder):
    # TODO - support on the fly embeddings, right now everything must be cached.
    model_type: str = 'projected_gpt3_encoder'

    def __init__(
            self,
            cached_embeddings_file: str = None,
            cached_strings_file: str = None,
            allow_api_access: bool = True,
            api_end_point: str = "text-embedding-ada-002",
            projection_head_layer_num: int = 3,
            projection_head_type: str = 'glu'
    ):
        super().__init__(
            cached_embeddings_file=cached_embeddings_file,
            cached_strings_file=cached_strings_file,
            allow_api_access=allow_api_access,
            api_end_point=api_end_point
        )
        self.projection_head_layer_num = projection_head_layer_num
        self.emb_size = 1536
        self.projection_head_type = projection_head_type

        projection = self.__setup_glu_projection__() if self.projection_head_type == 'glu' else \
            self.__setup_linear_projection__()

        self.projection = nn.Sequential(*projection)


        # TODO - bad... brought this over for training script to run, but there is no training in the raw variant.
        self.roberta_tokenizer = self.__tokenizer__

    def __setup_linear_projection__(self) -> List[nn.Module]:

        projection = []
        for i in range(self.projection_head_layer_num):
            if i < (self.projection_head_layer_num - 1):
                projection.append(GPT3LinearLayer(
                    input_dim=self.emb_size, output_dim=self.emb_size,
                    normalization=nn.LayerNorm(self.emb_size), activation=nn.LeakyReLU(),
                    dropout=0.0
                ))
            else:
                projection.append(GPT3LinearLayer(
                    input_dim=self.emb_size, output_dim=self.emb_size,
                    normalization=None, activation=None,
                    dropout=0.0
                ))
        return projection

    def __setup_glu_projection__(self) -> List[nn.Module]:

        projection = []
        for i in range(self.projection_head_layer_num):
            if i < (self.projection_head_layer_num - 1):
                projection.append(GPT3GluLayer(
                    input_dim=self.emb_size if len(projection) == 0 else projection[-1].true_output_dim,
                    output_dim=self.emb_size, residual_connection=True,
                    normalization=nn.LayerNorm(self.emb_size),
                    dropout=0.0
                ))
            else:
                projection.append(GPT3GluLayer(
                    input_dim=self.emb_size if len(projection) == 0 else projection[-1].true_output_dim,
                    output_dim=self.emb_size,
                    residual_connection=True,
                    normalization=None,
                    dropout=0.0
                ))
        return projection

    def __tokenizer__(self, string, *args, **kwargs):
        return string

    def get_kwargs(self):
        kwargs = super().get_kwargs()
        kwargs.update({
            'projection_head_layer_num': self.projection_head_layer_num,
            'projection_head_type': self.projection_head_type
        })
        return kwargs

    def tokenize(self, exs: Union[List[str], str]):
        return exs

    def forward(self, tokens: Union[torch.Tensor, List[str]]):
        embs = super().forward(tokens)
        return self.projection(embs)

    def get_encodings(self, strings: List[str]) -> torch.Tensor:
        return self(strings)

    @classmethod
    def __load__(cls, data: Dict, device: str, opt) -> 'ProjectedGPT3Encoder':
        kwargs = data.get('kwargs')
        assert kwargs is not None, f'Error loading node embedder from checkpoint: {ckpt}, no kwargs in file.'

        if 'opt_state' in data and opt:
            opt.load_state_dict(data['opt_state'])

        model = cls(**kwargs)

        state_dict = data.get('state_dict')
        assert state_dict is not None, f'Error loading node embedder from checkpoint: {ckpt}, no state dict in file.'
        model.load_state_dict(state_dict, strict=False)

        model.to(device)

        if opt:
            return model, opt
        return model
