from typing import Any, List, Union
import openai 

from dataclasses import dataclass


class ModelPermission:
    id : str
    object : str
    created : int
    allow_create_engine : bool
    allow_sampling: bool
    allow_logprobs: bool
    allow_search_indices: bool
    allow_view: bool
    allow_fine_tuning: bool
    organization: str
    group: None
    is_blocking: bool

@dataclass
class Model:
    id : str
    object : str
    created : int
    owned_by : str
    permissions : List[ModelPermission]




