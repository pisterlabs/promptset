from typing import Any, Dict, List

import langchain.schema as schema
from services.world.internal.models.event import Event


class Character(Event):
    name: str

    fact: List[schema.Document] = []
    memory: List[schema.Document] = []
    property: Dict[str, Any] = {}
