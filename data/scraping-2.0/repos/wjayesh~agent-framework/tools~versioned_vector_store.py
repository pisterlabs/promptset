from typing import List
from langchain.tools import VectorStoreQATool

from policies.base_unknown_policy import UnknownPolicy
from policies.ignore import IgnorePolicy

class VersionedVectorStoreTool(VectorStoreQATool):
    urls: List[str]
    version: str
    # TODO should this be a UnknownPolicy instead?
    # that way we can pass in params easily, for example
    # slack bot info, etc.
    # OTOH, if we keep it this way, people would first
    # initialize the policy -> lot of boilerplate code
    unknown_policy: UnknownPolicy = IgnorePolicy()