from pygbif import registry
from models import DatasetSearchResults, Result, Facet, Count
import json
from langchain.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=DatasetSearchResults)

out = parser.parse(json.dumps(registry.dataset_search(type='OCCURRENCE')))
print(out)