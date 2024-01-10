from langchain.document_loaders import ConfluenceLoader

import pathlib
import sys

_parentdir = pathlib.Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_parentdir))
print (_parentdir)

from scripts.config import Config

cfg = Config()

loader = ConfluenceLoader(
    url=cfg.jira_site, username=cfg.jira_user, api_key=cfg.jira_api_key
)

# Setear la key del espacio de trabajo de confluence en space_key
# limit es la cantidad de documentos a cargar consulta que hará loader, no el total de documentos a traer.
docs = loader.load(space_key="EINV", include_attachments=False, limit=50,)

print(docs[0].page_content[:500])