import inspect
from pathlib import Path
import re

import openai

path = Path(inspect.getfile(openai.api_requestor.APIRequestor))
with Path(path.name).open("r") as f:
    contents = f.read()

with path.open("w") as f:
    f.write(contents)
