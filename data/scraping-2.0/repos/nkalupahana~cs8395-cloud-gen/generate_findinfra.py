from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

import argparse
import glob
import itertools
import json
import os.path
import sys

from llm_test_helpers import get_llm, get_args
args = get_args(sys.argv)
llm = get_llm(args.model)

# Test generation prompt
infrafind_prompt = PromptTemplate.from_template("""Identify the AWS infrastructure that is being used in the following code. Be sure to include the name of the infrastructure in addition to the name of the service being used on the same line. Do not include local files or region information.

Code:
```
import boto3

s3 = boto3.client('s3')
s3.download_file('mybucket', 'hello.txt', 'hello.txt')
```

Output:
```
- S3 bucket named mybucket
```

Code:
```
{code}
```
                                                
Output:
```""")

for function in itertools.chain(glob.glob("scripts/*.py")):
    if "__init__" in function:
        continue

    filename = function.replace("scripts/", "").replace(".py", "")
    if os.path.isfile(f"infra/{filename}.txt"):
        print(f"Skipping {filename}, infra already exists.")
        continue

    # Get code and create prompt
    print(f"Generating infrastructure description for {function}.")
    with open(function) as f:
        code = f.read()
    code_prompt = infrafind_prompt.format(code=code)
    
    # Generate tests and clean up output
    output = llm.predict(code_prompt)
    output = output.replace("```md", "").replace("```", "").strip()

    # Write out final code
    with open(f"infra/{filename}.txt", "w") as f:
        f.write(output)