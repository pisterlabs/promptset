from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

import argparse
import glob
import itertools
import json
import os.path
import sys
from tqdm import tqdm

from llm_test_helpers import get_llm, get_args
args = get_args(sys.argv)
llm = get_llm(args.model)

stack_prompt = PromptTemplate.from_template("""Create an AWS CDK Stack class in TypeScript named CdkStack that creates an AWS Lambda function named test_function. The file to be deployed for the Lambda is a local file called test.zip. Additionally, create the following resources:
{resources}
Ensure that the Lambda function has access to the resources by importing and using IAM.
```""")

for function in tqdm(itertools.chain(glob.glob("infra/*.txt"))):
    if "__init__" in function:
        continue

    filename = function.replace("infra/", "").replace(".txt", "")

    # Get code and create prompt
    with open(function) as f:
        code = f.read()

    if os.path.isfile(f"lambda_stacks/{filename}.ts"):
        print(f"Skipping {filename}, test already exists.")
        continue
    
    code_prompt = stack_prompt.format(resources=code)
    
    # Generate tests and clean up output
    output = llm.predict(code_prompt)
    output = output.replace("```ts", "").replace("```typescript", "").replace("```", "").replace("typescript", "").strip()
    output += "\n\nimport * as NOCONFLICT_CDK from '@aws-cdk/core';\nconst app = new NOCONFLICT_CDK.App(); new CdkStack(app, 'CdkStack'); app.synth();"

    # Write out final code
    with open(f"lambda_stacks/{filename}.ts", "w") as f:
        f.write(output)