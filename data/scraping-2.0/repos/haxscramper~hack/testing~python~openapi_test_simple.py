#!/usr/bin/env python

import argparse
from pprint import pprint

parser = argparse.ArgumentParser(description="Openapi test")
parser.add_argument("key", type=str, help="OpenAI key")

args = parser.parse_args()

import os
import openai
openai.api_key = args.key
result = openai.Completion.create(
  model="gpt-4-32k",
  prompt="""
Convert the following Rust code to C++. Do not generate any extra definitions (enums, structs, functions etc.), besides the converted version of the code below. Resulting code should contain ONLY converted results and no extraneous definitions. Assume all required types, constants and procedures have already been defined, all required includes were added to the code. All comments must be preserved in the output. All source code must be mapped, no content should be skipped:

enum LabelKind {
    Inline,
    Multiline,
}

struct LabelInfo<'a, S> {
    kind: LabelKind,
    label: &'a Label<S>,
}
  """,
  max_tokens=200,
  temperature=0.5
)

pprint(result)
