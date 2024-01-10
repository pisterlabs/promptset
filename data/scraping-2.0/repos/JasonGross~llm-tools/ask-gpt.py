#!/usr/bin/env python3
import os, sys
import argparse
from memocache import Memoize
from pathlib import Path
import openai

# get script directory
script_dir = Path(os.path.dirname(os.path.realpath(__file__)))

if tuple(sys.argv[1:]) == ('--chat-help',):
    help(openai.ChatCompletion.create)
    sys.exit(0)

parser = argparse.ArgumentParser()
parser.add_argument('query', type=str)
parser.add_argument('--arg-key', type=str, nargs='*', default=[])
parser.add_argument('--arg-val', type=str, nargs='*', default=[])
parser.add_argument('--output', '-o', type=argparse.FileType('w'), default=sys.stdout)
parser.add_argument('--log', '-l', type=argparse.FileType('w'), default=sys.stderr)
parser.add_argument('--api-key', type=str, default='@~/.openai.apikey.txt')
parser.add_argument('--no-memoize', action='store_false', default=True, dest='memoize')
parser.add_argument('--memoize-dir', type=str, default=script_dir / 'ask-gpt-cache')
parser.add_argument('--context', type=str, default='''You are a helpful Large Language Model.''')
parser.add_argument('--model', '-m', type=str, default='gpt-3.5-turbo')
parser.add_argument('--top_p', type=float, default=1)
parser.add_argument('--temperature', type=float, default=1)

args = parser.parse_args()

if args.api_key.startswith('@'):
    # expand ~
    openai.api_key_path = os.path.expanduser(args.api_key[1:])
else:
    openai.api_key = args.api_key

if args.memoize:
    openai.Completion.create = Memoize(openai.Completion.create, name='Completion.create')
    openai.ChatCompletion.create = Memoize(openai.ChatCompletion.create, name='ChatCompletion.create')

if args.query == '-': args.query = sys.stdin.read()
elif args.query.startswith('@'):
    with open(args.query[1:], 'r') as f:
        args.query = f.read()

response = openai.ChatCompletion.create(
    messages=[{"role": "system", "content": args.context}, {"role": "user", "content": args.query}],
    model=args.model,
    top_p=args.top_p,
    temperature=args.temperature,
    **{k:v for k,v in zip(args.arg_key, args.arg_val)}
)

print(response.choices[0].message.content, file=args.output)
