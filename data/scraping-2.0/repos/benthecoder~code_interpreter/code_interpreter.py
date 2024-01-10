import ast
import json
import os
from inspect import signature, Parameter
import openai
from dotenv import load_dotenv
from pydantic import create_model
from fastcore.utils import nested_idx
import logging
import click

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


def askgpt(user, system=None, model="gpt-3.5-turbo", **kwargs):
    """Queries the OpenAI GPT model."""
    msgs = [{"role": "user", "content": user}]
    if system:
        msgs.insert(0, {"role": "system", "content": system})
    return openai.ChatCompletion.create(model=model, messages=msgs, **kwargs)


def response(compl):
    """Prints the content of the message from the given response."""
    print(nested_idx(compl, "choices", 0, "message", "content"))


def sum(a: int, b: int = 1) -> int:
    """Adds two numbers together."""
    return a + b


def schema(f):
    """Generates a schema for a given function using pydantic and inspect."""
    params = signature(f).parameters
    kw = {
        n: (o.annotation, ... if o.default == Parameter.empty else o.default)
        for n, o in params.items()
    }
    s = create_model(f"Input for `{f.__name__}`", **kw).model_json_schema()

    return {"name": f.__name__, "description": f.__doc__, "parameters": s}


def run(code: str):
    """Executes the given Python code and returns the result."""
    tree = ast.parse(code)
    last_node = tree.body[-1] if tree.body else None

    # If the last node is an expression, modify the AST to capture the result
    if isinstance(last_node, ast.Expr):
        tgts = [ast.Name(id="_result", ctx=ast.Store())]
        assign = ast.Assign(targets=tgts, value=last_node.value)
        tree.body[-1] = ast.fix_missing_locations(assign)

    ns = {}
    exec(compile(tree, filename="<ast>", mode="exec"), ns)
    return ns.get("_result", None)


def python(code: str):
    """Prompts the user to execute a Python code and returns the result."""
    if click.confirm(f"Do you want to run this code?\n```\n{code}\n```"):
        return run(code)
    return "#FAIL#"


def call_func(c, verbose=False):
    """Calls a function based on a message choice."""
    fc = nested_idx(c, "choices", 0, "message", "function_call")

    if not fc:
        return "No function created, try again..."

    if fc.name not in {"python", "sum"}:
        return f"Not allowed: {fc.name}"

    try:
        args = json.loads(fc.arguments)
        if verbose:
            logger.info(args["code"])
        f = globals()[fc.name]
        return f(**args)
    except json.JSONDecodeError:
        if verbose:
            logger.info(fc.arguments)
        return run(fc.arguments)


def code_interpreter(query, verbose=True):
    """Interprets the given query."""
    c = askgpt(
        query,
        system="Use Python for any required computation",
        functions=[schema(python)],
    )
    if nested_idx(c, "choices", 0, "message", "function_call"):
        return call_func(c, verbose)

    return response(c)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("query")
@click.option("--verbose", is_flag=True, help="Enable verbose output.")
def q(query, verbose):
    """Interprets the given query."""
    result = code_interpreter(query, verbose)
    click.echo(result)


if __name__ == "__main__":
    cli()
