import os
import re
import glob
import json
from argparse import ArgumentParser

from tree_sitter import Language, Parser, Tree, Node

PY_LANGUAGE = Language("./build/my-languages.so", "python")
parser = Parser()
parser.set_language(PY_LANGUAGE)


keys_per_method = {
    "used_in_openai_call": ["prompt", "messages"],
    "used_chat_function": ["text", "pos-01", "message"],
    "used_in_langchain_llm_call": [
        "message",
        "prompt",
        "template",
        "pos-01",
        "pos-02",
        "pos-03",
        "content",
        "prefix",
        "suffix",
    ],
    "used_langchain_tool_class": ["pos-01"],
    "used_langchain_tool": ["pos-01"],
    "used_prompt_or_template_name": None,
}


def parse_tree(tree: Tree):
    strings = []

    query = PY_LANGUAGE.query("(string) @string")
    for string, _ in query.captures(tree.root_node):
        # Drop anything without at least one whitespace
        if not re.search(r"[\s\\n]", string.text.decode("utf-8")):
            continue

        strings.append(string.text.decode("utf-8"))

    return strings


def parse_kwarg(tree: Node):
    splits = tree.text.decode("utf-8").split("=", maxsplit=1)
    if len(splits) == 2:
        text = splits[1].strip()
        query = PY_LANGUAGE.query(
            """(module 
                (expression_statement 
                    (list 
                        (dictionary 
                            (pair
                                key: (_) @key
                                value: (_) @val 
                                (#match? @key "content")
                            )
                        )
                    )
                )
            )"""
        )
        texts = []
        tree2 = parser.parse(bytes(text, "utf-8"))
        captures = query.captures(tree2.root_node)
        for capture, name in captures:
            if name == "val" and capture.type == "string":
                texts.append(capture.text.decode("utf-8").strip())

        if texts == [] and captures == []:
            texts.append(text)
        return splits[0].strip(), texts

    return None


def parse_args(tree: Tree):
    text = tree.root_node.text.decode("utf-8")
    fn_name = text.split("(", maxsplit=1)[0].strip()
    fn_name = fn_name.split("\n")[-1].strip()

    query = PY_LANGUAGE.query(
        f"""(call 
            function: (_) @fn
            arguments: (argument_list
                (_) @arg
            )
            (#eq? @fn "{fn_name}")
        )"""
    )

    args = {}
    for pos, (arg, aname) in enumerate(query.captures(tree.root_node)):
        if aname == "fn" or arg.type == "comment":
            pass

        elif kwarg := parse_kwarg(arg):
            for idx, k in enumerate(kwarg[1]):
                if k.strip() != "":
                    args[f"{kwarg[0]}-{idx:02d}"] = k
        elif arg.text.decode("utf-8") == "()":
            pass

        else:
            args[f"pos-{pos:02d}"] = arg.text.decode("utf-8")

    return args


def save_for_black(name, folder, calls, wrap, wrap_extra):
    for ogfilename, el in calls.items():
        filename = ogfilename.replace("/", "_")
        usages = el.get(name, [])

        # don't lose empty files
        if usages == []:
            usages.append("()")

        for idx, call in enumerate(usages):
            with open(f"{folder}/{hash(filename)}-{idx}.py", "w") as f:
                f.write(f"# {ogfilename}\n")
                if wrap_extra:
                    f.write("fn(" + call + ")")
                elif wrap:
                    f.write("fn" + call)
                else:
                    f.write(call)

    os.system(f"black {folder} -q")


def get_black_trees(folder):
    calls = []
    for file in glob.glob(f"{folder}/*.py"):
        with open(file) as f:
            filename = f.readline()[2:].strip()

        with open(file, "rb") as f:
            tree = parser.parse(f.read())
            calls.append((filename, parse_args(tree)))
    return calls


def not_empty(s: str) -> bool:
    return s != ""


def strip(s: str) -> str:
    return s.strip()


def run(name, in_data, out_data, keys=None, wrap=True, wrap_extra=False):
    keys = keys_per_method.get(name.replace("_sub", ""))

    folder = f"data/black/2.0-{name}"
    os.makedirs(folder, exist_ok=True)
    save_for_black(name, folder, in_data, wrap, wrap_extra)

    for filename, args in get_black_trees(folder):
        if filename not in out_data:
            out_data[filename] = []

        prompts = []
        for key, value in args.items():
            if keys is None or any(k in key for k in keys):
                prompts.append(value)

        for prompt in filter(not_empty, map(strip, prompts)):
            tree = parser.parse(bytes(prompt, "utf-8"))
            strings = parse_tree(tree)

            out_data[filename].extend(strings)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "--run_id",
        type=int,
        required=True,
    )
    args = argparser.parse_args()
    with open(f"data/repo_data_export_{args.run_id:03d}.json") as f:
        in_data = json.load(f)

    data = {}
    run("used_in_openai_call_sub", in_data, data)
    run("used_chat_function_sub", in_data, data)
    run("used_in_langchain_llm_call_sub", in_data, data, wrap=False)
    run("used_langchain_tool_class", in_data, data, wrap_extra=True)
    run("used_langchain_tool", in_data, data, wrap_extra=True)
    run("used_prompt_or_template_name", in_data, data, wrap_extra=True)

    with open(f"data/grouped-data-{args.run_id:03d}.json", "w") as w:
        json.dump(data, w, indent=2, ensure_ascii=False)
    print(len(data))
