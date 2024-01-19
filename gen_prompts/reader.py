import os
import re
import copy
import json
from argparse import ArgumentParser
from uuid import uuid4

from tree_sitter import Language, Parser, Tree, Node
from tqdm import tqdm

PY_LANGUAGE = Language("./build/my-languages.so", "python")
parser = Parser()
parser.set_language(PY_LANGUAGE)


keys_per_method = {
    "used_in_openai_call": ["prompt", "messages"],
    "used_chat_function": ["text", "pos-00", "message", "query"],
    "used_in_langchain_llm_call": [
        "message",
        "prompt",
        "template",
        "pos-00",
        "pos-01",
        "pos-02",
        "content",
        "prefix",
        "suffix",
    ],
    "used_langchain_tool_class": ["pos-00"],
    "used_langchain_tool": ["pos-00"],
    "used_prompt_or_template_name": None,
}


class Placeholder:
    def __init__(self):
        self.index = 0

    def __repr__(self) -> str:
        return "PLACEHOLDER"

    def __str__(self) -> str:
        return "PLACEHOLDER"

    def __getitem__(self, index):
        return "PLACEHOLDER"

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == 0:
            self.index += 1
            return "PLACEHOLDER"
        raise StopIteration

    def __add__(self, other):
        return str(self) + other

    def __len__(self):
        return 1

    def __eq__(self, o):
        return True

    def __lt__(self, o):
        return False


def prepped_eval(expr, locals_map, depth=10):
    if depth <= 0:
        return None

    _locals_map = copy.deepcopy(locals_map)
    for v in _locals_map.values():
        if isinstance(v, Placeholder):
            v.index = 0
    _locals_map["input"] = "INPUT"
    try:
        return str(eval(expr, {}, _locals_map))
    except NameError as e:
        # Don't include indexing or joining or list
        if (
            f"join({e.name})" in expr
            or f"[{e.name}]" in expr
            or expr == e.name
            or f"*{e.name}" in expr
        ):
            return None

        _locals_map[e.name] = Placeholder()
        return prepped_eval(expr, _locals_map, depth - 1)
    except IndentationError:
        return prepped_eval(expr.replace("\n", ""), _locals_map, depth - 1)
    except TypeError as e:
        if (
            e.args
            and e.args[0] == 'can only concatenate str (not "Placeholder") to str'
        ):
            return prepped_eval(
                expr, {k: str(v) for k, v in _locals_map.items()}, depth - 1
            )
    except Exception:
        return None
    return None


def interp(vars: list[str]):
    if not vars:
        return {}

    query = PY_LANGUAGE.query(
        """(augmented_assignment
            left: (identifier) @var.name
            right: (_)
        )"""
    )
    for var in vars:
        tree = parser.parse(bytes(var, "utf-8"))
        for capture, _ in query.captures(tree.root_node):
            return {}

    query = PY_LANGUAGE.query(
        """(assignment
            left: (identifier) @var.name
            right: (_) @expr
        )"""
    )
    ssa = {}
    ssa_trees: dict[str, Node] = {}
    name = None
    for var in vars:
        tree = parser.parse(bytes(var, "utf-8"))
        for capture, ctype in query.captures(tree.root_node):
            if ctype == "var.name":
                name = capture.text.decode("utf-8")
                ssa[name] = ssa.get(name, 0) + 1
            elif name and ctype == "expr":
                ssa_trees[name] = capture

    if any(value > 1 for value in ssa.values()):
        return {}

    locals_map = {}
    mystr = str(uuid4())
    for name in ssa_trees.keys():
        locals_map[name] = f"{name}{mystr}"

    variable_values = {}
    for name, tree in ssa_trees.items():
        result = prepped_eval(tree.text.decode("utf-8"), locals_map)
        if result:
            variable_values[name] = result

    for name, expr in variable_values.items():
        for var2, expr2 in variable_values.items():
            if var2 == name:
                continue

            if f"{var2}{mystr}" in expr:
                variable_values[name] = expr.replace(f"{var2}{mystr}", expr2)

    for key, tree in ssa_trees.items():
        if key not in variable_values:
            variable_values[key] = str(tree.text.decode("utf-8").strip("\"'"))

    return variable_values


def trim_tree(tree_bytes: bytes, node: Node, value: bytes):
    return tree_bytes[: node.start_byte] + value + tree_bytes[node.end_byte :]


def parse_tree(tree: Tree, vars: list[str]):
    strings = []
    variable_values = interp(vars)
    result = prepped_eval(tree.root_node.text.decode("utf-8"), variable_values)
    if result and result != "PLACEHOLDER":
        return [result]

    query = PY_LANGUAGE.query("(interpolation ((identifier) @ident))")
    for interpolation, _ in query.captures(tree.root_node):
        ident = interpolation.text.decode("utf-8")
        if value := variable_values.get(ident, ""):
            tree_bytes = (
                tree.root_node.text.decode("utf-8")
                .replace("{" + ident + "}", value)
                .strip("f")
            )
            tree = parser.parse(bytes(tree_bytes, "utf-8"))

    query = PY_LANGUAGE.query("(identifier) @identifier")
    for identifier, _ in query.captures(tree.root_node):
        ident = identifier.text.decode("utf-8")

        if value := variable_values.get(ident, ""):
            tree_bytes = trim_tree(
                tree.root_node.text, identifier, bytes(value, "utf-8")
            )
            tree = parser.parse(tree_bytes)

    query = PY_LANGUAGE.query("(string) @string")
    for string, _ in query.captures(tree.root_node):
        # Drop anything without at least one whitespace
        try:
            str_str = string.text.decode("utf-8")
        except UnicodeDecodeError:
            continue

        if not re.search(r"[\s\\n]", str_str.strip()) and len(str_str) < 12:
            continue

        res = prepped_eval(str_str, variable_values)
        if res:
            strings.append(res)

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
    pos = 0
    for arg, aname in query.captures(tree.root_node):
        if aname == "fn" or arg.type == "comment":
            pass

        elif kwarg := parse_kwarg(arg):
            for idx, k in enumerate(kwarg[1]):
                if k.strip() != "":
                    args[f"{kwarg[0]}-{idx:02d}"] = k
        else:
            args[f"pos-{pos:02d}"] = arg.text.decode("utf-8")
            pos += 1

    return args


def save_for_black(folder, calls, wrap, wrap_extra):
    for ogfilename, usages in calls.items():
        filename = ogfilename.replace("/", "_")

        for idx, call in enumerate(usages):
            with open(f"{folder}/{hash(filename)}-{idx}.py", "w") as f:
                f.write(f"# {ogfilename}\n")
                if wrap_extra:
                    f.write("fn(" + call + ")")
                elif wrap:
                    f.write("fn" + call)
                else:
                    f.write(call)

    print(f"black {folder} -q")
    os.system(f"black {folder} -q")


def get_black_trees(folder):
    for root_dir, _, files in os.walk(folder):
        for file in files:
            with open(os.path.join(root_dir, file), "rb") as f:
                name, tree_bytes = f.read().split(b"\n", 1)
            name = name.decode("utf-8")[2:].strip()
            tree = parser.parse(tree_bytes)
            yield (name, parse_args(tree))


def not_empty(s: str) -> bool:
    return s != ""


def strip(s: str) -> str:
    return s.strip()


def formatter(folder, name, in_data, wrap=True, wrap_extra=False):
    os.makedirs(folder, exist_ok=True)
    name_to_strs = {x: y[name] for x, y in in_data.items() if y[name]}
    save_for_black(folder, name_to_strs, wrap, wrap_extra)


def get_strings(folder, name, in_data, out_data):
    keys = keys_per_method.get(name)
    found = 0
    found_repos = 0
    for filename, args in tqdm(get_black_trees(folder)):
        prompts = []
        for key, value in args.items():
            if keys is None or any(k in key for k in keys):
                prompts.append(value)

        for prompt in filter(not_empty, map(strip, prompts)):
            tree = parser.parse(bytes(prompt, "utf-8"))
            strings = parse_tree(tree, in_data[filename]["variables"])

            out_data[filename].extend(strings)
            found += len(strings)

        found_repos += 1 if len(prompts) > 0 else 0
    print(f"Found {found} strings with {name}")
    print(f"Found {found_repos} repos with {name}")


def run(run_id, name, in_data, out_data, wrap=True, wrap_extra=False):
    folder = f"data/black/2.0-{run_id:03d}-{name}"
    # formatter(folder, name, in_data, wrap, wrap_extra)
    get_strings(folder, name, in_data, out_data)


def handle_from_file(in_data, out_data):
    fn_to_prompts = {
        "repos/amalabey~contributor/core~code_review~syntax.py": [
            "scrape/template_files/amalabey~contributor/methods.txt"
        ],
        "repos/amalabey~contributor/core~code_review~comments.py": [
            "scrape/template_files/amalabey~contributor/methods.txt"
        ],
        "repos/ayulockin~llm-eval-sweep/qa_full_sweeps.py": [
            "scrape/template_files/ayulockin~llm-eval-sweep/qa/prompt_template_1.txt",
            "scrape/template_files/ayulockin~llm-eval-sweep/qa/prompt_template_2.txt",
        ],
        "repos/ayulockin~llm-eval-sweep/maths_sweeps.py": [
            "scrape/template_files/ayulockin~llm-eval-sweep/maths/maths_prompt_template_1.txt",
            "scrape/template_files/ayulockin~llm-eval-sweep/maths/maths_prompt_template_2.txt",
            "scrape/template_files/ayulockin~llm-eval-sweep/maths/maths_prompt_template_3.txt",
        ],
        "repos/GreenWizard2015~AIEnhancedTranslator/core~CAIAssistant.py": [
            "scrape/template_files/GreenWizard2015~AIEnhancedTranslator/translate_deep.txt",
            "scrape/template_files/GreenWizard2015~AIEnhancedTranslator/translate_shallow.txt",
        ],
        "repos/jfran75~utils/ai~code~tests~api_requests~request_local_model_like_openai2.py": [
            "scrape/template_files/jfran75~utils/prompt-generate-csharp-class.txt",
            "scrape/template_files/jfran75~utils/prompt-generate-python-class.txt",
            "scrape/template_files/jfran75~utils/prompt_file_template.txt",
        ],
        "repos/jfran75~utils/ai~code~tests~api_requests~request_local_model_like_openai1.py": [
            "scrape/template_files/jfran75~utils/prompt-generate-csharp-class.txt",
            "scrape/template_files/jfran75~utils/prompt-generate-python-class.txt",
            "scrape/template_files/jfran75~utils/prompt_file_template.txt",
        ],
        "repos/luckrnx09~abook/writer~comparison.py": [
            "scrape/template_files/luckrnx09~abook/prompts/concept_instruction.txt",
            "scrape/template_files/luckrnx09~abook/prompts/api_comparison.txt",
        ],
        "repos/luckrnx09~abook/writer~environment.py": [
            "scrape/template_files/luckrnx09~abook/prompts/environment.txt",
        ],
        "repos/luckrnx09~abook/writer~resource.py": [
            "scrape/template_files/luckrnx09~abook/prompts/resource.txt",
        ],
        "repos/xiye17~CachedLMQuery/examples~example_chat.py": [
            "scrape/template_files/xiye17~CachedLMQuery/demo_chat_template.tpl",
            "scrape/template_files/xiye17~CachedLMQuery/demo_template.tpl",
        ],
    }
    added = 0
    for fn, items in in_data.items():
        if items.get("find_from_file"):
            for prompt_file in fn_to_prompts.get(fn, []):
                with open(prompt_file) as f:
                    out_data[fn].append(f.read())
                    added += 1
    print(f"Added {added} prompts with from_file")


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--run_id", type=int, required=True)
    args = argparser.parse_args()
    with open(f"data/repo_data_export_{args.run_id:03d}.json") as f:
        in_data = json.load(f)

    data = {fn: [] for fn in in_data.keys()}
    run(args.run_id, "used_chat_function", in_data, data)
    run(args.run_id, "used_in_openai_call", in_data, data)
    run(args.run_id, "used_in_langchain_llm_call", in_data, data, wrap=False)
    run(args.run_id, "used_langchain_tool_class", in_data, data, wrap_extra=True)
    run(args.run_id, "used_langchain_tool", in_data, data, wrap_extra=True)
    run(args.run_id, "used_prompt_or_template_name", in_data, data, wrap_extra=True)
    handle_from_file(in_data, data)

    with open(f"data/grouped-data-{args.run_id:03d}-2.json", "w") as w:
        json.dump(data, w, indent=2, ensure_ascii=False)
