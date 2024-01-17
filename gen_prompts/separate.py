import os
import re
import json
from argparse import ArgumentParser

from tree_sitter import Language, Parser, Tree

PY_LANGUAGE = Language("./build/my-languages.so", "python")
parser = Parser()
parser.set_language(PY_LANGUAGE)
SEP = "\n----------------------------------------------------------------------------------------\n"

argparser = ArgumentParser()
argparser.add_argument(
    "--run_id",
    type=int,
    required=True,
)
args = argparser.parse_args()
run_id = args.run_id


def parse_tree(tree: Tree):
    strings = []
    identifiers = []
    interpolations = []

    query = PY_LANGUAGE.query("(string) @string")
    for string in query.captures(tree.root_node):
        if not re.search(r"[\s\\n]", string[0].text.decode("utf-8")):
            continue

        strings.append(string[0].text.decode("utf-8"))

    query = PY_LANGUAGE.query("(identifier) @identifier")
    for identifier in query.captures(tree.root_node):
        identifiers.append(identifier[0].text.decode("utf-8"))

    query = PY_LANGUAGE.query("(interpolation) @identifier")
    for interpolation in query.captures(tree.root_node):
        interpolations.append(interpolation[0].text.decode("utf-8"))

    return strings, identifiers, interpolations


def parse_prompts(in_data, name, data):
    prompt_count = 0
    strings_found = 0
    unique_strings = set()
    for repo, prompts in in_data.items():
        prompts = prompts.get(name, [])
        prompts = list(map(lambda x: x.strip(), prompts))
        prompts = list(filter(lambda x: x != "", prompts))
        prompt_count += len(prompts)
        if repo not in data:
            data[repo] = {"strings": [], "identifiers": [], "interpolations": []}

        for prompt in prompts:
            tree = parser.parse(bytes(prompt, "utf-8"))
            strings, identifiers, interpolations = parse_tree(tree)

            # Do we need this?
            # map(lambda x: re.sub(r"\{[^\"]*\"([^\"]*)\"[^}]*\}", r"\1", x), strings)
            strings_found += len(strings)
            data[repo]["strings"].extend(strings)
            data[repo]["identifiers"].extend(identifiers)
            data[repo]["interpolations"].extend(interpolations)
            unique_strings.update(strings)

    print("Found", prompt_count, "prompts in", name)
    print("Found", strings_found, "strings in", name)
    print("Found", len(unique_strings), "unique strings in", name)


def parse_metadata_file(filename):
    with open(filename) as f:
        metadata = json.load(f)

    data = {}
    clean_data = {}
    for key, value in metadata.items():
        data[key] = {}
        clean_data[key] = []
        for _, prompts in value.items():
            for prompt in prompts:
                prompt = re.sub(r"\{[^\"]*\"[^\"]*\"[^}]*\}", prompt)
                tree = parser.parse(bytes(prompt, "utf-8"))
                strings, identifiers, interpolations = parse_tree(tree)
                data[key][prompt] = {
                    "strings": strings,
                    "identifiers": identifiers,
                    "interpolations": interpolations,
                }
                clean_data[key].extend(strings)

    filename = os.path.splitext(os.path.basename(filename))[0]
    with open(f"xseparated-{filename}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    with open(f"clean-{filename}.json", "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    with open(f"grouped-data-{run_id:03d}.json") as w:
        data = json.load(w)
    out_data = {}

    # parse_prompts("chat-completions.txt")
    parse_prompts(data, "used_in_openai_call_sub", out_data)

    # parse_prompts("langchain-prompts.json")
    parse_prompts(data, "used_in_langchain_llm_call_sub", out_data)

    # parse_prompts("cohere-prompts.json")
    parse_prompts(data, "used_chat_function_sub", out_data)

    parse_prompts(data, "used_langchain_tool_class", out_data)
    parse_prompts(data, "used_langchain_tool", out_data)

    parse_prompts(data, "used_prompt_or_template_name", out_data)
    with open(f"separated-data-{run_id:03d}.json", "w") as w:
        json.dump(out_data, w, indent=2, ensure_ascii=False)

    # parse_metadata_file("reader_prompt_metadata.json")
    # parse_metadata_file("reader_prompt_metadata_plus.json")
