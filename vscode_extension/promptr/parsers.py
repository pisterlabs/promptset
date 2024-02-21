import json
import os
from itertools import product
from tree_sitter import Language, Parser, Tree
from tqdm import tqdm
import uuid

# TODO: This is a hack to get the language to work. FIND RELATIVE ADDRESS!!!    
if not os.path.exists("build/my-languages.so"):
    Language.build_library("build/my-languages.so", ["/home/dpaul/prompt-linter/vscode_extension/promptr/tree-sitter-python"])

PY_LANGUAGE = Language("build/my-languages.so", "python")


# Parsers should be of type: Tree -> list[prompt_dict]
# prompt_dict should be a dictionary with a prompt key and a metadata key
def find_from_file(tree: Tree):
    results = []
    query = PY_LANGUAGE.query(
        """(call
            function: (attribute
                object: (identifier) @obj
                attribute: (identifier) @fn
            )
            arguments: (_) @args
            (#eq? @fn "from_file")
            (#match? @obj "Template")
        )"""
    )

    for capture, name in query.captures(tree.root_node):
        if name != "args":
            continue
        results.append(capture.text.decode("utf-8"))
    return results


def find_assignments(tree: Tree):
    rems: list[str] = []
    for aug, left_type, right_type in product(
        ["augmented_", ""],
        ["attribute", "identifier"],
        ["integer", "string", "binary_operator"],
    ):
        query = PY_LANGUAGE.query(
            f"""({aug}assignment
                left: ({left_type}) @var.name
                right: ({right_type}) @var.value
            ) @assign"""
        )
        for capture, name in query.captures(tree.root_node):
            if name == "assign":
                rems.append(capture.text.decode("utf-8"))
    return rems


def all_strings(tree: Tree):
    """All Strings heuristic"""
    result = []

    query = PY_LANGUAGE.query("((string) @var.value)")

    for usage in query.captures(tree.root_node):
        string = usage[0].text.decode("utf-8")
        if string.count(" ") > 2:
            result.append({"string": usage[0].text.decode("utf-8"), "metadata": {}})

    return result


def new_line_in_string(tree: Tree):
    """New Line in String definition heuristic"""
    result = []

    var_def_query = PY_LANGUAGE.query(
        """(expression_statement
            (assignment
                left: (identifier)
                right: (string) @var.value
            )
        )"""
    )

    for usage in var_def_query.captures(tree.root_node):
        # heuristic, check if string has a newline in it, if so then it's probably a prompt
        res = usage[0].text.decode("utf-8")
        if "\n" in res:
            result.append({"prompt": res, "metadata": {}})

    return result


def used_prompt_or_template_name(tree: Tree):
    """Look for prompt or template in variable name"""
    result = []

    query = PY_LANGUAGE.query(
        """(expression_statement
            (assignment
                left: (identifier) @var.name
                right: (_)
            )
            (#match? @var.name "([Pp][Rr][Oo][Mm][Pp][Tt]|[Tt][Ee][Mm][Pp][Ll][Aa][Tt][Ee])")
        ) @assign"""
    )
    query_2 = PY_LANGUAGE.query(
        """(expression_statement
            (augmented_assignment
                left: (identifier) @var.name
                right: (_)
            )
            (#match? @var.name "([Pp][Rr][Oo][Mm][Pp][Tt]|[Tt][Ee][Mm][Pp][Ll][Aa][Tt][Ee])")
        ) @assign"""
    )

    for usage, name in query.captures(tree.root_node):
        if name == "assign":
            result.append(usage.text.decode("utf-8"))

    for usage, name in query_2.captures(tree.root_node):
        if name == "assign":
            result.append(usage.text.decode("utf-8"))

    return result


def used_langchain_tool_class(tree: Tree):
    tool_query = PY_LANGUAGE.query(
        """(class_definition
            name: (identifier)
            superclasses: (argument_list
                (identifier) @superclass
            )
            (#match? @superclass "Tool")
            body: (block
                (expression_statement
                    (assignment
                        left: (identifier) @ident
                        right: (string) @string
                    )
                    (#eq? @ident "description")
                )
            )
        )"""
    )
    result = []
    for capture, name in tool_query.captures(tree.root_node):
        if name == "string":
            result.append(capture.text.decode("utf-8"))
    return result


def used_langchain_tool(tree: Tree):
    tool_query = PY_LANGUAGE.query(
        """(decorated_definition
        (decorator (identifier) @dec)
        definition: (function_definition
            name: (identifier)
            parameters: (_)
            return_type: (_)
            body: (block
                (expression_statement (string) @docstring))
        )
        (#eq? @dec "tool")
    )"""
    )
    result = []
    for capture, name in tool_query.captures(tree.root_node):
        if name == "docstring":
            result.append(capture.text.decode("utf-8"))
    return result


def used_in_langchain_llm_call(tree: Tree):
    """Find variables used in langchain llm calls"""
    result = []
    from_template_query = PY_LANGUAGE.query(
        """(call 
        function: 
        (attribute
            object: (identifier) @ob
            attribute: (identifier)
        )
        (#match? @ob "Template$")
        arguments: (argument_list)
    ) @call"""
    )

    template_query = PY_LANGUAGE.query(
        """(call 
        function: (identifier) @ob
        (#match? @ob "(Template|Message)$")
        arguments: (argument_list)
    ) @call"""
    )

    for capture in template_query.captures(tree.root_node):
        if capture[1] == "call":
            result.append(capture[0].text.decode("utf-8"))

    for capture in from_template_query.captures(tree.root_node):
        if capture[1] == "call":
            result.append(capture[0].text.decode("utf-8"))

    return result


def used_chat_function(tree: Tree):
    query = PY_LANGUAGE.query(
        """(call 
        function: 
        (attribute
            object: (identifier)
            attribute: (identifier) @fn
        )
        (#match? @fn "^(chat|summarize)$")
        arguments: (argument_list)
    ) @call"""
    )

    results = []
    for capture in query.captures(tree.root_node):
        if capture[1] == "call":
            results.append(capture[0].text.decode("utf-8"))
    return results


def used_in_openai_call(tree: Tree):
    """Find native openai library calls"""
    result = []

    call_query = PY_LANGUAGE.query(
        """(call
            function: (attribute) @fn.name
            arguments: (argument_list) @fn.args
            (#match? @fn.name "(\.[Cc]hat)?\.?[cC]ompletions?\.create")
        )"""
    )

    for usage in filter(
        lambda x: x[1] == "fn.args", call_query.captures(tree.root_node)
    ):
        result.append(usage[0].text.decode("utf-8"))

    return result


class PromptDetector:
    def __init__(self):
        self.parser = Parser()
        self.parser.set_language(PY_LANGUAGE)

        self.heuristics = []

    def add_heuristic(self, heuristic):
        self.heuristics.append(heuristic)

    def detect_prompts(self, filenames: list[str], run_id):
        results = {}

        for filename in tqdm(filenames):
            results |= self._detect_prompts(filename)

        with open(f"/home/dpaul/prompt-linter/vscode_extension/promptr/prompts-{uuid.uuid4()}.json", "w") as w:
            json.dump(results, w)
        
        return results

    def _detect_prompts(self, filename: str):
        with open(filename, "rb") as f:
            tree = self.parser.parse(f.read())

        results = {}
        for heuristic in self.heuristics:
            results[heuristic.__name__] = heuristic(tree)
        results["variables"] = find_assignments(tree)

        return {filename: results}