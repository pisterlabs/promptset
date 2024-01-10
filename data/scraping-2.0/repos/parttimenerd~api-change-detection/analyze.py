#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Optional
from tree_sitter_languages import get_parser
from openai import OpenAI
import openai
import tiktoken
import requests
import json


def change_prompt(diff: str, classes: List[str]) -> str:
    """ Prompt for asking for the API changes"""
    return f"""
Please list the abstract changes to the classes {', '.join(classes)} API's methods 
and properties that caused the following changes in the test code, 
in JSON of the form 
'[{{"class": class name, "old": old method signature or property, "new": new method signature or property, 
"decription": description of the change, 
"example": {{"old": old code, "new": new code}}}}, ...]': 
The diff is:

{diff}


in JSON, be as abstract and concise as possible, 
generalize wherever possible to keep the list small.
Only list changes supported by the diff.
Keep it short and general.
Take the examples from the diff.
"""


BASEDIR = Path(__file__).resolve().parent
JDK_DIR = BASEDIR / "jdk"
OPENAI_KEY_FILE = BASEDIR / ".openai.key"
OPENAI_KEY_FILE_FORMAT = """
key=<your key>
server=https://api.openai.com
"""
CLASS_CACHE = BASEDIR / ".class.cache"

@dataclass
class Class:
    name: str
    methods: List[str]
    properties: List[str]

    def to_json(self):
        return {
            "name": self.name,
            "methods": self.methods,
            "properties": self.properties
        }

    @staticmethod
    def from_json(json: dict) -> 'Class':
        return Class(json["name"], json["methods"], json["properties"])


@dataclass
class ClassCacheEntry:
    revision: str
    pattern: str
    classes: List[Class]

    @staticmethod
    def from_json(json: dict) -> 'ClassCacheEntry':
        return ClassCacheEntry(
            json["revision"],
            json["pattern"],
            [Class.from_json(c) for c in json["classes"]]
        )

    def to_json(self) -> dict:
        return {
            "revision": self.revision,
            "pattern": self.pattern,
            "classes": [c.to_json() for c in self.classes]
        }


@dataclass
class OpenAISettings:
    key: str
    server: str

    @staticmethod
    def from_file(file: Path) -> 'OpenAISettings':
        settings = {}
        for line in file.read_text().split("\n"):
            if line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            settings[key] = value
        return OpenAISettings(settings["key"], settings["server"])

    @staticmethod
    def get():
        if OPENAI_KEY_FILE.exists():
            return OpenAISettings.from_file(OPENAI_KEY_FILE)
        print("Please store your OpenAI API key in .openai.key",
              file=sys.stderr)
        # create file
        OPENAI_KEY_FILE.write_text(OPENAI_KEY_FILE_FORMAT)
        sys.exit(1)


def prompt_gpt(prompt: str, model="gpt-35-4k-0613"):
    settings = OpenAISettings.get()
    url = f"{settings.server}/openai/deployments/{model}/chat/completions?api-version=2023-05-15"
    headers = {
        "Content-Type": "application/json",
        "api-key": settings.key
    }
    data = {
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code != 200:
        raise Exception(f"Unexpected response code: {response.status_code}")

    response_json = response.json()
    choices = response_json.get('choices', [])
    if len(choices) != 1:
        raise Exception(f"Unexpected number of choices: {len(choices)}")

    choice = choices[0]
    text = choice.get('message', {}).get('content', '')
    finish_reason = choice.get('finish_reason', '')
    if finish_reason != 'stop':
        raise Exception(f"Unexpected finish reason: {finish_reason}")

    return text


def count_tokens(prompt: str) -> int:
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(enc.encode(prompt))


class WithRevision:
    """ bring the jdk folder to the revision """

    def __init__(self, revision: str):
        self.revision: str = revision
        self.old_revision: Optional[str] = None

    def __enter__(self):
        self.old_revision = subprocess.check_output(
            "git rev-parse HEAD",
            cwd=JDK_DIR, shell=True).decode("utf-8").strip()
        subprocess.check_call(f"git checkout -f {self.revision}",
                              cwd=JDK_DIR, shell=True,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        subprocess.check_call(f"git reset --hard master",
                              cwd=JDK_DIR, shell=True,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        subprocess.check_call(f"git checkout {self.old_revision}",
                              cwd=JDK_DIR, shell=True,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)


tree_sitter_parser = get_parser("java")


def public_classes_in_file(file: Path) -> List[Class]:
    """ returns a list of all public classes in a Java file """
    content = file.read_text()
    classes: List[Class] = []

    def modifiers(decl_node) -> Set[str]:
        return set(cc.text.decode() for child in decl_node.children if
                   child.type == "modifiers" for cc in child.children)

    def name(decl_node) -> str:
        return [child.text.decode() for child in decl_node.children
                if child.type == "identifier"][0]

    def body(decl_node):
        return [child for child in decl_node.children
                if child.type in ["class_body", "interface_body"]][0]

    def properties(decl_node):
        return [child for child in body(decl_node).children
                if child.type == "field_declaration"] + [c for child in body(decl_node).children
                if child.type == "constant_declaration" for c in child.children if c.type == "variable_declarator"]

    def methods(decl_node):
        return [child for child in body(decl_node).children
                if child.type == "method_declaration"]

    def recursive_walk(node, parent: str = "",
                       is_parent_interface: bool = False):
        is_interface = node.type == "interface_declaration"
        is_class = node.type == "class_declaration"
        if not is_interface and not is_class:
            return
        mods = modifiers(node)
        if not is_parent_interface and not is_interface:
            if "public" not in mods:
                return
        combined_name = f"{parent}.{name(node)}" \
            if parent else name(node)
        classes.append(Class(combined_name, [name(m) for m in methods(node)],
                             [name(p) for p in properties(node)]))
        for child in body(node).children:
            recursive_walk(child, combined_name, is_interface)


    tree = tree_sitter_parser.parse(content.encode("utf-8"))
    for n in tree.root_node.children:
        recursive_walk(n)
    return classes


def _public_classes(revision: str, file_pattern: str) -> List[Class]:
    """
    Returns a list of all public classes and interfaces in the JDK
    :param revision:
    :param file_pattern:
    :return:
    """
    def inner(revision: str, file_pattern: str) -> List[str]:
        classes = []
        for file in JDK_DIR.glob(file_pattern):
            if file.is_dir():
                classes.extend(
                    inner(revision, f"{file}/*"))
                continue
            if file.suffix != ".java":
                continue
            classes.extend(public_classes_in_file(file))
        return classes

    with WithRevision(revision):
        return inner(revision, file_pattern)


@dataclass
class ClassCache:
    entries: List[ClassCacheEntry]

    @staticmethod
    def load() -> 'ClassCache':
        if not CLASS_CACHE.exists():
            return ClassCache([])
        return ClassCache(
            [ClassCacheEntry.from_json(e) for e in
             json.loads(CLASS_CACHE.read_text())]
        )

    def save(self):
        CLASS_CACHE.write_text(json.dumps(
            [e.to_json() for e in self.entries],
            indent=2
        ))

    @staticmethod
    def get(revision: str, pattern: str) -> Optional[List[str]]:
        cache = ClassCache.load()
        for entry in cache.entries:
            if entry.revision == revision and entry.pattern == pattern:
                return entry.classes
        classes = _public_classes(revision, pattern)
        cache.entries.append(
            ClassCacheEntry(revision, pattern, classes))
        cache.save()
        return classes


def public_classes(revision: str, file_pattern: str) -> List[Class]:
    return ClassCache.get(revision, file_pattern)


def public_class_names(revision: str, file_pattern: str) -> List[str]:
    return [c.name for c in public_classes(revision, file_pattern)]

def method_names(revision: str, file_pattern: str) -> List[str]:
    return [m for c in public_classes(revision, file_pattern) for m in c.methods]

def property_names(revision: str, file_pattern: str) -> List[str]:
    return [p for c in public_classes(revision, file_pattern) for p in c.properties]

@dataclass
class Example:
    old: Optional[str]
    new: Optional[str]

    def to_json(self):
        return {
            "old": self.old,
            "new": self.new
        }


@dataclass
class EntityDiff:
    old: str
    new: str
    description: str
    example: Optional[Example]

    def to_json(self, class_name: str) -> dict:
        return {
            "class": class_name,
            "old": self.old,
            "new": self.new,
            "description": self.description,
            "example": self.example.to_json() if self.example else None
        }

    @staticmethod
    def from_json(json: dict) -> 'EntityDiff':
        return EntityDiff(json["old"], json["new"],
                          json["description"],
                          Example(json["example"]["old"] if "old" in json["example"] else None,
                                  json["example"]["new"] if "new" in json["example"] else None) if "example" in json else None)


@dataclass
class ClassDiff:
    name: str
    entity_diffs: List[EntityDiff]

    def to_json(self) -> list:
        return [e.to_json(self.name) for e in self.entity_diffs]

    @staticmethod
    def from_json(json: list) -> 'ClassDiff':
        assert (len(json) > 0 and
                all(json[0]["class"] == e["class"] for e in json))
        return ClassDiff(json[0]["class"],
                         [EntityDiff.from_json(e) for e in json])

    def filter(self, classes: List[str], old_idents: List[str], new_idents: List[str], special_classes: List[str]) -> 'ClassDiff':
        """ filter out all diffs that are broken """

        def _check_name(name: str, idents: List[str]) -> bool:
            if self.name in special_classes:
                return True
            parts = name.split("(")[0].replace("...", "").split(".")
            if name in ["", "N/A"]:
                return True
            if len(parts) <= 2:
                return parts[-1] in (idents + classes)
            return False

        def check_name(name: str, idents: List[str]) -> bool:
            ret = _check_name(name, idents)
            return ret

        return ClassDiff(
            self.name,
            [entity_diff for entity_diff in self.entity_diffs
             if entity_diff.example is not None and entity_diff.example.old is not None and entity_diff.example.new is not None \
             and "no change " not in entity_diff.description.lower() and check_name(entity_diff.old, old_idents) and check_name(entity_diff.new, new_idents) \
             and entity_diff.old != entity_diff.new]
        )

@dataclass
class AnalyzedDiff:
    class_diffs: List[ClassDiff]

    def to_json(self):
        return [c.to_json() for c in self.class_diffs if len(c.entity_diffs) > 0]

    @staticmethod
    def from_json(json: list) -> 'AnalyzedDiff':
        # collect JSON for each class
        class_json: Dict[str, list] = {}
        for entry in json:
            if entry["class"] not in class_json:
                class_json[entry["class"]] = []
            class_json[entry["class"]].append(entry)

        # create ClassDiff objects
        class_diffs = []
        for class_name, class_json in class_json.items():
            class_diffs.append(ClassDiff.from_json(class_json))
        return AnalyzedDiff(class_diffs)

    def filter(self, classes: List[str], old_idents: List[str], new_idents: List[str], special_classes: List[str]) -> 'AnalyzedDiff':
        """ filter out all diffs that are not in classes """
        return AnalyzedDiff(
            [class_diff.filter(classes, old_idents, new_idents, special_classes) for class_diff in self.class_diffs]
        )


def analyze_diff(diff: str, classes: List[str]) -> AnalyzedDiff:
    prompt = change_prompt(diff, classes)
    result = prompt_gpt(prompt)
    return AnalyzedDiff.from_json(json.loads(result))


def get_diff(ref1: str, ref2: str, file: Path,
             remove_license_diff: bool = True) -> str:
    diff = subprocess.check_output(
        f"git diff {ref1}..{ref2} '{file}'",
        cwd=JDK_DIR, shell=True).decode("utf-8").strip()
    if remove_license_diff:
        lines = diff.split("\n")
        new_lines = []
        found_copyright = False
        found_end_of_license = False
        for line in lines:
            if found_end_of_license:
                new_lines.append(line)
                continue
            if "Copyright (c)" in line:
                found_copyright = True
            if not found_copyright:
                new_lines.append(line)
                continue
            if len(line) < 3 or not line[2] == "*":
                found_end_of_license = True
                new_lines.append(line)
        diff = "\n".join(new_lines)
    return diff


def combine(analyzed_diffs: List[AnalyzedDiff]) -> AnalyzedDiff:
    """ combine multiple analyzed diffs into one """
    class_diffs = {}
    for analyzed_diff in analyzed_diffs:
        for class_diff in analyzed_diff.class_diffs:
            if class_diff.name not in class_diffs:
                class_diffs[class_diff.name] = class_diff
            else:
                class_diffs[class_diff.name].entity_diffs.extend(
                    class_diff.entity_diffs)
    return AnalyzedDiff(list(class_diffs.values()))


def analyze_diff_safe(diff: str, classes: List[str], old_idents: List[str], new_idents: List[str], special_classes: List[str], count=3) -> Optional[AnalyzedDiff]:
    if count == 0:
        return None
    try:
        analyzed = analyze_diff(diff, classes)
        print("======= analyzed diff =======")
        print(json.dumps(analyzed.to_json(), indent=2))
        print("======= filtered diff =======")
        filtered = analyzed.filter(classes, old_idents, new_idents, special_classes)
        print(json.dumps(filtered.to_json(), indent=2))
        if len(filtered.class_diffs) == 0:
            return analyze_diff_safe(diff, classes, old_idents, new_idents, special_classes, count - 1)
        return filtered
    except Exception as e:
        return analyze_diff_safe(diff, classes, old_idents, new_idents, special_classes, count - 1)


def analyze_ref_diffs(ref1: str, ref2: str,
                      test_file_pattern: str,
                      api_file_pattern: str,
                      special_classes: List[str]) -> List[AnalyzedDiff]:
    classes = list(set(public_class_names(ref1, api_file_pattern) +
                       public_class_names(ref2, api_file_pattern)))
    old_idents = list(set(method_names(ref1, api_file_pattern) +
                        property_names(ref1, api_file_pattern)))
    new_idents = list(set(method_names(ref2, api_file_pattern) +
                        property_names(ref2, api_file_pattern)))
    res: List[AnalyzedDiff] = []
    # analyze all test files
    for file in JDK_DIR.glob(test_file_pattern):
        if file.is_dir():
            continue
        if file.suffix != ".java":
            continue
        diff = get_diff(ref1, ref2, file)
        print(f"======= {file} =======")
        print(diff)
        print("---")
        analyzed = analyze_diff_safe(diff, classes, old_idents, new_idents, special_classes)
        if analyzed is None:
            print("No diff found")
            continue
        print(json.dumps(analyzed.to_json(), indent=2))
        res.append(analyzed)
    print("======= combined =======")
    combined = combine(res)
    print(json.dumps(combined.to_json(), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_classes_parser = subparsers.add_parser("list_classes")
    list_classes_parser.add_argument("ref")
    list_classes_parser.add_argument("file_pattern")
    list_classes_parser.add_argument("--sep", default=", ")
    analyze_diff_parser = subparsers.add_parser("analyze_diff")
    analyze_diff_parser.add_argument("diff_file", type=Path)
    analyze_diff_parser.add_argument("classes", nargs="*")
    analyze_ref_parser = subparsers.add_parser("analyze_ref")
    analyze_ref_parser.add_argument("ref1")
    analyze_ref_parser.add_argument("ref2")
    analyze_ref_parser.add_argument("test_file_pattern")
    analyze_ref_parser.add_argument("api_file_pattern")
    analyze_ref_parser.add_argument("special_classes", default="")
    analyze_panama_parser = subparsers.add_parser("analyze_panama")
    analyze_panama_parser.add_argument("ref1")
    analyze_panama_parser.add_argument("ref2")
    # print help if no arguments are given
    # help should list all commands properly

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    # execute command
    if args.command == "list_classes":
        print(args.sep.join(
            public_classes_names(args.ref, args.file_pattern)))
    elif args.command == "analyze_diff":
        analyze_diff(args.diff_file, args.classes)
    elif args.command == "analyze_ref":
        analyze_ref_diffs(args.ref1, args.ref2,
                          args.test_file_pattern,
                          args.api_file_pattern,
                          args.special_classes.split(","))
    elif args.command == "analyze_panama":
        analyze_ref_diffs(args.ref1, args.ref2,
                          "test/jdk/java/foreign/**/*.java",
                          "src/java.base/share/classes/java/lang/foreign/"
                          "**/*.java", ["MethodHandles", "ConstantBootstraps"])
