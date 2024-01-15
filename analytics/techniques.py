import json


def process_strings(strings):
    strings = set(list(filter(lambda x: len(x) >= 10, strings)))

    vars = {}
    vars["length"] = len(strings)
    for prompt in strings:
        # Find CoT prompts
        if "step-by-step" in prompt.lower():
            vars["CoT"] = vars.get("CoT", 0) + 1
        elif "step by step" in prompt.lower():
            vars["CoT"] = vars.get("CoT", 0) + 1
        elif "thoughts:" in prompt.lower():
            vars["CoT"] = vars.get("CoT", 0) + 1
        elif "thought:" in prompt.lower():
            vars["CoT"] = vars.get("CoT", 0) + 1
        elif "chain of thought" in prompt.lower():
            vars["CoT"] = vars.get("CoT", 0) + 1
        elif "chain-of-thought" in prompt.lower():
            vars["CoT"] = vars.get("CoT", 0) + 1
        elif "let's think" in prompt.lower():
            vars["CoT"] = vars.get("CoT", 0) + 1
        elif "lets think" in prompt.lower():
            vars["CoT"] = vars.get("CoT", 0) + 1

        if "scratchpad" in prompt.lower():
            vars["Scratchpad"] = vars.get("Scratchpad", 0) + 1
        elif "scratch pad" in prompt.lower():
            vars["Scratchpad"] = vars.get("Scratchpad", 0) + 1

        if "tool:" in prompt.lower():
            vars["Tool use"] = vars.get("Tool use", 0) + 1
        elif "tools:" in prompt.lower():
            vars["Tool use"] = vars.get("Tool use", 0) + 1

        if "example:" in prompt.lower():
            vars["Few Shot"] = vars.get("Few Shot", 0) + 1
        elif "examples:" in prompt.lower():
            vars["Few Shot"] = vars.get("Few Shot", 0) + 1
        elif "exemplar" in prompt.lower():
            vars["Few Shot"] = vars.get("Few Shot", 0) + 1

        if "```" in prompt.lower():
            vars["Code Block"] = vars.get("Code Block", 0) + 1

        if "###" in prompt.lower():
            vars["Instruction Block"] = vars.get("Instruction Block", 0) + 1

        if "doc" in prompt.lower():
            vars["doc"] = vars.get("doc", 0) + 1

        if "be concise" in prompt.lower():
            vars["concise"] = vars.get("concise", 0) + 1

        if "<|" in prompt.lower() or "|>" in prompt.lower():
            vars["Special Tokens"] = vars.get("Special Tokens", 0) + 1

    return vars


def print_res(res):
    c_count = "c" * (len(res["a, b, c, d"]) + 2)
    print(
        """\\begin{table*}
\caption{Research Technique Detection}
\label{tab:technique}
\\begin{tabular}{"""
        + c_count
        + "}"
    )
    print("\t\\toprule")
    print("\tSet & Total", end=" ")
    for key in res["a, b, c, d"].keys():
        if key != "length":
            print(f"& {key}", end=" ")
    print("\\\\")

    print("\t\\midrule")
    order = []
    for key, value in res.items():
        length = value.pop("length")
        if order == []:
            order = list(value.keys())

        print(f"\t{key} & {length} ")
        for ord in order:
            if ord not in value:
                value[ord] = 0

            print(f"& {value[ord]} ({value[ord] / length * 100:.1f})", end=" ")
        print("\\\\")
    print("""\t\\bottomrule\n\end{tabular}\n\end{table*}""")


res = {}
with open("strings.json") as f:
    strings = json.load(f)
    res["a, b, c, d"] = process_strings(strings)

with open("strings_plus.json") as f:
    strings = json.load(f)
    res["e"] = process_strings(strings)

with open("dev_gpt_prompts_v2.json") as f:
    strings = json.load(f)
    res["f"] = process_strings(strings)


print_res(res)
