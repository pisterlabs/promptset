"""
Pares and process question specifications
"""
import os

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def list_models(code=None):
    if code is None:
        r = openai.FineTune.list()
        return [e["fine_tuned_model"] for e in list(r["data"])]
    else:
        return [e for e in list_models() if code in e]


def last_model(code=None, idx=-1):

    return list_models(code)[idx]


def get_completion(question, model_id=None, all=False):
    if model_id is None:
        model_id = last_model()

    prompt = "question: " + question + " ->"

    try:
        r = openai.Completion.create(
            model=model_id, prompt=prompt, max_tokens=50, stop=" ###", temperature=0
        )
    except Exception as e:
        print(f'ERROR for prompt\n"{prompt}"\n\n', e)
        raise
        return None

    if all:
        return r
    else:
        return r["choices"][0]["text"]


def parse_specification(s):

    s = s.replace(" specification: ", " ")
    s = s.replace("]", "]---")
    d = {}

    for p in [e.strip() for e in s.split("---") if e.strip()]:
        if p.startswith("["):
            d["measure"] = p.strip("[]").strip()
        else:
            try:
                pre, term = [e.strip() for e in p.split(" ", 1)]
                d[pre] = term.strip("[]").strip()
            except Exception as e:
                print("Error", s)
                if "errors" not in d:
                    d["errors"] = []
                d["errors"].append((e, p))
    return d


def parse_specification_dict(e, s=None):
    """Parse a specification into a dictionary"""

    if s is None:
        q, s = e["question"], e["specification"]
    else:
        q = e

    # d = {k:None for k in 'measure of by in during'.split() }
    d = {"question": q}

    d = d.update(parse_specification(s))

    return d


def unparse_specification(r, return_dict=False):
    """Unparse a specification dictionary into a string"""

    try:
        d = r.fillna("").to_dict()
    except AttributeError:
        d = r

    parts = []
    parts.append(f"[{d['measure']}]")

    for part_name in "of for by in during".split():
        if d.get(part_name):
            parts.append(f"{part_name} [{d[part_name]}]")

    s = " ".join(parts)

    if return_dict:
        return {"question": r.question, "specification": s}
    else:
        return s
