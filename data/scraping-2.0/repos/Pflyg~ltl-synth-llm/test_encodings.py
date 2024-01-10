from benchmark import Benchmark
import prompting
import bosy
import syfco
from functools import cache
import functools
from frozendict import frozendict
import os
import subprocess
import json
import verify
import csv
import asyncio
from tempfile import NamedTemporaryFile

from aigertoverilog import aiger_to_verilog
from utils import *

import vertexai  # to interact with googles code chatbot ai
from vertexai.preview.language_models import ChatModel
import google
import time

import openai
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.
openai.api_key = os.getenv("OPENAI_KEY")

vertexai.init(project="rg-finkbeiner-30141001", location="us-central1")
chat_model = ChatModel.from_pretrained("chat-bison")
parameters = {"temperature": 0.5, "max_output_tokens": 1024}

# We are testing here which verilog encoding of the bosy solutions works best
# We will be testing three different varieties: the native translation from BoSyBackend,
# directly translating the aag output as well as first optimizing the aag through yosys


# https://stackoverflow.com/questions/6358481/using-functools-lru-cache-with-dictionary-arguments


def freezeargs(func):
    """Transform mutable dictionnary
    Into immutable
    Useful to be compatible with cache
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple(
            [frozendict(arg) if isinstance(arg, dict) else arg for arg in args]
        )
        kwargs = {
            k: frozendict(v) if isinstance(v, dict) else v for k, v in kwargs.items()
        }
        return func(*args, **kwargs)

    return wrapped


@freezeargs
@cache
def synthesize(spec, overwrite_params, module_name="fsm", target="verilog"):
    return bosy.synthesize(
        spec=spec,
        overwrite_params=overwrite_params,
        target=target,
        module_name=module_name,
        timeout=240,
    )


def bosy_verilog_standard(spec: str, overwrite_params: dict, module_name: str):
    return synthesize(
        spec=spec, overwrite_params=overwrite_params, module_name=module_name
    )


def bosy_verilog_aag(spec: str, overwrite_params: dict, module_name: str):
    try:
        aag = synthesize(spec=spec, target="aiger", overwrite_params=overwrite_params)
        return aiger_to_verilog(aag, module_name=module_name)
    except AttributeError:
        print("aag", aag, module_name, overwrite_params)
        return ""
    except bosy.BosyError:
        print("aagb", spec, overwrite_params)
        return


def bosy_verilog_opt_aag(spec: str, overwrite_params: dict, module_name: str):
    aag = synthesize(spec=spec, target="aiger", overwrite_params=overwrite_params)
    f = NamedTemporaryFile("w", delete=False, encoding="utf-8")
    f.file.write(aag)
    name = f.name
    f.file.close()
    subprocess.run(
        args=f'yosys -qq -f aiger -p "opt -fine; write_aiger -symbols -ascii {name}" {name}',
        shell=True,
    )
    f = open(name, "r", encoding="utf-8")
    aag = f.read()
    f.close()
    os.remove(name)
    try:
        return aiger_to_verilog(aag, module_name=module_name)
    except AttributeError:
        print("optaag", aag, module_name, overwrite_params)
        return ""


def bosy_verilog_opt_verilog(spec: str, overwrite_params: dict, module_name: str):
    code = synthesize(
        spec=spec,
        target="verilog",
        overwrite_params=overwrite_params,
        module_name=module_name,
    )
    f = NamedTemporaryFile("w", delete=False, encoding="utf-8")
    f.file.write(code)
    name = f.name
    f.file.close()
    subprocess.run(
        args=f'yosys -qq -f verilog -p "synth; opt -fine; aigmap; write_aiger -symbols -ascii {name}" {name}',
        shell=True,
    )
    f = open(name, "r", encoding="utf-8")
    aag = f.read()
    f.close()
    os.remove(name)
    try:
        return aiger_to_verilog(aag, module_name=module_name)
    except AttributeError:
        print("optveri", code, aag, module_name, overwrite_params)
        return ""


def build_prompt_bosy(
    bm: Benchmark,
    template=prompting.DefaultPromptTemplate,
    synth=bosy_verilog_standard,
    params: dict = {},
):
    prompt = template()
    spec = read_file(bm.specification)
    for impl in bm.implementations:
        prompt.add_example(
            {
                "SPEC": syfco.convert(spec, "ltl", overwrite_params=impl["params"]),
                "IMPL": synth(
                    spec=spec, overwrite_params=impl["params"], module_name=bm.name
                ),
                "PARAMS": " and ".join(
                    [k + "=" + str(v) for (k, v) in impl["params"].items()]
                ),
            }
        )
    return prompt.build_prompt(
        {"SPEC": syfco.convert(spec, "ltl", overwrite_params=params)}
    )


def run_single_bosy(bm, type):
    gens = {
        "aag": bosy_verilog_aag,
        "opt_aag": bosy_verilog_opt_aag,
        "opt_verilog": bosy_verilog_opt_verilog,
        "standard": bosy_verilog_standard,
    }
    # bm.build_prompt might do some heavy work like synthesizing examples with bosy
    prompt = build_prompt_bosy(bm, params=bm.generate_params, synth=gens[type])
    # print(prompt)
    chat = chat_model.start_chat()
    response = chat.send_message(prompt, **parameters)
    code = extract_normalized_verilog_code(response.text, bm.name)
    # print(code if code else "NO_CODE::\n" + response.text)
    if code == None:
        return "NO_CODE"
    else:
        res = verify.verify_code(
            bm.specification, code, overwrite_params=bm.generate_params
        )
        return res.name


def run_single_openai(bm, type):
    gens = {
        "aag": bosy_verilog_aag,
        "opt_aag": bosy_verilog_opt_aag,
        "opt_verilog": bosy_verilog_opt_verilog,
        "standard": bosy_verilog_standard,
    }
    # bm.build_prompt might do some heavy work like synthesizing examples with bosy
    messages = build_prompt_bosy(
        bm, params=bm.generate_params, synth=gens[type], template=prompting.PromptOpenAI
    )
    # print(prompt)
    count = 0
    while True:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=messages,
                temperature=0.8,
                # presence_penalty = -0.5
            )
            break
        except (
            openai.error.RateLimitError,
            openai.error.APIError,
            openai.error.APIConnectionError,
            openai.error.ServiceUnavailableError,
            openai.error.Timeout,
            openai.error.TryAgain,
        ):  # OpenAI API seems to be quite unstable
            print("trying again " + str(count))
            time.sleep(60)
            count = count + 1
            if count > 5:
                print(bm)
                print("wtf")
                return "AI_RATELIMIT"

    response = completion.choices[0].message.content
    print(response)
    code = extract_normalized_verilog_code(response, bm.name)
    if code == None:
        return "NO_CODE"
    else:
        res = verify.verify_code(
            bm.specification, code, overwrite_params=bm.generate_params
        )
        return res.name


benchmarks = [
    Benchmark(bm, "../../verilog/")
    for bm in json.loads(read_file("benchmarks_bosy.json"))
]
bm = benchmarks[0]
spec = read_file(bm.specification)
params = bm.implementations[0]["params"]
module_name = bm.name

# print(build_prompt_bosy(bm, synth=bosy_verilog_standard))

gens = [
    bosy_verilog_aag,
    bosy_verilog_opt_aag,
    bosy_verilog_opt_verilog,
    bosy_verilog_standard,
]


def run_benchmarks(
    benchmarks,
    file,
    example_types=["self", "bosy", "strix", "none"],
    run_single=run_single_bosy,
):
    # setting up csv writing
    f = open(file, "w", newline="")
    csvwriter = csv.DictWriter(
        f,
        fieldnames=["benchmark"] + example_types,
        dialect="unix",
        quoting=csv.QUOTE_NONE,
    )
    csvwriter.writeheader()

    # get event loop to be able to run the requests in parallel
    loop = asyncio.get_event_loop()

    def _run_single(bm, type):
        try:
            return run_single(bm, type)
        except TimeoutError:
            return "TIMEOUT"
        except (
            google.api_core.exceptions.InternalServerError,
            openai.error.InvalidRequestError,
        ):
            return "AI_ERROR"
        # except Exception as e:
        #  print(e, e.__class__)

    def run_single_benchmark(bm):
        result = {"benchmark": bm.name}
        async_res = [_run_single(bm, type) for type in example_types]
        for i, res in enumerate(async_res):
            result[example_types[i]] = res
        csvwriter.writerow(result)

    for bm in benchmarks:
        run_single_benchmark(bm)

    f.close()


run_benchmarks(
    benchmarks,
    run_single=run_single_openai,
    example_types=["aag", "opt_aag", "opt_verilog", "standard"],
    file="bosy_compare_openai_2.csv",
)
