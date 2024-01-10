# -*- coding: utf-8 -*-
""" How it got done: change sys.argv to our setup
    remake ../anthropics_evals/persona/machiavellianism.jsonl to evals/registry/data/macia/macia.jsonl
    run /Users/admin/prog/evals/evals/cli/oaieval.py  ( env keys are set in IDE)
    use DataFrame.to_str() to show output file as table -- needs be tried.
"""
# from copy import copy, deepcopy  # won't work for sys.argv!!!

from scripts.generate_machiavellianism_samples_homegrown_in_brno import generate_machi_fast, write_list_to_jsonl
from scripts.view_result_as_DataFrame import load_result_to_DataFrame

from evals.cli.oaieval import main

def datei():
    import subprocess

    # Execute the 'date -I' command and capture its output
    return subprocess.check_output(['date', '-I']).decode().strip()


if __name__ == "__main__":
    import os
    dirname = os.path.dirname(__file__)

    ## Reformating/migration of Anth. evals:
    # scripts.convert_anthropic_samples.reformat_ant_eval best run from that file.

    ## Or reformatting our own funk:
    # scripts.generate_machiavellianism_samples_homegrown_in_brno.write_list_to_jsonl best run from that file.

    ## Running the generated eval:
    #Todo from sys.exec(..) #with same path?.?
    # os.system("echo 'shell sees key' $OPENAI_API_KEY") # seems ^it could work well

    ### Changing temperature setup in the simplest eval type:
    # from evals.elsuite.basic.includes import Includes
    # print("temper:", Includes.temperature)
    # print("  ->")
    # Includes.temperature = 0.9
    # print("temper:", Includes.temperature)
    # # exit(1)

    # import sys
    # orig_argv = list(sys.argv)
    # ## completion_fn eval --args  #hopefully
    # # model = "text-davinci-003" # AKA completion_fn .. also 001 and 002 can be considered
    # # model = "text-curie-001" # AKA completion_fn
    # # model = "davinci-instruct-beta" # AKA completion_fn .. this one is dumb and says always A
    # # for model in ["text-curie-001", "text-ada-001", "text-davinci-003", "text-babbage-001"]:
    # for model in ["gpt-3.5-turbo", "text-ada-001"]:
    #     sys.argv = orig_argv
    #     eval_name = "machiavellianism"
    #     sample_count = 100
    #     out_path = f"./output/{eval_name}_model_{model}_{sample_count}_samples_{datei()}.jsonl" # = record_path
    #     sys.argv = orig_argv + f"{model} {eval_name} --seed=0 --max_samples={sample_count} --record_path={out_path}".split()
    #     print(f"Starting {sys.argv}")
    #     main()


    ### quick way to export:
    # df = load_result_to_DataFrame("output/macia.dev.v1.out.jsonl")#(out_path)
    # df = load_result_to_DataFrame("output/machi_brno_01.jsonl")
    # print(df.to_string())
    # df.to_clipboard()

    import sys
    # os.environ["OPENAI_API_BASE"]="http://localhost:8000/v1" # or openai.api_base=
    import openai
    openai.api_base="http://localhost:8000/v1"
    orig_argv = list(sys.argv)
    ## completion_fn eval --args  # but maybe needs tweaking..
    for model in ["/Users/admin/Downloads/LLaMA/Llama2/llama-2-7b-chat.ggmlv3.q2_K.bin"]:
        # print(model[model.rfind("/")+1:]);exit()
        model_short_name = model[model.rfind("/")+1:]
        sys.argv = orig_argv
        eval_name = "macia.dev.v0.1"
        sample_count = 5
        out_path = f"./output/{eval_name}_model_{model_short_name}_{sample_count}_samples_{datei()}.jsonl" # = record_path
        sys.argv = orig_argv + f"{model} {eval_name} --seed=0 --max_samples={sample_count} --record_path={out_path}".split()
        print(f"Starting {sys.argv}")
        main()
