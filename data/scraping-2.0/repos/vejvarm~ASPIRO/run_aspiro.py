import argparse
import json
import logging
import pathlib
import statistics
import time

from langchain.schema import OutputParserException
from tqdm import tqdm

from error_analysis import analyze_and_save_errors
from flags import (LOG_ROOT, TemplateErrors, ERROR_MESSAGES, ModelChoices,
                   RDF_EXAMPLE_FILE_NAME, BACKUP_TEMPLATE, DATA_ROOT)
from helpers import setup_logger, load_examples, load_and_validate_config
from models import NShotGenerator, LLMBuilder
from parsing import (build_output_dict, prepare_prompt_and_parser, MultiRetryParser, ConsistencyValidator, TextOutputParser)

LOGFILE_PATH = LOG_ROOT.joinpath(pathlib.Path(__file__).name.removesuffix(".py")+".log")
LOGGER = setup_logger(__name__, loglevel=logging.WARNING, output_log_file=LOGFILE_PATH)


def dehalucinate(dc: ConsistencyValidator, text: str, metadata: dict, keep_result_with_better_score):
    text = dc.run(text, metadata, keep_result_with_better_score)
    return text


# debugging
START_FROM, BREAK_AFTER = (None, None)   # @debug param: start from example num. `START_FROM` and stop at num. `BREAK_AFTER`


def _to_results(result_entry_dict: dict, final_output_dict: dict, intermediate_result_file: pathlib.Path):
    with intermediate_result_file.open("a") as f_intermediate_result:
        f_intermediate_result.write(json.dumps(result_entry_dict)+"\n")
    final_output_dict.update(result_entry_dict)


def main(args):

    config = load_and_validate_config(args.config)
    n_runs = config["n_runs"]  # @param
    dataset_choice = config["dataset_choice"]  # @param
    template_file = config["initial_template"]  # @param
    model_choices = config["llm_stack"]  # @param
    load_in_8bit = config["load_in_8bit"] if "load_in_8bit" in config.keys() else False
    load_in_4bit = config["load_in_4bit"] if "load_in_4bit" in config.keys() else False
    max_retry_shots = config["max_retry_shots"] if "max_retry_shots" in config.keys() else len(model_choices) - 1  # @param (0 ... zero-shot, 1 ... one-shot, ....)
    consist_val_model = config["consist_val_model"]  # @param  (if None, don't use dehalucination)
    example_format = config["example_format"]  # @param
    max_fetched_examples_per_pid = config["max_fetched_examples_per_pid"]  # @param
    error_dump_subfolder = config["error_dump_subfolder"]  # @param (if None, errors are not saved)
    use_backup = config["use_backup"]  # @param (to change backup template, change flags.BACKUP_TEMPLATE)

    # GENERAL MODEL HYPERPARAMS
    # max_tokens_to_generate = config["max_tokens_to_generate"]  # @param
    # temperature = config["temperature"]  # @param
    # stop_sequences = config["stop_sequences"]  # @param

    # Consistency Validation HYPERPARAMS
    cv_metric = ConsistencyValidator.Metrics.PARENT
    cv_threshold = config["cv_threshold"]  # about 157 prompts in the original
    cv_template = config["cv_template"]
    cv_keep_better = config["cv_keep_better"]  # @param
    cv_load_in_8bit = config["cv_load_in_8bit"] if "cv_load_in_8bit" in config.keys() else False
    cv_load_in_4bit = config["cv_load_in_4bit"] if "cv_load_in_4bit" in config.keys() else False

    dataset_folder = dataset_choice.value
    output_folder = pathlib.Path(args.output).joinpath(dataset_folder.relative_to(DATA_ROOT))

    # INITIALIZE PROMPT and PARSER
    prompt, output_parser = prepare_prompt_and_parser(template_file, example_format)
    # LOGGER.warning(prompt.format(examples="hi\nyou"))

    # Load pids and rdfs examples from path_to_fetched_example_json
    path_to_fetched_example_json = dataset_folder.joinpath(RDF_EXAMPLE_FILE_NAME)
    pid_examples_dict = load_examples(path_to_fetched_example_json, max_fetched_examples_per_pid, example_format,
                                      dataset_choice)

    run = 0
    runs_left = n_runs
    elapsed_time = {}
    retry_models = ",".join([mch.name for mch in model_choices[1:]]) if (max_retry_shots > 0
                                                                         and len(model_choices) > 1) else "NONE"
    path_to_experiment_folder = output_folder.joinpath(f"{template_file.name}").joinpath(
        f"{model_choices[0].name}({retry_models})({max_retry_shots}shot)({consist_val_model.name or 'NONE'})"
        ).joinpath(f"max_examples{max_fetched_examples_per_pid}")

    while runs_left > 0:
        # Define files for results
        path_to_output_template_json = path_to_experiment_folder.joinpath(f"run{run:02d}").joinpath(
            f"templates-{dataset_choice.name.lower()}_{template_file.name}.json")
        if path_to_output_template_json.parent.exists():
            print(f"RUN NUMBER: {run} (EXISTS)")
            run += 1
            continue

        # INITIALIZE LANGCHAIN with specified `model_choices` and `prompt`
        llm_builder = LLMBuilder()
        llm_builder.initialize_chains(model_choices, prompt, template_file,
                                      # max_tokens_to_generate=max_tokens_to_generate,
                                      # temperature=temperature,
                                      # stop_sequences=stop_sequences,
                                      # load_in_8bit=load_in_8bit,
                                      # load_in_4bit=load_in_4bit,
                                      **config)
        llms = llm_builder.llms
        llm_chains = llm_builder.chains
        retry_parser = MultiRetryParser.from_llms(parser=output_parser, llms=llms)

        # INITIALIZE dehalucination chain class
        dc_prompt_version = f"_{cv_template.name.lower()}"
        consistency_validator_log = LOG_ROOT.joinpath(dataset_choice.name).joinpath(template_file.name).joinpath(
            f"{','.join(m.name for m in model_choices)}({max_retry_shots}shot)({consist_val_model.name}){dc_prompt_version}.jsonl")  # @param

        if consist_val_model.value is not None:
            consistency_validator_log.parent.mkdir(parents=True, exist_ok=True)
            prompt_template = cv_template.value.open().read()
            prompt_metadata = json.load(cv_template.value.with_suffix(".json").open())
            # TODO make work for versions below v4
            dc = ConsistencyValidator(cv_metric, cv_threshold, llm_builder, consist_val_model, prompt_template,
                                      source_data_key=prompt_metadata["source_data_key"],
                                      first_key=prompt_metadata["first_key"], output_key=prompt_metadata["output_key"],
                                      stop=prompt_metadata["stop"],
                                      path_to_jsonl_results_file=consistency_validator_log,
                                      load_in_8bit=cv_load_in_8bit, load_in_4bit=cv_load_in_4bit)
        else:
            dc = None

        # run start
        start_time = time.time()
        runs_left -= 1
        path_to_output_template_json.parent.mkdir(parents=True, exist_ok=True)
        print(f"RUN NUMBER: {run} (left: {runs_left})")

        output_pid_template_dict = {}
        intermediate_result_file = path_to_output_template_json.with_suffix(".jsonl")
        k = 0
        while intermediate_result_file.exists():
            k += 1
            LOGGER.warning(f"(k={k}):\n\t"
                           f"The intermediate results file already exists at path: {intermediate_result_file}")
            intermediate_result_file = intermediate_result_file.with_stem(f"{intermediate_result_file.stem}({k})")
        backup_count = 0
        for i, (pid, example) in tqdm(enumerate(pid_examples_dict.items()), total=len(list(pid_examples_dict.keys()))):
            # prepare input
            rdf_example, subj_labs, rel_labs, obj_labs = example
            unique_rel_labs = list(set(rel_labs))
            if len(unique_rel_labs) == 1:
                rel_lab = unique_rel_labs[0]
            else:
                raise NotImplementedError("Example structures must have only 1 unique relation in all their entries")
            inp = {"examples": rdf_example}
            if "subjects" in prompt.input_variables:
                inp["subjects"] = subj_labs
            if "relation" in prompt.input_variables:
                inp["relation"] = rel_lab
            if "objects" in prompt.input_variables:
                inp["objects"] = obj_labs

            # debugging purposes
            if START_FROM is not None and i < START_FROM:
                for mdl in model_choices:
                    if isinstance(mdl.value, pathlib.Path):
                        _ = llm_chains[0].run(inp)
                continue

            # debugging purposes
            if BREAK_AFTER is not None and i == BREAK_AFTER:
                break

            if not rdf_example:
                err = ERROR_MESSAGES[TemplateErrors.NA]
                LOGGER.warning(f"({pid}) {TemplateErrors.NA.value}: {err}']")
                out_dict = {pid: build_output_dict("", [TemplateErrors.NA.value], [err], rdf_example, subj_labs, obj_labs)}
                _to_results(out_dict, output_pid_template_dict, intermediate_result_file)
                continue

            metadata = {"data": rdf_example, "reference": rel_lab, "relation_label": rel_lab,
                        "rdf_example": rdf_example, "subj_labels": subj_labs, "obj_labels": obj_labs}

            # Zero-shot
            try:
                answer = llm_chains[0].run(inp)
            except Exception as err:
                LOGGER.warning(f"({pid}) {TemplateErrors.API.value}: {err}.")
                out_dict = {pid: build_output_dict("", [TemplateErrors.API.value], [repr(err)],
                                                                        rdf_example, subj_labs, obj_labs)}
                _to_results(out_dict, output_pid_template_dict, intermediate_result_file)
                continue

            # parse the answer
            shot = 0
            try:
                # TODO: change to Retry prompt STACK (change prompt version with each shot)
                shot, output_dict = retry_parser.parse_with_prompt(answer, prompt.format_prompt(**inp),
                                                                    shot=shot, max_shots=max_retry_shots, metadata=metadata)
            except OutputParserException as err:
                LOGGER.info(f'({pid}) {TemplateErrors.PARSING.value}: {err}')
                shot = max_retry_shots
                output_dict = json.loads(str(err))

            if use_backup:
                if not ("<subject>" in output_dict["output"] and "<object>" in output_dict["output"]):
                    output_dict["output"] = BACKUP_TEMPLATE.format("<subject>", rel_lab, "<object>")
                    backup_count += 1

            output_dict = build_output_dict(output=output_dict["output"],
                                            error_codes=output_dict["error_codes"],
                                            error_messages=output_dict["error_messages"],
                                            rdf_example=rdf_example,
                                            subj_labels=subj_labs, obj_labels=obj_labs, shot=shot)
            # dehalucinate
            if dc is not None:
                text = dehalucinate(dc, output_dict["output"], metadata, cv_keep_better)
                output_dict["output"] = text

            final_templates = {pid: output_dict}
            _to_results(final_templates, output_pid_template_dict, intermediate_result_file)

        json.dump(output_pid_template_dict, path_to_output_template_json.open("w"), indent=2)
        print(f"Output saved into {path_to_output_template_json}")

        if error_dump_subfolder is not None:
            _folder_to_dump_error_jsons = path_to_output_template_json.parent.joinpath(error_dump_subfolder)
            _folder_to_dump_error_jsons.mkdir(parents=True, exist_ok=True)
            analyze_and_save_errors(path_to_output_template_json, _folder_to_dump_error_jsons, parser=TextOutputParser())
            err_counts_file = _folder_to_dump_error_jsons.joinpath("errCOUNTS.json")
            err_counts_dict = json.load(err_counts_file.open("r"))
            err_counts_dict["BACKUPS"] = backup_count
            json.dump(err_counts_dict, err_counts_file.open("w"), indent=2)
            print(f"Error analysis saved into: {_folder_to_dump_error_jsons}")

        elapsed_time[str(run)] = time.time() - start_time
        print(f"time taken: {elapsed_time[str(run)]:.3f} seconds")

    # save run times to json file
    path_to_runtime_json = path_to_experiment_folder.joinpath("runtime_seconds.json")
    if path_to_runtime_json.exists():
        old_el_time = json.load(path_to_runtime_json.open("r"))
        old_el_time.update(elapsed_time)
        elapsed_time = old_el_time
    elapsed_time["mean"] = statistics.mean(elapsed_time.values())
    json.dump(elapsed_time, path_to_runtime_json.open("w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process the path to the configuration file.')
    parser.add_argument('--config', type=str, default="setups/json_default.json", help='Path to the configuration file.')
    parser.add_argument('--output', type=str, default="data", help='Path to output folder.')
    main(parser.parse_args())