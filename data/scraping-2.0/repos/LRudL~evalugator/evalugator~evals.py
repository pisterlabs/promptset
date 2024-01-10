"""
This file deals with "exporting" piles to other eval formats. "Export" is in
quotes because the most common export format is OpenAI Evals, which isn't really
an export as evalugator fully abstracts over it.

This file also contains the functions for actually running evals.
"""


import os
import traceback

from typing import Any, Sequence, Tuple

try:
    from evals.cli.oaieval import run

    # from evals.cli.oaieval import OaiEvalArguments, run
    from evals.registry import Registry
except Exception as e:
    print(f"Warning: evalugator/evals.py could not do imports from OpenAI Evals.")
    traceback.print_exc()

from evalugator.evals_utils import (
    DEFAULT_ARGS,
    OaiEvalArguments,
    get_eval_export_info_path,
    get_eval_info,
    get_eval_log_path,
    get_eval_txt_path,
    get_oaieval_data_path,
)
from evalugator.evals_completers.utils_evals_completers import (
    copy_completer_specs_to_registry,
    special_env_vars,
)
from evalugator.saving import (
    save_struct,
)
from evalugator.structs import (
    ExportSettings,
    TrialExportInfo,
    canonical_trial_sort_key,
    trial_export_info_dump,
)
from evalugator.utils import (
    clean_str_for_path,
    set_temporary_env_vars,
    sort_like,
    write_to_gzip,
    write_to_jsonl,
)
from evalugator.vars import (
    EVAL_LOG_DIR,
    EVAL_REGISTRY_DIR,
)
from evalugator.batches import load_trial_batch


def pile_to_export_format(
    format: str, export_settings: ExportSettings
) -> Tuple[Sequence[Any], Sequence[TrialExportInfo]]:
    """
    This function does the core of the exporting logic. Based on an export
    format (e.g. OpenAI Evals), and an ExportSettings object, it returns:
    (1) a sequence of export data objects, whatever that may look like for a
    particular export format (for each type of Trial, there is a TrialBatch that
    contains the relevant export logic -- see batches.py)
    (2) a sequence of TrialExportInfo objects, which are used to help piece
    together the eval results in a readable format (containing things like
    answer information that are not included in the raw trial export data)
    """
    pile = export_settings.pile
    export_datas = []
    trial_export_infos = []
    trials = []
    for batch_name in pile.batches:
        trial_batch = load_trial_batch(batch_name)
        export_data, new_export_infos = trial_batch.export_to(format, export_settings)
        export_datas.extend(export_data)
        trial_export_infos.extend(new_export_infos)
        trials.extend(trial_batch.data.trials)
    assert len(trials) == len(export_datas) == len(trial_export_infos)
    # Need to ensure canonical sort order for pairing up with results in eval
    # result generation:
    export_datas, trial_export_infos = sort_like(
        list(map(canonical_trial_sort_key, trials)), [export_datas, trial_export_infos]
    )
    for i, export_info in enumerate(trial_export_infos):
        export_info.i = i
    return export_datas, trial_export_infos


def export_to_txt(export_settings: ExportSettings, verbose=True):
    txts, export_infos = pile_to_export_format("txt", export_settings)
    txt = "".join(txts)
    path = get_eval_txt_path(export_settings.name)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(txt)
    if verbose:
        print(f"Wrote readable text of eval {export_settings.name} to {path}")
    return path


def export_to_oaieval(export_settings: ExportSettings, verbose=True):
    name = export_settings.name
    save_struct(get_eval_info(export_settings))
    formatted_dataset, export_infos = pile_to_export_format("oaieval", export_settings)
    export_infos = [trial_export_info_dump(einfo) for einfo in export_infos]
    export_settings.eval_type.write_spec(verbose=verbose)
    path = get_oaieval_data_path(name)
    write_to_jsonl(formatted_dataset, path)
    write_to_jsonl(export_infos, get_eval_export_info_path(name))
    # and now we do this ridiculous hack due to
    # https://github.com/openai/evals/issues/1099 :
    write_to_gzip(formatted_dataset, f"{path}.gz")
    return path


def run_eval(eval_name: str, completer_name: str, **kwargs):
    """This will call OpenAI Evals to run an eval based on the name of an export
    and a completer (OpenAI API model, or something defined in
    evalugator.evals_completers, or, if running with model-graded evals, the
    completer_name can be a comma-separated string of length 2, where the last
    one is used by OAI Evals to run the model-grading, and the first one to
    generate the actual answers."""
    eval_name = clean_str_for_path(eval_name)
    if not os.path.exists(EVAL_REGISTRY_DIR):
        os.makedirs(EVAL_REGISTRY_DIR)
    if not os.path.exists(EVAL_LOG_DIR):
        os.makedirs(EVAL_LOG_DIR)
    # This copies over the custom completer functions (anything non-OpenAI API,
    # e.g. Claude) to the registry location so that the evals framework can find
    # them:
    copy_completer_specs_to_registry()
    oai_eval_args = OaiEvalArguments(**DEFAULT_ARGS)
    oai_eval_args.completion_fn = completer_name
    oai_eval_args.eval = eval_name
    oai_eval_args.record_path = get_eval_log_path(eval_name, completer_name)
    oai_eval_args.extra_eval_params = (
        f"samples_jsonl={get_oaieval_data_path(eval_name)}.gz"
    )
    # wait, why .gz? see https://github.com/openai/evals/issues/1099 :) :) :)
    for kwarg, value in kwargs.items():
        setattr(oai_eval_args, kwarg, value)
    registry = Registry([EVAL_REGISTRY_DIR])
    with set_temporary_env_vars(*special_env_vars(completer_name)):
        run_id = run(oai_eval_args, registry=registry)
    return run_id, oai_eval_args.record_path
