import json
from typing import Tuple

import openai

from wmb_browser.backend.cemba_cell import cemba_cell

categorical_variables = [
    "CCFRegionAcronym",
    "CCFRegionBroadName",
    "CEMBARegion",
    "CellClass",
    "CellGroup",
    "CellSubClass",
    "DissectionRegion",
    "MajorRegion",
    "Sample",
    "SubRegion",
    "Technology",
]

continuous_variables = [
    "Slice",
    "PlateNormCov",
    "FinalmCReads",
    "InputReads",
    "GlobalOverallmCCCFrac",
    "GlobalOverallmCGFrac",
    "GlobalOverallmCHFrac",
]

modalities = [
    "ImputeChrom100KMatrix",
    "ImputeChrom10KMatrix",
    "RawChrom100KMatrix",
    "mCHFrac",
    "mCGFrac",
    "ATAC",
    "DomainBoundaryProba",
    "CompartmentScore",
]
modalities_1d = [
    "mCHFrac",
    "mCGFrac",
    "ATAC",
    "DomainBoundaryProba",
    "CompartmentScore",
]
modalities_2d = ["ImputeChrom100KMatrix", "ImputeChrom10KMatrix", "RawChrom10KMatrix"]

alias = {
    "GlobalOverallmCCCFrac": "mCCCFrac",
    "GlobalOverallmCGFrac": "mCGFrac",
    "GlobalOverallmCHFrac": "mCHFrac",
    "CCFRegionAcronym": "CCF_acronym",
    "CCFRegionBroadName": "CCF_broad",
    "CellCluster": "CellGroup",
    "SubClass": "CellSubClass",
    "ImputeChrom100KMatrix": "Impute 100K",
    "ImputeChrom10KMatrix": "Impute 10K",
    "RawChrom100KMatrix": "Raw 100K",
    "ATAC": "ATAC CPM",
    "mCHFrac": "mCH Frac",
    "mCGFrac": "mCG Frac",
    "DomainBoundaryProba": "Domain Boundary",
    "CompartmentScore": "Compartment Score",
}

# make alias key case insensitive
alias = {k.lower(): v for k, v in alias.items()}

FUNCTIONS = [
    {
        "name": "make_cell_scatter_plot",
        "description": (
            "Making tsne or umap scatter plot color by categorical or continous variable on named coordinates."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "coord": {
                    "type": "string",
                    "description": (
                        "The coords name can be any one of these regex: "
                        "'mc_all_(tsne|umap)', '\w+_mr_(tsne|umap)', 'slice\d+_merfish'. "
                        "The 'mc_all_(tsne|umap)' stand for a global coords for the entire dataset; "
                        "The '\w+_mr_(tsne|umap)' stand for major brain region coords, including these brain regions: "
                        f"({cemba_cell.get_metadata('MajorRegion').cat.categories.tolist()}); "
                        "the 'slice\d+_merfish' stand for MERFISH spatial coords for cornoal brain slices."
                    ),
                    "default": (
                        "If no coords provided, use 'mc_all_tsne'; if not clear about which major region coords, "
                        "use 'HPF_mr_tsne'; if not clear about which merfish MERFISH coords, use 'slice59_merfish'"
                    ),
                },
                "color": {
                    "type": "string",
                    "description": (
                        "A variable name for scatter color. "
                        f"Categorical names: {categorical_variables}; "
                        f"Continuous names: {continuous_variables}; "
                        "Continuous variable can also be in the form of "
                        "VALUE_TYPE:GENE_NAME, for example 'mch:Gad1', 'mcg:Foxp2', 'rna:Rorb'. "
                        "mch stands for gene mCH fraction; mcg stands for gene mCG fraction; "
                        "rna stands for gene expression."
                    ),
                    "default": (
                        "If user isn't clear about color, use the 'CCFRegionAcronym' for MERFISH coords, "
                        "and use 'CellSubclass' for other coords"
                    ),
                },
                "scatter_type": {
                    "type": "string",
                    "description": (
                        "Determine the type of coloring variable. This can be infered from the color parameter."
                    ),
                    "enum": ["continuous", "categorical"],
                },
            },
            "required": ["scatter_type", "color", "coord"],
        },
    },
    {
        "name": "higlass_browser",
        "description": "Making a cell type HiGlass browser. Each browser can take one or two or multiple cell types.",
        "parameters": {
            "type": "object",
            "properties": {
                "cell_types": {
                    "type": "array",
                    "description": "A list of cell types to be plotted in the browser.",
                    "items": {
                        "type": "string",
                        "descriptions": (
                            "Cell types names are short terms of "
                            f"{cemba_cell.get_metadata('CellSubClass').cat.categories[[0, 32, 50, 80, 100, 150]]}"
                        ),
                    },
                    "default": ["CA3 Glut", "Sst Gaba"],
                },
                "modalities": {
                    "type": "array",
                    "description": "A list of modalities to be plotted in the browser.",
                    "items": {
                        "type": "string",
                        "enum": modalities,
                    },
                },
                "browser_type": {
                    "type": "string",
                    "description": (
                        "The type of the browser to be plotted. "
                        "The multi_cell_type_1d or _2d browser can fit in multiple cell types. "
                        "The two_cell_type_diff browser is for comparing the track "
                        "difference between two cell types. "
                        "The loop_zoom_in browser is for the large-scale and "
                        "zoom-in view of a single cell type."
                    ),
                    "enum": ["multi_cell_type_1d", "multi_cell_type_2d", "two_cell_type_diff", "loop_zoom_in"],
                    "default": "multi_cell_type_2d",
                },
                "region": {
                    "type": "string",
                    "description": (
                        "The genome region of the browser, can be CHROM:START-END or a gene name. "
                        "For example: chr1:2000000-2100000 or Gad1"
                    ),
                    "default": "Gad1",
                },
            },
            "required": ["cell_types", "browser_type", "region"],
        },
    },
]


def parse_user_input(user_input: str) -> Tuple[str, str, dict]:
    """Parse user input and return dataset, plot_type, and kwargs."""
    messages = [{"role": "user", "content": user_input}]
    functions = FUNCTIONS
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )

    response_message = response["choices"][0]["message"]

    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        # function_name = response_message["function_call"]["name"]
        try:
            _func_call = response_message["function_call"]
            function_name = _func_call["name"]
            function_args = json.loads(_func_call["arguments"])
        except (json.JSONDecodeError, KeyError):
            function_args = None
            function_name = None
        return function_name, function_args, response
    else:
        return None, None, response


def _check_required_key(response, required_key, func_args):
    for key in required_key:
        if key not in func_args:
            print(response)
            raise KeyError(f"GPT func_args do not have required key {key}")
    return


def _alias_to_real_value(func_args):
    real_func_args = func_args.copy()
    for key, value in func_args.items():
        if isinstance(value, str):
            real_func_args[key] = alias.get(value.lower(), value)
        elif isinstance(value, list):
            real_func_args[key] = [alias.get(v.lower(), v) for v in value]
        else:
            pass
    return real_func_args


def gpt_response_to_function_call(func_name, func_args, gpt_response):
    if func_name is None:
        print(gpt_response)
        raise ValueError("GPT failed to all functions")
    elif func_name == "make_cell_scatter_plot":
        # call scatter plot function
        dataset = "cemba_cell"

        required_key = ["scatter_type", "color", "coord"]
        _check_required_key(gpt_response, required_key, func_args)

        # dealwith color
        color = func_args["color"]
        if color.startswith("mch:") or color.startswith("mcg:") or color.startswith("rna:"):
            func_args["color"] = "gene_" + color

        _type = func_args.pop("scatter_type")
        plot_type = f"{_type}_scatter"

        func_args = _alias_to_real_value(func_args)
        return dataset, plot_type, func_args
    elif func_name == "higlass_browser":
        # call higlass browser function
        dataset = "higlass"

        required_key = ["cell_types", "browser_type"]
        _check_required_key(gpt_response, required_key, func_args)

        plot_type = func_args.pop("browser_type")

        # separate 1D and 2D modalities
        modalities = func_args.pop("modalities", [])
        _m1d = [m for m in modalities if m in modalities_1d]
        _m2d = [m for m in modalities if m in modalities_2d]
        if len(_m1d) == 0:
            _m1d = None
        if len(_m2d) == 0:
            _m2d = None
        else:
            _m2d = _m2d[0]
        if plot_type == "multi_cell_type_1d":
            func_args["modalities"] = _m1d
        else:
            func_args["modality_1d"] = _m1d
            func_args["modality_2d"] = _m2d

        # dealwith cell types
        if plot_type == "two_cell_type_diff":
            ct1, ct2, *_ = func_args.pop("cell_types")
            func_args["cell_type_1"] = ct1
            func_args["cell_type_2"] = ct2
        elif plot_type == "loop_zoom_in":
            ct, *_ = func_args.pop("cell_types")
            func_args["cell_type"] = ct
        else:
            pass

        # dealwith region
        region = func_args.pop("region", "Gad1")
        if plot_type == "multi_cell_type_1d":
            func_args["region"] = region
        else:
            func_args["region1"] = region

        func_args = _alias_to_real_value(func_args)
        return dataset, plot_type, func_args
    else:
        print(gpt_response)
        raise ValueError(f"GPT function {func_name} is not supported")


def chatgpt_string_to_args_and_kwargs(string):
    func_name, func_args, gpt_response = parse_user_input(string)
    dataset, plot_type, kwargs = gpt_response_to_function_call(func_name, func_args, gpt_response)
    return dataset, plot_type, [], kwargs
