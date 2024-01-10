import json
import openai
import os
from dataset_builder.data_sample_generators import generate_data_sample_from
from description_builder.description_generator import generate_description_from
from snippet_builder.snippet_generator import generate_snippet_from
from util.argparser import parse_arguments
from util.json_utils import write_jsonl
from util.print_utils import pretty_print_conversation
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    """
    modes:
        snippet
        Run function task -> snippets.

        description
        Run function snippet -> description.

        sample
        Run function description -> training sample.
    """

    openai.api_key = os.getenv("OPENAI_API_KEY", "")

    args = parse_arguments()
    input_file_name = args.input_file_name
    output_file_name = args.output_file_name
    n_samples = args.n_samples
    temperature = args.temperature
    model = args.model
    mode = args.mode

    data = []
    if os.path.exists(input_file_name):
        with open(input_file_name, "r") as json_file:
            # prefer readable json
            content = ""
            for line in json_file:
                content += line.strip()
                if line.startswith("}"):  # end of a JSON object
                    data.append(json.loads(content))
                    content = ""
    else:
        raise FileNotFoundError(
            f"File with name {input_file_name} does not exist!")

    results = []
    if os.path.exists(output_file_name):
        with open(output_file_name, "r") as json_file:
            content = ""
            for line in json_file:
                content += line.strip()
                if line.startswith("}"):  # end of a JSON object
                    results.append(json.loads(content))
                    content = ""

    match mode:
        case "snippet" | "snippets":
            input_key = "task"
            output_key = "snippet"
            generate_function = generate_snippet_from
        case "description" | "descriptions":
            input_key = "snippet"
            output_key = "description"
            generate_function = generate_description_from
        case "sample" | "samples":
            input_key = "description"
            output_key = "data"
            generate_function = generate_data_sample_from
        case _:
            raise KeyError(
                "Choose a mode from 'snippet', 'description', or 'sample'!")

    for function in data:
        function_input = function.get(input_key)
        function_id = function.get("function_id")

        if function_input is None:
            raise KeyError(
                f"Function data in {input_file_name} should have field '{input_key}'")

        messages, output, meta = generate_function(
            function_input, model, temperature)
        messages.append(output["message"])
        pretty_print_conversation(messages)

        try:
            completion_dict = json.loads(output.get("message").get("content"))
        except Exception:
            completion_dict = output.get("message").get("content")
            if mode == "description":
                completion_dict = None  # set to null to see which failed
        result = {
            "function_id": function_id,
            input_key: function_input,
            "completion-content": output.get("message").get("content"),
            output_key: completion_dict,
            "metadata": meta,
        }
        results.append(result)
        write_jsonl(output_file_name, results)
