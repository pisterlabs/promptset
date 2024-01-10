import json
import openai
import os

from description_builder.description_generator import generate_description_from
from util.argparser import parse_arguments
from util.json_utils import write_jsonl
from util.print_utils import pretty_print_conversation
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    """Run function description -> training samples."""

    openai.api_key = os.getenv("OPENAI_API_KEY", "")

    args = parse_arguments()
    input_file_name = args.input_file_name
    output_file_name = args.output_file_name
    n_samples = args.n_samples
    temperature = args.temperature
    model = args.model

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
        raise FileNotFoundError(f"File with name {input_file_name} does not exist!")

    results = []
    if os.path.exists(output_file_name):
        with open(output_file_name, "r") as json_file:
            content = ""
            for line in json_file:
                content += line.strip()
                if line.startswith("}"):  # end of a JSON object
                    results.append(json.loads(content))
                    content = ""

    for function in data:
        function_snip = function.get("snippet")
        if function_snip is None:
            raise KeyError(f"Function data in {input_file_name} should have field 'snippet'")

        messages, output, meta = generate_description_from(function_snip, model, temperature)
        messages.append(output["message"])
        pretty_print_conversation(messages)
        
        try:
            completion_dict = json.loads(output.get("message").get("content"))
        except Exception:
            completion_dict = None
        result = {
            "description": function_snip,
            "completion-content": output.get("message").get("content"),
            "data": completion_dict,
            "metadata": meta,
        }
        results.append(result)
        write_jsonl(output_file_name, results)

