from datasets import Dataset, load_dataset
from benchmark import Benchmark, Ranger
import logging as log
import os


def upload_legalbench() -> None:
    # load dataset from hugginface or local
    dataset = load_dataset(
        "legalbench/definition_classification/",
        "ssla_company_defendents/",  # TODO
        delimiter="\t",
        column_names=["text", "label"],
    )

    legalbench = Benchmark(name="legalbench")

    def my_parser(input, **kwargs):
        baseprompt = kwargs["template"]
        input = kwargs["input"]
        return baseprompt.replace("{{input}}", input)

    legalbench.generate_from_template = my_parser
    legalbench.add_dataset("abc", dataset)

    f = open(
        "legalbench/supply_chain_disclosure_best_practice_certification/base_prompt.txt",
        "r",
    )

    template = f.read()

    legalbench.add_assignment(
        "supply_chain_disclosure_best_practice",
        dataset_name="abc",
        input_col="text",
        output_col="answer",
        template=template,
    )

    Ranger.upload_benchmark(legalbench, "ranger-uploads", "legalbench")


# TODO: save results from a run on AWS and associate with benchmark_run Id.
def save_results_aws():
    return

def generate(template, input):
        return template.replace("{{text}}",input)

def legalbench() -> None:
    model_source = "NousResearch/Nous-Hermes-Llama2-13b"

    ranger: Ranger = Ranger(model_source,"baseten",5,"DZ0KPsrZ.dGcR1CXBTSQSTzH1Vur2GCX5W1kSk7PL","1PylM20")

    custom_benchmark: Benchmark = Benchmark("legalbench")

    # add dataset, assignment and baseprompt for each test tsv and txt in legalbench folder
    for assignment_name in os.listdir("legalbench"):
        if os.path.isdir(f"legalbench/{assignment_name}"):
            for file in os.listdir(f"legalbench/{assignment_name}"):
                if file.endswith("test.tsv"):
                    dataset_name = file
                    custom_benchmark.add_dataset_from_csv(
                        dataset_name,
                        f"legalbench/{assignment_name}/{file}",
                        delimiter="\t",
                    )
                    custom_benchmark.add_assignment(
                        f"{assignment_name}_assignment",
                        dataset_name,
                        "text",
                        "answer",
                    )
                elif file.endswith(".txt"):
                    with open(f"legalbench/{assignment_name}/{file}", "r") as f:
                        template = f.read()
                    for assignment in custom_benchmark.assignments:
                        if assignment.name == f"{assignment_name}_assignment":
                            assignment.template = template
                            assignment.generate_from_template = generate
                            

    ranger.add_benchmark(custom_benchmark)
    ranger.run_benchmark("legalbench")

    for result in ranger.get_results():
        log.info(result)

def openai_evals():
    # TODO import OPENAIEVALS
    from openai_utils import OpenAIEvals
    import json

    evals = OpenAIEvals()
    benchmark: Benchmark = Benchmark("openai_evals")
    input_col = "input"
    output_col = "ideal"

    model_source = "baseten"
    ranger: Ranger = Ranger(
        model_source,
        "baseten",
        500,
        "DZ0KPsrZ.dGcR1CXBTSQSTzH1Vur2GCX5W1kSk7PL",
        "1PylM20",
    )

    # Iterating over all of the assignents and their data
    for i, directory in enumerate(evals.DATA_PATH.iterdir()):
        # for i, directory in enumerate((evals.DATA_PATH / 'sarcasm').iterdir()):
        # XXX: For debugging, delete later
        # {{{ Toggle fold
        if i > 4:
            break
        # }}}

        if not directory.is_dir():
            log.debug(f"Weird file: {directory} in {evals.DATA_PATH}")
            continue
        name = directory.absolute().name  # basename of the file
        dataset_name = f"{name}_dataset"

        # Iterate over each file; create a dataset for each file; create an assignmnet
        dataset_files = list(directory.glob("*.jsonl"))
        for file in dataset_files:
            with open(file) as f:
                log.debug(f"Adding file: {file}")
                frames = list()
                # FIXME: This is likely EXTREMELY slow
                for line in f:
                    json_data = json.loads(line)
                    actual_input = ""
                    # {{{ Convert Input column to strings
                    for input_data in json_data[input_col]:
                        content = input_data["content"]
                        actual_input = f"{actual_input}\n\n{content}"
                    # }}}
                    json_data["input"] = actual_input
                    frames.append(json_data)
                dataset = Dataset.from_list(frames)
                benchmark.add_dataset(dataset_name, dataset)  # type: ignore[code]
            # If @OrestesK adds adding multiple datasets indent the add_assignment line, create an array of the datasets
            # and then add them in the array as a second argument
        benchmark.add_assignment(name, dataset_name, input_col, output_col)

    ranger.add_benchmark(benchmark)
    ranger.run_benchmark("openai_evals")

    for result in ranger.get_results():
        log.info(result)


def openai_testing():
    from openai_utils import OpenAIEvals
    import json

    evals = OpenAIEvals()
    benchmark: Benchmark = Benchmark("openai_evals")
    input_col = "input"
    output_col = "ideal"

    model_source = "baseten"
    ranger: Ranger = Ranger(
        model_source, "baseten", "DZ0KPsrZ.dGcR1CXBTSQSTzH1Vur2GCX5W1kSk7PL", "1PylM20"
    )
    name = "sarcasm"
    file = (
        evals.DATA_PATH / f"{name}/samples.jsonl"
    )  # change this if you want to test it out on a different file

    # Iterating over all the assignents and their data
    with open(file) as f:
        log.debug(f"Adding file: {file}")
        frames = list()
        # FIXME: This is PROBABLY slow
        for line in f:
            json_data = json.loads(line)
            actual_input = ""
            # {{{ Convert Input column to strings
            for input_data in json_data[input_col]:
                content = input_data["content"]
                actual_input = f"{actual_input}\n\n{content}"
            # }}}
            json_data["input"] = actual_input
            frames.append(json_data)
        dataset = Dataset.from_list(frames)
        benchmark.add_dataset(f"{name}_dataset", dataset)  # type: ignore[code]
    # If @OrestesK adds multiple datasets indent the add_assignment line, create an array of the datasets
    # and then add them in the array as a second argument

    benchmark.add_assignment(name, f"{name}_dataset", input_col, output_col)

    # TODO: @OrestesK, can you add logging for which assignment is being run?
    ranger.add_benchmark(benchmark)
    ranger.run_benchmark("openai_evals")

    for result in ranger.get_results():
        log.info(result)


if __name__ == "__main__":
    log.basicConfig(level=log.DEBUG, filename="benchmark_runner.log")
    # upload_legalbench()
    legalbench()
    # openai_evals()
    #openai_testing()
