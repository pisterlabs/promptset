import argparse
import json
import os
import subprocess
import time
from typing import Dict, List, Tuple

import wandb
import yaml
from openai import OpenAI
from openai.types.chat import ChatCompletion
from tenacity import retry, stop_after_attempt, wait_random_exponential
from termcolor import colored


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model", type=str, default="gpt-4-1106-preview", choices=["gpt-4-1106-preview", "gpt-3.5-turbo-1106"]
    )
    parser.add_argument(
        "--description",
        type=str,
        default="Fine-tuning a BERT-large model for stance detection between tweets and frames of communication, "
        "utilizing multi-headed attention with moral and human value embeddings. Be sure to keep `output_dim` divisible  by `num_heads`.",
        help="Description of the experiment to optimize.",
    )
    parser.add_argument("--experiments", type=int, default=10, help="Number of experiments to run.")
    parser.add_argument(
        "--hyperparameters",
        nargs="+",
        default=[
            "learning_rate:number",
            "batch_size:integer",
            "accumulate_grad_batches:integer",
            "lr_warm_up:number",
            "max_epochs:integer",
            "weight_decay:number",
            "output_dim:integer",
            "num_heads:integer",
            "dropout:number",
        ],
        help="List of hyperparameters to optimize.",
    )
    parser.add_argument(
        "--skip_config_keys",
        nargs="+",
        default=[
            "class_path",
            "seed_everything",
            "threshold",
            "check_val_every_n_epoch",
            "deterministic",
            "num_sanity_val_steps",
            "accelerator",
            "devices",
            "default_root_dir",
            "enable_checkpointing",
            "logger",
            "callbacks",
            "label_name",
            "num_workers",
            "frame_path",
            "train_path",
            "val_path",
            "test_path",
            "predict_path",
            "value_list",
            "cultural_list",
            "moral_list",
            "label_map",
            "tokenizer_name",
            "update_threshold",
        ],
        help="List of hyperparameters to optimize.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--org", type=str, default="hltri", help="Organization to use for wandb.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "val_f1",
            "val_p",
            "val_r",
            # "val_loss",
            # "train_loss",
        ],
        help="Metrics to show.",
    )
    parser.add_argument("--metric", type=str, default="val_f1", help="Metric to optimize.")
    parser.add_argument("--direction", type=str, default="maximize", choices=["maximize", "minimize"])
    parser.add_argument("--delay", type=int, default=10, help="Minimum delay between experiments in seconds.")
    parser.add_argument("--text", type=str, default="text.txt", help="Output file for printed text.")
    parser.add_argument("--messages", type=str, default="messages.jsonl", help="Output file for messages as jsonl.")
    parser.add_argument("--train_size", type=int, default=10250, help="Size of the training set.")
    parser.add_argument("--val_size", type=int, default=1115, help="Size of the validation set.")
    parser.add_argument(
        "--device",
        type=str,
        default="NVIDIA TITAN V w/ 12GB VRAM",
        help="Accelerator device to help inform hyperparameter selection.",
    )


def get_ex_idx(config_path: str):
    _, file_name = os.path.split(config_path)
    ex_name = file_name.split(".")[0]
    try:
        if "-" not in ex_name:
            return 0
        # v1, v2, v3, etc.
        version = ex_name.split("-")[-1]
        ex_idx = int(version[1:])
    except ValueError:
        ex_idx = 0
    return ex_idx


def get_new_ex_path(config_path: str, i: int):
    ex_path, file_name = os.path.split(config_path)
    ex_name = file_name.split(".")[0]
    # bert, bert-v1, bert-v2, etc.
    if "-" not in ex_name:
        new_ex_name = f"{ex_name}-v{i}.yaml"
    else:
        version = ex_name.split("-")[-1]
        new_ex_name = f"{ex_name[:-len(version)]}v{i}.yaml"
    new_config_path = os.path.join(ex_path, new_ex_name)
    return new_config_path


def prune_config(config: Dict[str, str], skip_config_keys: set):
    for k, v in list(config.items()):
        if k in skip_config_keys:
            del config[k]
        elif k == "init_args":
            for k2, v2 in list(v.items()):
                if k2 in skip_config_keys:
                    del v[k2]
                    continue
                elif isinstance(v2, dict):
                    prune_config(v2, skip_config_keys)
                config[k2] = v2
            del config[k]
        elif isinstance(v, dict):
            prune_config(v, skip_config_keys)
    return config


# recursively update the config with the hyperparameters if they match any keys in the config
def update_config(config: Dict[str, str], hyperparameters: Dict[str, str], ex_name: str, logs_path=None, project=None):
    for k, v in config.items():
        if k in hyperparameters:
            config[k] = hyperparameters[k]
        elif k == "logger":
            logs_path = v["init_args"]["save_dir"]
            project = v["init_args"]["project"]
            v["init_args"]["name"] = ex_name
        elif k == "default_root_dir":
            path, _ = os.path.split(v)
            config[k] = os.path.join(path, ex_name)
        elif isinstance(v, dict):
            config[k], logs_path, project = update_config(config[k], hyperparameters, ex_name, logs_path, project)
    return config, logs_path, project


def create_new_config(hyperparameters: Dict[str, str], config_path: str, i: int):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    new_ex_path = get_new_ex_path(config_path, i)
    new_ex_name = os.path.split(new_ex_path)[-1].split(".")[0]
    new_config, logs_path, project = update_config(config, hyperparameters, new_ex_name)

    with open(new_ex_path, "w") as f:
        yaml.dump(new_config, f)

    return new_ex_path, logs_path, project


def print_message(message, fo):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "tool": "magenta",
    }

    if message["role"] == "system":
        print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        fo.write(f"system: {message['content']}\n\n")
    elif message["role"] == "user":
        print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        fo.write(f"user: {message['content']}\n\n")
    elif message["role"] == "assistant" and message.get("tool_calls"):
        tool_calls = message["tool_calls"]
        lines = ["assistant:"]
        for tool_call in tool_calls:
            lines.append(f"  tool ({tool_call['function']['name']}):")
            try:
                hyperparameters = json.loads(tool_call["function"]["arguments"])
                for k, v in hyperparameters.items():
                    lines.append(f"    {k}: {v}")
            except json.decoder.JSONDecodeError:
                hyperparameters = tool_call["function"]["arguments"]
                lines.append(f"    {hyperparameters}")
        print(colored("\n".join(lines) + "\n", role_to_color[message["role"]]))
        fo.write("\n".join(lines) + "\n\n")
    elif message["role"] == "assistant":
        print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        fo.write(f"assistant: {message['content']}\n\n")
    elif message["role"] == "tool":
        print(colored(f"tool ({message['name']}): {message['content']}", role_to_color[message["role"]]))
        fo.write(f"tool ({message['name']}): {message['content']}\n")


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def wandb_summary(org: str, project: str, ex_id: str):
    api = wandb.Api()
    run = api.run(f"{org}/{project}/{ex_id}")
    summary: dict = run.summary
    return list(sorted(summary.items(), key=lambda x: x[0]))


def run_experiment(
    hyperparameters: Dict[str, str], config_path: str, i: int, org: str, metric: str, metrics: List[str]
):
    ex_config_path, logs_path, project = create_new_config(hyperparameters, config_path, i)
    print(f"Running experiment: {ex_config_path}\n")
    ex_metric = 0.0
    try:
        start = time.time()
        subprocess.run(
            ["python", "pytorch_gleam/ex/gleam.py", "fit", "--config", f"{ex_config_path}"],
            capture_output=True,
            check=True,
        )
        end = time.time()
        seconds = end - start
        # outputs = out.stdout.decode()
        run_dir = os.path.join(logs_path, "wandb", "latest-run")
        ex_id = None
        for file in os.listdir(run_dir):
            if file.endswith(".wandb"):
                # run-ID.wandb
                ex_id = file.split(".")[0].split("-")[-1]
                break
        summary = wandb_summary(org, project, ex_id)
        # TODO consider entire run history, not just the last values
        # https://docs.wandb.ai/guides/track/public-api-guide#runhistory
        outputs = []
        for k, v in summary:
            if k.startswith("_") or isinstance(v, dict):
                continue
            if k not in metrics:
                continue
            outputs.append(f"{k}: {v:.4f}")
            if k == metric:
                ex_metric = v
        outputs.append(f"seconds: {seconds:.0f}")
        outputs = "\n".join(outputs)

    except subprocess.CalledProcessError as e:
        # TODO possibly include stdout
        # TODO handle errors better
        outputs = e.stderr.decode()
    return outputs, ex_config_path, ex_metric


def run_test(ex_config_path: str):
    print(f"Running test evaluation: {ex_config_path}\n")
    subprocess.run(
        ["python", "pytorch_gleam/ex/gleam.py", "test", "--config", f"{ex_config_path}"],
        capture_output=True,
        check=True,
    )


class MinimumDelay:
    def __init__(self, delay: int):
        self.delay = delay
        self.start = None

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        seconds = end - self.start
        if self.delay > seconds:
            time.sleep(self.delay - seconds)


class MessageContext:
    def __init__(self, file_path: str = None, text_file_path: str = None, load: bool = False):
        self.messages = []
        self.file_path = file_path
        self.file = None
        self.text_file_path = text_file_path
        self.text_file = None
        self.load = load

    def __enter__(self):
        if self.file_path is not None:
            if os.path.exists(self.file_path) and self.load:
                # first read the existing messages
                with open(self.file_path, "r") as f:
                    for line in f:
                        message = json.loads(line)
                        self.messages.append(message)
                self.file = open(self.file_path, "a")
            else:
                self.file = open(self.file_path, "w")
        if self.text_file_path is not None:
            if os.path.exists(self.text_file_path) and self.load:
                self.text_file = open(self.text_file_path, "a")
            else:
                self.text_file = open(self.text_file_path, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_path is not None:
            self.file.close()
        if self.text_file_path is not None:
            self.text_file.close()

    def add_message(self, message):
        self.messages.append(message)
        if self.text_file_path is not None:
            print_message(message, self.text_file)
        if self.file_path is not None:
            json.dump(message, self.file)
            self.file.write("\n")

    def add_messages(self, messages):
        for message in messages:
            self.add_message(message)

    def get_messages(self):
        return self.messages


def run_step(
    delay: int,
    mg: MessageContext,
    model: str,
    config_path: str,
    i: int,
    org: str,
    metric: str,
    metrics: List[str],
    seed: int,
    tools: List[dict],
    client: OpenAI,
):
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def chat(**kwargs) -> ChatCompletion:
        return client.chat.completions.create(**kwargs)

    with MinimumDelay(delay):
        # TODO handle exceptions: https://platform.openai.com/docs/guides/error-codes/python-library-error-types
        chat_completion = chat(
            messages=mg.get_messages(),
            model=model,
            max_tokens=512,
            seed=seed,
            top_p=0.7,
            tool_choice="auto",
            tools=tools,
        )
        tool_calls = chat_completion.choices[0].message.model_dump()["tool_calls"]
        mg.add_message(
            {
                "role": "assistant",
                "tool_calls": tool_calls,
                "content": "",  # hack to get the tool calls to show up
            }
        )

        # TODO could fail to use either tools
        # TODO could fail to use only one tool
        assert len(tool_calls) == 1
        tool_call = tool_calls[0]

        if tool_call["function"]["name"] == "end":
            return None, None

        # TODO could fail to use the run tool
        assert tool_call["function"]["name"] == "run"

        tool_call_id = tool_call["id"]
        try:
            hyperparameters = json.loads(tool_call["function"]["arguments"])
            results, ex_config_path, ex_metric = run_experiment(hyperparameters, config_path, i, org, metric, metrics)
            response_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_call["function"]["name"],
                "content": results,
            }
        except json.decoder.JSONDecodeError:
            response_message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_call["function"]["name"],
                "content": f"Invalid function arguments JSON: {tool_call['function']['arguments']}",
            }

        mg.add_message(response_message)

    with MinimumDelay(delay):
        # TODO handle exceptions: https://platform.openai.com/docs/guides/error-codes/python-library-error-types
        # have the model discuss the results and what was learned
        chat_completion = chat(
            messages=mg.get_messages(),
            model=model,
            max_tokens=512,
            seed=seed,
            top_p=0.7,
        )
        mg.add_message(
            {
                "role": "assistant",
                "content": chat_completion.choices[0].message.content,
            }
        )
    return ex_config_path, ex_metric


def optimize(
    model: str,
    description: str,
    experiments: int,
    config_path: str,
    seed: int,
    hyperparameter_names_types: List[Tuple[str, str]],
    skip_config_keys: set,
    org: str,
    metric: str,
    metrics: List[str],
    direction: str,
    delay: int,
    text_output: str,
    message_output: str,
    device: str,
    train_size: int,
    val_size: int,
):
    hyperparameter_names = [h for (h, _) in hyperparameter_names_types]
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = prune_config(config, skip_config_keys)
    config_str = yaml.dump(config)

    client = OpenAI()

    system_prompts = [
        "You are an assistant to a graduate research scientist.",
        "You will assist in optimizing the hyperparameters of ongoing experiments.",
        "You will be given a description of the experiment and a set of hyperparameters to optimize.",
        "Results of each experiment will be provided to you.",
        "You will continue to propose new hyperparameters and run experiments with the goal of improving the results.",
        "Furthermore, after each experiment, you will be asked to discuss the results and what was learned.",
        "Only propose one set of hyperparameters for a single experiment.",
        "Be sure to explore the hyperparameter space and not get stuck in a local optimum.",
    ]
    system_prompt = " ".join(system_prompts)

    user_prompts = [
        f"This experiment involves: {description}",
        f"The training set will contain {train_size:,} examples, "
        + f"while the validation set will contain {val_size:,} examples.",
        f"Experiments will be performed on a {device}.",
        "The initial configuration file for this experiment is:",
        f"```yaml\n{config_str}```",
        f"The hyperparameters to optimize are: {', '.join(hyperparameter_names)}",
        "Please begin to propose hyperparameters.",
        "You can run the experiment with the given hyperparameters by using the `run` tool.",
        "You can also end the session by using the `end` tool.",
        "Only use the `end` tool when you are done and do not believe you can improve the results any further.",
        f"Please {direction} the {metric} metric.",
    ]

    run_tool = {
        "type": "function",
        "function": {
            "name": "run",
            "description": "Run the experiment with the given hyperparameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    h: {
                        "type": t,
                        "description": f"The value for the {h} hyperparameter.",
                    }
                    for (h, t) in hyperparameter_names_types
                },
                "required": hyperparameter_names,
            },
        },
    }

    end_tool = {
        "type": "function",
        "function": {
            "name": "end",
            "description": "End this session of running the experiment. Only use this tool when "
            + "you are done and do not believe you can improve the results any further.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "The reason for why the session is ending.",
                    }
                },
                "required": ["reason"],
            },
        },
    }

    tools = [run_tool, end_tool]

    results = []

    with MessageContext(file_path=message_output, text_file_path=text_output) as mg:
        start_idx = get_ex_idx(config_path)
        mg.add_message({"role": "system", "content": system_prompt})
        mg.add_message({"role": "user", "content": "\n".join(user_prompts)})

        for i in range(start_idx + 1, start_idx + experiments + 1):
            ex_config_path, ex_metric = run_step(
                delay=delay,
                mg=mg,
                model=model,
                config_path=config_path,
                i=i,
                org=org,
                metric=metric,
                metrics=metrics,
                seed=seed,
                tools=tools,
                client=client,
            )
            if ex_config_path is None:
                break
            results.append((ex_metric, ex_config_path))

    print("Results:")
    sorted_results = list(
        sorted(
            results,
            key=lambda x: x[0],
            reverse=(direction == "maximize"),
        )
    )
    for ex_metric, ex_config_path in sorted_results:
        print(f"{ex_metric:.4f}: {ex_config_path}")
    print()
    print("Best:")
    best_metric, best_config_path = sorted_results[0]
    print(f"{best_metric:.4f}: {best_config_path}")
    print()
    run_test(best_config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment configuration file.")

    args = parser.parse_args()

    optimize(
        model=args.model,
        description=args.description,
        experiments=args.experiments,
        config_path=args.config,
        seed=args.seed,
        hyperparameter_names_types=[h.split(":") for h in args.hyperparameters],
        skip_config_keys=set(args.skip_config_keys),
        org=args.org,
        metric=args.metric,
        metrics=args.metrics,
        direction=args.direction,
        delay=args.delay,
        text_output=args.text,
        message_output=args.messages,
        device=args.device,
        train_size=args.train_size,
        val_size=args.val_size,
    )
