import argparse
import datasets as nlp
import sys
import os
import openai
import csv
import random
import time
from tqdm import tqdm
import ast
from datetime import datetime
import git
import copy
import itertools
import jsonlines

"""
Usage: python get_cose_gpt3_completion_accuracy.py [task_type] [model_engine] [num_samples_per_instance] [get_shuffled_examples (Bool)] [prime_source ("ours" or "dataset")] [testing] [total_n_from_training_data]
Example: python get_cose_gpt3_completion_accuracy.py jointBoth davinci 1 True ours False
"""
openai.api_key = os.getenv("OPENAI_API_KEY")


def complete_gpt3(prompt, args):
    """
    This function extends 1 or more GPT-3 prompts

    Inputs:
        prompt: str or [str] is a prompt or list of prompts

        generation_length: maximum length of output in tokens

        model: GPT-3 version (davinci, curie, babbage, or ada)

        num_log_probs: number k of top_k log_probabilities to include for each token gen

        top_p: for nucleus sampling in generation

        stop: stop token

        echo: whether or not to include the input in the GPT-3 response

    Output:
        response: the raw response from GPT-3, with simple json format
            note that the 'choices' field is where generations are in the format
            [c_00, c01, ... cnm] where c_ij is the jth generation (based on input n)
            of the ith prompt

    Function modified from Peter West
    """

    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    if not args.testing:
        while not received:
            try:
                response = openai.Completion.create(
                    engine=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=args.generation_length,
                    top_p=args.top_p,
                    frequency_penalty=args.frequency_penalty,
                    presence_penalty=args.presence_penalty,
                    stop=args.stop_token,
                    n=args.num_samples - 1,  # remove 1 for greedy
                    echo=args.echo,
                    logprobs=args.logprobs,
                )
                received = True
            except:
                error = sys.exc_info()[0]
                if (
                    error == openai.error.InvalidRequestError
                ):  # something is wrong: e.g. prompt too long
                    print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                    assert False

                print("API error:", error)
                time.sleep(0.2)
    else:
        response = {"choices": [{"text": "blah blah blah"}]}

    return response


def complete_greedy(prompt, args):

    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    if not args.testing:
        while not received:
            try:
                response = openai.Completion.create(
                    engine=args.model,
                    prompt=prompt,
                    temperature=0,
                    max_tokens=args.generation_length,
                    top_p=args.top_p,
                    frequency_penalty=args.frequency_penalty,
                    presence_penalty=args.presence_penalty,
                    stop=args.stop_token,
                    n=1,
                    echo=args.echo,
                    logprobs=args.logprobs,
                )
                received = True
            except:
                error = sys.exc_info()[0]
                if (
                    error == openai.error.InvalidRequestError
                ):  # something is wrong: e.g. prompt too long
                    print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                    assert False

                print("API error:", error)
                time.sleep(0.2)
    else:
        response = {"choices": [{"text": "blah blah blah"}]}

    return response


def get_completion_nll(choice):
    """
    Given a ``choice'' field from a GPT-3 response, get
    the negative log-likelihood of the full generation (including stop token)

    Assumes this was called with echo=False (gets nll of full choice)

    Function from Peter West
    """
    nll = None
    try:
        j = choice["logprobs"]["text_offset"].index(
            max(choice["logprobs"]["text_offset"])
        )
    except:
        try:
            # if stop-token is not in list, get nll of full list
            j = len(choice["logprobs"]["text_offset"])
        except:
            nll = "not computable"

    if nll != "not computable":
        # sum of log probs over the target tokens
        nll = -sum(choice["logprobs"]["token_logprobs"][:j]) / len(
            choice["logprobs"]["token_logprobs"][:j]
        )
    return nll


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_type",
        type=str,
        choices=[
            "labelOnly",
            "explanationOnly",
            "jointBoth",
        ],
        required=True,
    )
    parser.add_argument("--model", type=str, choices=["ada", "davinci"], required=True)
    parser.add_argument("--num_samples", type=int, required=False, default=1)
    parser.add_argument(
        "--prime_source", type=str, choices=["ours", "dataset"], required=True
    )
    parser.add_argument(
        "--split", type=str, choices=["validation", "test", "train"], required=True
    )
    parser.add_argument(
        "--dataset", type=str, choices=["cose_v1.11", "ecqa"], required=True
    )
    parser.add_argument(
        "--total_train_instances",
        type=int,
        required=False,
        help="# of instances from training dataset to sample for (only used if dataset split == train",
    )
    parser.add_argument("--stop_token", type=str, required=False, default="###")
    parser.add_argument("--generation_length", type=int, required=False, default=50)
    parser.add_argument(
        "--temperature", type=int, required=False, default=0
    )  # greedy sampling
    parser.add_argument("--top_p", type=int, required=False, default=1)
    parser.add_argument("--random_seed", type=int, required=False, default=10)
    parser.add_argument("--frequency_penalty", type=int, required=False, default=0)
    parser.add_argument("--presence_penalty", type=int, required=False, default=0)
    parser.add_argument("--logprobs", type=int, required=False, default=0)
    parser.add_argument("--echo", action="store_const", required=False, const=True)
    parser.add_argument(
        "--get_shuffled_examples", action="store_const", required=False, const=True
    )
    parser.add_argument("--testing", action="store_const", required=False, const=True)

    args, _ = parser.parse_known_args()
    args.command = " ".join(["python"] + sys.argv)

    # check args
    if args.get_shuffled_examples and args.task_type == "labelOnly":
        raise Exception(
            "can only produce explanations for judging if explanations part of prediction task"
        )

    # defines draw of *other* seeds
    random.seed(args.random_seed)

    # draw args.total_train_instances seeds
    if args.split == "train":
        # must specify # of samples
        assert args.total_train_instances is not None
        prompt_seeds = random.sample(
            [i for i in range(10000)], args.total_train_instances
        )
    elif args.split == "test":
        # use pre-specified number
        args.total_train_instances = 250
        prompt_seeds = random.sample(
            [i for i in range(1000)], args.total_train_instances
        )
    elif args.split == "validation":
        # use pre-specified number
        args.total_train_instances = 115
        prompt_seeds = random.sample(
            [i for i in range(1000)], args.total_train_instances
        )

    if not args.testing:
        # create a save directory
        if not os.path.exists("./gpt3_outputs/"):
            os.mkdir("./gpt3_outputs/")
        save_path = f"./gpt3_outputs/{args.dataset}/"
        save_dir = os.path.join(save_path, datetime.now().strftime("%m%d%y_%H%M%S"))
        assert os.path.exists(save_path)
        assert not os.path.exists(save_dir)
        os.makedirs(save_dir)

        # get git hash and branch where deployed
        repo = git.Repo(search_parent_directories=True)
        git_hash = repo.head.object.hexsha
        git_branch = repo.active_branch.name

        # write command to logfile
        with open(os.path.join(save_dir, "commandline_args.txt"), "w") as f:
            f.write("Git branch: " + git_branch + "\n")
            f.write("Git hash: " + git_hash + "\n")
            for key in args.__dict__:
                f.write(f"{key}: {args.__dict__[key]}\n")

        outfile = os.path.join(save_dir, "generations.csv")
        if args.get_shuffled_examples:
            outfile_shuff = os.path.join(save_dir, "shuffled_generations.csv")
            g = open(outfile_shuff, "w")
            shuff_writer = csv.writer(g)
            shuff_writer.writerow(
                [
                    "id",
                    "question",
                    "answer_choices",
                    "gold_label",
                    "source_sample_1",
                    "source_sample_2",
                    "nll_1",
                    "nll_2",
                    "explanation_1",
                    "explanation_2",
                ]
            )

    if args.prime_source == "dataset" and args.dataset == "ecqa":
        ecqa_expls = {}
        # load ECQA explanations from file
        with jsonlines.open("./ecqa.jsonl") as f:
            dataset = [obj for obj in f]
        for it in dataset:
            ecqa_expls[it["id"]] = it["explanation"]

    # load full train+validation prime set
    primes = []
    prime_ids = []
    with open(f"../data/handwritten_cose_v1.11_examples.csv", "r") as f:
        reader = csv.DictReader(f)
        for line in reader:
            prime_ids.append(line["id"])
            if args.prime_source == "dataset" and args.dataset == "cose_v1.11":
                # load dataset explanations for primes
                expl = line["orig_explanation"]
            elif args.prime_source == "dataset" and args.dataset == "ecqa":
                # get associated ECQA explanation
                expl = ecqa_expls[line["id"]].replace("\n", " ")
            else:
                # load our hand-written explanations for primes
                expl = line["our_explanation"]

            primes.append(
                {
                    "abstractive_explanation": expl,
                    "choices": ast.literal_eval(line["choices"]),
                    "answer": line["answer"],
                    "question": line["question"],
                }
            )
    assert len(primes) == 115

    if args.split == "test":
        # load "test" set (that also has gold explanations)
        ds = nlp.load_dataset("cos_e", "v1.11", split="validation")
        ds = ds.shuffle(seed=10)
        # select random subset of 250 instances
        splits = ds.train_test_split(test_size=(250 / len(ds)))
        dataset = splits["test"]
        assert len(dataset) == 250

    elif args.split == "validation":
        # doing prompting LOO-style on primes themselves
        dataset = primes

    elif args.split == "train":
        ds = nlp.load_dataset("cos_e", "v1.11", split="train[15:]")
        ds = ds.shuffle(seed=10)
        # select random subset of args.total_train_instances size
        dataset = []
        i = 0
        while len(dataset) < args.total_train_instances:
            # add elements
            if ds[i]["id"] not in prime_ids:
                dataset.append(ds[i])
            i += 1
    assert len(dataset) == args.total_train_instances

    # write out for annotation
    with open(outfile, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "sample_number",
                "question",
                "answer_choices",
                "gold_label",
                "predicted_label",
                "gold_explanation",
                "predicted_explanation",
                "prediction_nll",
                "num_primes",
                "prompt",
            ]
        )
        expl_pairs = []
        acc = []

        for i, (new_seed, inst) in tqdm(
            enumerate(zip(prompt_seeds, dataset)), total=len(dataset)
        ):
            instance_pairs = []
            # select prime indices
            random.seed(new_seed)
            if args.dataset == "ecqa":
                # 24 ECQA primes is sometimes too long for the API context window (2049 tokens)
                num_primes = random.choice([8, 16])
            else:
                num_primes = random.choice([8, 16, 24])

            if args.split == "validation":
                # remove self from primes list
                partial_list = copy.deepcopy(primes)
                partial_list.remove(inst)
                selected_primes = random.sample(partial_list, num_primes)
            else:
                # select from all primes
                selected_primes = random.sample(primes, num_primes)
            random.shuffle(selected_primes)

            # construct prompt shared by all variants of this instance
            if args.task_type == "labelOnly":
                prompt = "Let's perform a classification task.\n\n"
                for item in selected_primes:
                    # shuffle answer choices
                    ch = random.sample(item["choices"], len(item["choices"]))
                    prompt += f"question: {item['question']}\n{', '.join(ch[:-1]) + ', or ' + ch[-1]}? {item['answer']}\n###\n"
            elif args.task_type in {"explanationOnly", "jointBoth"}:
                prompt = "Let's explain classification decisions.\n\n"
                for item in selected_primes:
                    # shuffle answer choices
                    ch = random.sample(item["choices"], len(item["choices"]))
                    prompt += f"question: {item['question']}\n{', '.join(ch[:-1]) + ', or ' + ch[-1]}? {item['answer']}\nwhy? {item['abstractive_explanation']}\n###\n"

            quest = inst["question"]
            gold_e = inst["abstractive_explanation"]
            gold_l = inst["answer"]

            # shuffle answer choices
            choices = random.sample(inst["choices"], len(inst["choices"]))
            choice_string = ", ".join(choices[:-1]) + ", or " + choices[-1]

            # reset seed (for good measure)
            random.seed(args.random_seed)

            # append instance to prompt
            if args.task_type in {"labelOnly", "jointBoth"}:
                prompt_list = [prompt + f"question: {quest}\n{choice_string}?"]
            elif args.task_type == "explanationOnly":
                prompt_list = [
                    prompt + f"question: {quest}\n{choice_string}? {gold_l}\nwhy?"
                ]
            else:
                raise Exception

            if i == 0:
                print("#######################")
                print(prompt_list[0])
                print("#######################")

            assert len(prompt_list) == 1

            for custom_prompt in prompt_list:
                if args.num_samples > 1:
                    # perform GPT-3 inference to get predicted label and explanation
                    response = complete_gpt3(custom_prompt, args)

                    # iterates over n returned generations
                    for j, choice in enumerate(response["choices"]):

                        if args.task_type == "labelOnly":
                            pred_l = (
                                choice["text"]
                                .split("\n")[0]
                                .strip("\n")
                                .strip()
                                .lower()
                            )
                            pred_e = "n/a"
                        elif args.task_type == "explanationOnly":
                            pred_e = (
                                choice["text"]
                                .split("\n")[0]
                                .strip("\n")
                                .strip()
                                .strip('"')
                                .strip()
                            )
                            if not isinstance(pred_e, str):
                                breakpoint()
                            pred_l = "n/a"
                        elif args.task_type == "jointBoth":
                            splits = choice["text"].split("why?")
                            pred_l = splits[0].strip("\n").strip().lower()
                            if len(splits) > 1:
                                pred_e = (
                                    splits[1].strip("\n").strip().strip('"').strip()
                                )
                            else:
                                pred_e = "n/a"

                        # echo must be false to get nll using this function
                        assert not args.echo
                        pred_nll = get_completion_nll(choice)

                        writer.writerow(
                            [
                                inst["id"],
                                j,
                                quest,
                                choice_string,
                                gold_l,
                                pred_l,  # sample-specific
                                gold_e,
                                pred_e,  # sample-specific
                                pred_nll,  # sample-specific
                                num_primes,
                                custom_prompt.replace("\n", "\\n"),
                            ]
                        )
                        instance_pairs.append((j, pred_e, pred_nll))

                        # only consider the first (highest probability) response in accuracy calculations and explanations to compare to golds
                        if j == 0:

                            # add accuracy score
                            if pred_l == gold_l or args.task_type == "explanationOnly":
                                acc.append(1)
                            else:
                                acc.append(0)

                # repeat for the greedy sample
                greedy = complete_greedy(custom_prompt, args)

                # also add greedy
                pred_e = (
                    greedy["choices"][0]["text"]
                    .split("\n")[0]
                    .strip("\n")
                    .strip()
                    .strip('"')
                    .strip()
                )
                if not isinstance(pred_e, str):
                    breakpoint()
                pred_l = "n/a"
                assert not args.echo
                pred_nll = get_completion_nll(greedy["choices"][0])
                writer.writerow(
                    [
                        inst["id"],
                        "greedy",
                        quest,
                        choice_string,
                        gold_l,
                        pred_l,  # sample-specific
                        gold_e,
                        pred_e,  # sample-specific
                        pred_nll,  # sample-specific
                        num_primes,
                        custom_prompt.replace("\n", "\\n"),
                    ]
                )
                instance_pairs.append(("greedy", pred_e, pred_nll))

                if args.get_shuffled_examples:
                    # create 10 combos of explanation pairs and append
                    rank_combos = itertools.combinations(instance_pairs, 2)
                    for i, item in enumerate(rank_combos):
                        tmp_lst = [item[0], item[1]]
                        random.shuffle(tmp_lst)
                        # tuple order: sample #, explanation, NLL
                        shuff_writer.writerow(
                            [
                                inst["id"],
                                quest,
                                choice_string,
                                gold_l,
                                tmp_lst[0][0],
                                tmp_lst[1][0],
                                tmp_lst[0][2],
                                tmp_lst[1][2],
                                tmp_lst[0][1],
                                tmp_lst[1][1],
                            ]
                        )
                    assert i == 9

        # compute final accuracy
        print("Number of examples: ", len(acc))
        print("Final Accuracy: ", sum(acc) / float(len(acc)) * 100)
        print("Save directory: ", save_dir)
