from tqdm import tqdm
from utils import main_data_utils as mdu
from datasets import load_dataset, Dataset, DatasetDict, Dataset
import os, warnings, regex, numpy as np, collections, math, time

from utils.data_utils.augment_slices import openai_complete


def _normalize(probs):
    """
    Given label probs Dict[str: list]
    first, normalizes probablities of tokens predicted multiple times
    then, normalizes across all the predicted labels
    """
    # NOTE: not using assertion because davinci and curie give differet probs
    # for the same prediction sometimes
    # for k, v in probs.items():
    # prob should be the same for all the multiple predictions of the label
    # assert len(set(v)) == 1
    # probs[k] = v[0]
    probs = {k: np.mean(v) for k, v in probs.items()}
    return {k: v / sum(probs.values()) for k, v in probs.items()}


def gpt3mix_complete(prompt, n, labels_list, exp_dict, name2id):
    """
    Given a seed_text and its corresponding seed_intent (name, not id),

    1. generate x_dash (n augmentations per seed_text)
    2. generate y_dash (soft label using name2id)
    """
    pattern = regex.compile(rf"(?r)Sentence: (.*)\(intent: (.*)\)")
    # gpt prompting to generate x_dashes
    completions = openai_complete(
        engine=exp_dict["gpt3_engine"],
        prompt=prompt,
        temp=exp_dict["gpt3_temp"],
        top_p=1.0,
        n=n,
        stop="\n",
        max_tokens=50,
        frequency_penalty=0.02,
    )

    augmentations = {"text": [], "intent": []}
    for c in completions:
        match = pattern.search("Sentence:" + c.text)
        if match is None:  # invalid prediction
            continue
        _txt = match.group(1).strip().lower()
        if not _txt:
            continue

        # prompt GPT3 again to create soft label
        label_completions = openai_complete(
            engine=exp_dict["gpt3_engine"],
            prompt=prompt + f" {_txt}. (intent:",
            temp=exp_dict["gpt3_temp"],
            n=100,
            top_p=1.0,
            max_tokens=20,
            stop="\n",
            logprobs=1,
        )

        # construct probabilities for all the predicted labels
        label_probs = collections.defaultdict(list)
        for _lc in label_completions:
            _log_probs = _lc.logprobs
            _match = pattern.search(f"Sentence: {_txt}. (intent:" + _lc.text)
            if _match is None:  # incomplete prediction
                continue

            _pred = _match.group(2).strip().lower()
            if _pred not in labels_list:  # invalid label
                continue

            # NOTE: we are looking at token_logprobs v/s top_logprobs used
            # by the GPT3Mix paper because we are sampling to compute
            # p(y_dash| x_dash) as opposed to looking at logprobs of top 100
            # most likely tokens. default value limits us to just 5 now.
            _curr_log_prob = 0
            for t, p in zip(_log_probs["tokens"], _log_probs["token_logprobs"]):
                # if the code reaches here, ) is guaranteed to be present
                # as regex check earlier would trigger a `continue` otherwise
                if t == ")":
                    label_probs[_pred].append(math.exp(_curr_log_prob))
                    break

                # add logprobs (multiply probs) for sub words of _pred as
                # class names are not single tokens
                _curr_log_prob += p

        # normalize label_probs
        label_probs = _normalize(label_probs)
        # create soft label
        soft_label = [0] * exp_dict["dataset"]["num_labels"]
        for k, v in label_probs.items():
            soft_label[name2id[k]] = v

        augmentations["text"].append(_txt)
        augmentations["intent"].append(soft_label)
    return augmentations


def generate_for_gpt3mix(base_ds, ex2_ds, exp_dict, interim_save_path):
    num_labels = exp_dict["dataset"]["num_labels"]
    ds_name = exp_dict["dataset"]["name"]
    id2name = mdu.read_json(f"data/{ds_name}/id2name.json")
    name2id = mdu.read_json(f"data/{ds_name}/name2id.json")
    ds_config = mdu.get_ds_config(ds_name)
    k = ds_config.num_examples

    labels_list = list(name2id.keys())
    if "oos" in labels_list:
        labels_list.remove("oos")

    train_lines, train_labels = [], []
    if os.path.exists(interim_save_path):
        interim_copy = mdu.read_pickle(interim_save_path)
    else:
        interim_copy = {}

    for domain in ex2_ds:
        if domain in interim_copy:
            print(f"Domain: {domain} already GPT3Mix augmented. Moving on...")
            continue

        print(f"Augmenting domain: {domain}")
        texts = ex2_ds[domain]["F"]["train"]["text"]
        hard_labels = ex2_ds[domain]["F"]["train"]["intent"]
        _lines, _labels = [], []
        # NOTE: this loop will never be executed for oos--both lists will be []
        for text, intent in tqdm(zip(texts, hard_labels), total=len(texts)):
            # add gold example to training set
            one_hot = [0.0] * num_labels
            one_hot[intent] = 1.0
            # for interim copy
            _lines.append(text)
            _labels.append(one_hot)

            # construct prompt header
            prompt = "Each item in the following list contains a sentence and the respective intent."
            label_enum_str = [f"'{l.lower()}'" for l in labels_list]
            prompt += f" Intent is one of {', or '.join(label_enum_str)}"
            prompt += ".\n"
            prompt += f"Sentence: {text}. (intent: {id2name[str(intent)]})\n"

            # remove current intent from candidates to sample from
            _lbl_list = [l for l in labels_list if l != id2name[str(intent)]]
            # sample k-1 random intents from the label_set (k=9)
            other_lbls = np.random.choice(_lbl_list, k - 1, replace=False)
            # fetch a sample for each of these new intents and add to the prompt
            for lbl in other_lbls:
                # find the domain of lbl
                _domain, _domain_found = None, False
                for _d, _i_l in ds_config.domain_to_intent.items():
                    if not _domain_found and lbl in _i_l:
                        _domain_found = True
                        _domain = _d

                gt_txts = ex2_ds[_domain]["F"]["train"]["text"]
                gt_lbls = ex2_ds[_domain]["F"]["train"]["intent"]
                _start = gt_lbls.index(name2id[lbl])
                # select a random sentence for lbl
                _text = np.random.choice(gt_txts[_start : _start + k], 1)[0]
                # add the _text, lbl pair to prompt
                prompt += f"Sentence: {_text}. (intent: {lbl})\n"
            prompt += "Sentence:"

            # generated examples with soft labels
            augs = gpt3mix_complete(prompt, 10, labels_list, exp_dict, name2id)
            _lines.extend(augs["text"])
            _labels.extend(augs["intent"])

        train_lines.extend(_lines)
        train_labels.extend(_labels)

        # save an interim copy now
        interim_copy[domain] = {"text": _lines, "intent": _labels}
        mdu.write_pickle(interim_copy, interim_save_path)

        print("Sleeping...for a minute")
        time.sleep(60)

    # Add OOS samples
    oos_texts, oos_labels = extract_oos(base_ds["train"], exp_dict["dataset"]["oos_id"])
    for text, intent in tqdm(zip(oos_texts, oos_labels), total=len(oos_texts)):
        # add gold example to training set
        one_hot = [0.0] * num_labels
        one_hot[intent] = 1.0
        train_lines.append(text)
        train_labels.append(one_hot)

    # delete interim copy
    del interim_copy
    return {"text": train_lines, "intent": train_labels}


def prepare_for_seq2seq(dataset, id2name_path):
    """
    dataset: Dict[str]: <list>
    """
    id2name = mdu.read_json(id2name_path)
    return {
        "text": [t + " </s>" for t in dataset["text"]],
        # intents are class ids here, not names
        "intent": [id2name[str(i)] + " </s>" for i in dataset["intent"]],
    }


def filter_oos(data_dict, oos_id, soft_label=False):
    """Removes oos samples from the data dict"""
    lines, labels = data_dict["text"], data_dict["intent"]
    # some datasets (like SNIPS) don't have an OOS class
    if oos_id is None:
        return lines, labels
    _lines, _labels = [], []
    for idx, intent_id in enumerate(labels):
        if soft_label and np.array(intent_id).argmax(-1) == oos_id:
            continue
        if not soft_label and intent_id == oos_id:
            continue
        _lines.append(lines[idx])
        _labels.append(labels[idx])
    # print(len(_lines), len(_labels))
    return _lines, _labels


def extract_oos(data_dict, oos_id):
    """Extract the OOS samples from the data dict. It is the
    opposite of filter_oos"""
    lines, labels = data_dict["text"], data_dict["intent"]
    # some datasets (like SNIPS) don't have an OOS class
    _lines, _labels = [], []
    for idx, intent_id in enumerate(labels):
        if intent_id != oos_id:
            continue
        _lines.append(lines[idx])
        _labels.append(labels[idx])
    return _lines, _labels


class DatasetLoader:
    """
    Available datasets:
        - Clinc original: We can define whether to get the `full` version or the `small` version.
        - Pure Fewshot Clinc:
            baseline: Contains 10 example per class (except the OOS) which is randomly sampled from the original full clinc.

    """

    def __init__(self, data_root, exp_dict):
        dataset_config = exp_dict["dataset"]["config"]

        var_path = "full" if dataset_config.startswith("f") else "small"
        ds_name = exp_dict["dataset"]["name"]
        basic_data_path = os.path.join(data_root, ds_name, var_path, "dataset.pkl")
        ex2_data_path = os.path.join(
            data_root, ds_name, var_path, "data_full_suite.pkl"
        )

        if dataset_config == "few_pure":
            base_ds = mdu.read_pickle(basic_data_path)
            data_set = mdu.read_pickle(ex2_data_path)
            oos_id = exp_dict["dataset"]["oos_id"]
            train_lines, train_labels = [], []
            if exp_dict["exp_type"] == "baseline":
                print("Loading dataset for full few-shot baseline")
                for domain in data_set:
                    train_lines.extend(data_set[domain]["F"]["train"]["text"])
                    train_labels.extend(data_set[domain]["F"]["train"]["intent"])
            elif exp_dict["exp_type"] in ["eda"]:
                exp_type = exp_dict["exp_type"]
                print(f"Loading dataset for full few-shot {exp_type.upper()}")
                # lump in EDA examples with all few-shot samples
                for domain in data_set:
                    train_lines.extend(
                        data_set[domain]["F"]["train"]["text"]
                        + data_set[domain]["F"][exp_type]["text"]
                    )
                    train_labels.extend(
                        data_set[domain]["F"]["train"]["intent"]
                        + data_set[domain]["F"][exp_type]["intent"]
                    )
            elif exp_dict["exp_type"] in ["gpt3", "eda"]:
                print(f"Loading dataset for full few-shot {exp_dict['exp_type']}")
                # set correct attribute to fetch from the dataset
                if exp_dict["exp_type"] == "gpt3":
                    engine, temp = exp_dict["gpt3_engine"], exp_dict["gpt3_temp"]
                    attr = f"{engine}_{temp}"
                else:  # eda
                    attr = exp_dict["exp_type"]

                # lump in the fetched examples with all few-shot samples
                for domain in data_set:
                    train_lines.extend(
                        data_set[domain]["F"]["train"]["text"]
                        + data_set[domain]["F"][attr]["text"]
                    )
                    train_labels.extend(
                        data_set[domain]["F"]["train"]["intent"]
                        + data_set[domain]["F"][attr]["intent"]
                    )
            elif exp_dict["exp_type"] in [
                "gpt3_oracle",
                "eda_oracle",
                "gpt3mix_oracle",
            ]:
                # the few shot sentences are taken from the ex2 setup data
                # and the relabeled samples are taken from al_dataset.pkl
                print(f"Loading dataset for full few-shot {exp_dict['exp_type']}")
                # Use relabeled dataset as base
                al_path = os.path.join(data_root, ds_name, "full", "al_dataset.pkl")
                al_ds = mdu.read_pickle(al_path)

                # set correct attribute to fetch from the dataset
                if exp_dict["exp_type"] == "gpt3_oracle":
                    engine, temp = exp_dict["gpt3_engine"], exp_dict["gpt3_temp"]
                    attr = f"{engine}_{temp}"
                elif exp_dict["exp_type"] == "gpt3mix_oracle":
                    attr = f"gpt3mix_{exp_dict['gpt3_engine']}"
                else:  # eda_oracle
                    attr = exp_dict["exp_type"].split("_")[0]  # just eda

                for domain in data_set:
                    train_lines.extend(data_set[domain]["F"]["train"]["text"])
                    train_labels.extend(data_set[domain]["F"]["train"]["intent"])
                train_lines.extend(al_ds["generated"][attr]["text"])
                train_labels.extend(al_ds["generated"][attr]["intent"])

            elif exp_dict["exp_type"] == "gpt3mix":
                print("Loading labelled pool for full few-shot gpt3mix")
                engine = exp_dict["gpt3_engine"]
                gpt3mix_path = f"data/{ds_name}/full/gpt3mix_{engine}.pkl"

                # augs will also contain the seed samples
                if os.path.exists(gpt3mix_path):  # load from existing pkl
                    print(f"Loading existing GPT3Mix data for {engine.upper()}")
                    augs = mdu.read_pickle(gpt3mix_path)
                else:  # otherwise, generate gpt3mix pickle
                    print(f"Generating GPT3Mix data with {engine.upper()}")
                    interim_save_path = gpt3mix_path[:-4] + "_interim.pkl"
                    augs = generate_for_gpt3mix(
                        base_ds, data_set, exp_dict, interim_save_path
                    )
                    # save complete augmented data
                    mdu.write_pickle(augs, gpt3mix_path)

                train_lines, train_labels = augs["text"], augs["intent"]

            val_lines, val_labels = base_ds["val"]["text"], base_ds["val"]["intent"]
            test_lines, test_labels = (
                base_ds["test"]["text"],
                base_ds["test"]["intent"],
            )

            # add oos samples to train set (gpt3mix setting already adds)
            if oos_id is not None and exp_dict["exp_type"] != "gpt3mix":
                # add oos samples to the dataset
                oos_lines, oos_labels = extract_oos(base_ds["train"], oos_id)
                train_lines.extend(oos_lines)
                train_labels.extend(oos_labels)

            # remove oos samples appropriately
            if oos_id is None:
                name2id_path = os.path.join(data_root, ds_name, "name2id.json")
                temp_oos_id = mdu.read_json(name2id_path).get("oos", None)
                if exp_dict["exp_type"] == "gpt3mix":
                    train_set = {"text": train_lines, "intent": train_labels}
                    # remove oos samples add to train set here by default
                    train_lines, train_labels = filter_oos(
                        train_set, oos_id, soft_label=True
                    )
                # remove oos samples from the val set added by default
                val_lines, val_labels = filter_oos(base_ds["val"], temp_oos_id)
                test_lines, test_labels = filter_oos(base_ds["test"], temp_oos_id)

            print(len(train_lines), len(train_labels))
            self.dataset = DatasetDict(
                train=Dataset.from_dict({"text": train_lines, "intent": train_labels}),
                validation=Dataset.from_dict({"text": val_lines, "intent": val_labels}),
                test=Dataset.from_dict({"text": test_lines, "intent": test_labels}),
            )
        elif dataset_config == "full":
            # read the original FULL version of the dataset
            data_set = mdu.read_pickle(basic_data_path)

            if exp_dict["exp_type"] == "intrinsic":
                print("Loading utils for intrinsic evaluation")

                oos_id = exp_dict["dataset"]["oos_id"]
                train_lines, train_labels = filter_oos(data_set["train"], oos_id)
                val_lines, val_labels = filter_oos(data_set["val"], oos_id)
                test_lines, test_labels = filter_oos(data_set["test"], oos_id)

                self.dataset = DatasetDict(
                    train=Dataset.from_dict(
                        {"text": train_lines, "intent": train_labels}
                    ),
                    validation=Dataset.from_dict(
                        {"text": val_lines, "intent": val_labels}
                    ),
                    test=Dataset.from_dict({"text": test_lines, "intent": test_labels}),
                )
                # add different set of generated lines as test set
                augmented_data = mdu.mdu.read_pickle(ex2_data_path)
                domains = list(augmented_data.keys())
                for e in ["ada", "babbage", "curie", "davinci", "gptj"]:
                    # for t in np.linspace(0.5, 2, int((2.1-.5)/.1)):
                    for t in [1.0]:
                        _lines, _intents = [], []
                        for d in domains:
                            if d == "oos":
                                continue
                            _lines.extend(augmented_data[d]["F"][f"{e}_{t}"]["text"])
                            _intents.extend(
                                augmented_data[d]["F"][f"{e}_{t}"]["intent"]
                            )
                        self.dataset[f"{e}_{t}"] = Dataset.from_dict(
                            {"text": _lines, "intent": _intents}
                        )
            elif exp_dict["exp_type"] == "baseline":
                print("Loading utils for baseline version")
                self.dataset = DatasetDict(
                    train=Dataset.from_dict(data_set["train"]),
                    validation=Dataset.from_dict(data_set["val"]),
                    test=Dataset.from_dict(data_set["test"]),
                )

        elif dataset_config.startswith("full_"):
            print(f"Loading utils for {dataset_config}")
            # read the augmented version of the dataset
            data_set = mdu.read_pickle(ex2_data_path)
            # the few-shot domain
            val_domain = dataset_config.split("_", 1)[1]
            # train set = D_{M, train} + D_{F, train}
            train_lines = (
                data_set[val_domain]["M"]["train"]["text"]
                + data_set[val_domain]["F"]["train"]["text"]
            )

            train_labels = (
                data_set[val_domain]["M"]["train"]["intent"]
                + data_set[val_domain]["F"]["train"]["intent"]
            )

            if exp_dict["exp_type"] == "upsample":
                train_lines.extend(data_set[val_domain]["F"]["upsample"]["text"])
                train_labels.extend(data_set[val_domain]["F"]["upsample"]["intent"])
            elif exp_dict["exp_type"] == "gpt3":
                engine = exp_dict["gpt3_engine"]
                temp = exp_dict["gpt3_temp"]
                train_lines.extend(
                    data_set[val_domain]["F"][f"{engine}_{temp}"]["text"]
                )
                train_labels.extend(
                    data_set[val_domain]["F"][f"{engine}_{temp}"]["intent"]
                )

            full_val_lines = (
                data_set[val_domain]["M"]["val"]["text"]
                + data_set[val_domain]["F"]["val"]["text"]
            )

            full_val_labels = (
                data_set[val_domain]["M"]["val"]["intent"]
                + data_set[val_domain]["F"]["val"]["intent"]
            )

            full_test_lines = (
                data_set[val_domain]["M"]["test"]["text"]
                + data_set[val_domain]["F"]["test"]["text"]
            )

            full_test_labels = (
                data_set[val_domain]["M"]["test"]["intent"]
                + data_set[val_domain]["F"]["test"]["intent"]
            )

            # add oos samples to the dataset for oos-aware classifiers
            if exp_dict["dataset"]["oos_id"] is not None:
                print("adding OOS samples to the dataset")
                base_ds = mdu.mdu.read_pickle(basic_data_path)
                oos_id = exp_dict["dataset"]["oos_id"]

                # augment training set
                oos_train_lines, oos_train_labels = extract_oos(
                    base_ds["train"], oos_id
                )
                train_lines.extend(oos_train_lines)
                train_labels.extend(oos_train_labels)

                # augment validation set
                oos_val_lines, oos_val_labels = extract_oos(base_ds["val"], oos_id)
                full_val_lines.extend(oos_val_lines)
                full_val_labels.extend(oos_val_labels)

                # augment test set
                oos_test_lines, oos_test_labels = extract_oos(base_ds["test"], oos_id)
                full_test_lines.extend(oos_test_lines)
                full_test_labels.extend(oos_test_labels)

            self.dataset = DatasetDict(
                train=Dataset.from_dict({"text": train_lines, "intent": train_labels}),
                validation=Dataset.from_dict(data_set[val_domain]["F"]["val"]),
                test=Dataset.from_dict(data_set[val_domain]["F"]["test"]),
                full_test=Dataset.from_dict(
                    {"text": full_test_lines, "intent": full_test_labels}
                ),
                full_validation=Dataset.from_dict(
                    {"text": full_val_lines, "intent": full_val_labels}
                ),
            )
        else:
            warnings.warn("At the moment we can only load clinc_oos")
            self.dataset = load_dataset(ds_name, dataset_config, cache_dir=data_root)

    def get_split(self, split):
        return self.dataset[split]
