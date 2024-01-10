import json
import re

import requests
import spacy
from colorama import Fore
from spacy.matcher import Matcher
from thefuzz import fuzz

import openai_prompts as prompts
from coreference_resolution.coref import resolve_references
from create_bpmn_structure import create_bpmn_structure
from graph_generator import GraphGenerator
from logging_utils import clear_folder, write_to_file

BPMN_INFORMATION_EXTRACTION_ENDPOINT = "https://api-inference.huggingface.co/models/jtlicardo/bpmn-information-extraction-v2"
ZERO_SHOT_CLASSIFICATION_ENDPOINT = (
    "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
)


def get_sentences(text: str) -> list[str]:
    """
    Creates a list of sentences from a given text.
    Args:
        text (str): the text to split into sentences
    Returns:
        list: a list of sentences
    """

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [str(i) for i in list(doc.sents)]
    return sentences


def create_sentence_data(text: str) -> list[dict]:
    """
    Creates a list of dictionaries containing the sentence data (sentence, start index, end index)
    Args:
        text (str): the input text
    Returns:
        list: a list of dictionaries containing the sentence data
    """

    sentences = get_sentences(text)

    start = 0
    sentence_data = []

    for sent in sentences:
        end = start + len(sent)
        sentence_data.append({"sentence": sent, "start": start, "end": end})
        start += len(sent) + 1

    write_to_file("sentence_data.json", sentence_data)
    return sentence_data


def query(payload: dict, endpoint: str) -> list:
    """
    Sends a POST request to the specified endpoint with the given payload in JSON format, and returns the response.
    Args:
        payload (dict): the payload to send to the endpoint
        endpoint (str): the endpoint to send the request to
    Returns:
        list: the response from the endpoint
    """
    data = json.dumps(payload)
    response = requests.request("POST", endpoint, data=data)
    return json.loads(response.content.decode("utf-8"))


def extract_bpmn_data(text: str) -> list[dict]:
    """
    Extracts BPMN data from the process description by calling the model endpoint hosted on Hugging Face.
    Args:
        text (str): the process description
    Returns:
        list: model output
    """

    print("Extracting BPMN data...\n")
    data = query(
        {"inputs": text, "options": {"wait_for_model": True}},
        BPMN_INFORMATION_EXTRACTION_ENDPOINT,
    )
    write_to_file("model_output.json", data)

    if "error" in data:
        print(
            f"{Fore.RED}Error when extracting BPMN data: {data['error']}{Fore.RESET}\n"
        )
        return None

    return data


def fix_bpmn_data(data: list[dict]) -> list[dict]:
    """
    If the model that extracts BPMN data splits a word into multiple tokens for some reason,
    this function fixes the output by combining the tokens into a single word.
    Args:
        data (list): the model output
    Returns:
        list: the model output with the tokens combined into a single word
    """

    data_copy = data.copy()

    for i in range(len(data_copy) - 1):
        if (
            data_copy[i]["entity_group"] == "TASK"
            and data_copy[i + 1]["entity_group"] == "TASK"
            and data_copy[i]["end"] == data_copy[i + 1]["start"]
        ):
            print("Fixing BPMN data...")
            print(
                "Combining",
                data_copy[i]["word"],
                "and",
                data_copy[i + 1]["word"],
                "into one entity\n",
            )
            if data_copy[i + 1]["word"].startswith("##"):
                data[i]["word"] = data[i]["word"] + data[i + 1]["word"][2:]
            else:
                data[i]["word"] = data[i]["word"] + data[i + 1]["word"]
            data[i]["end"] = data[i + 1]["end"]
            data[i]["score"] = max(data[i]["score"], data[i + 1]["score"])
            data.pop(i + 1)

    if data != data_copy:
        write_to_file("model_output_fixed.json", data)

    return data


def classify_process_info(text: str) -> dict:
    """
    Classifies a PROCESS_INFO entity by calling the model endpoint hosted on Hugging Face.
    Possible classes: PROCESS_START, PROCESS_END, PROCESS_SPLIT, PROCESS_RETURN
    Args:
        text (str): sequence of text classified as process info
    Returns:
        dict: model output containing the following keys: "sequence", "labels", "scores"
    """

    data = query(
        {
            "inputs": text,
            "parameters": {
                "candidate_labels": [
                    "beginning",
                    "end",
                    "split",
                    "going back",
                    "continuation",
                ]
            },
            "options": {"wait_for_model": True},
        },
        ZERO_SHOT_CLASSIFICATION_ENDPOINT,
    )

    if "error" in data:
        print(
            f"{Fore.RED}Error when classifying PROCESS_INFO entity: {data['error']}{Fore.RESET}\n"
        )
        return None

    return data


def batch_classify_process_info(process_info_entities: list[dict]) -> list[dict]:
    """
    Classifies a list of PROCESS_INFO entities into PROCESS_START, PROCESS_END, PROCESS_SPLIT or PROCESS_RETURN.
    Args:
        process_info_entities (list): a list of PROCESS_INFO entities
    Returns:
        list: a list of PROCESS_INFO entities with the entity_group key updated
    """

    updated_entities = []

    print("Classifying PROCESS_INFO entities...\n")

    for entity in process_info_entities:
        text = entity["word"]
        data = classify_process_info(text)
        process_info_dict = {
            "beginning": "PROCESS_START",
            "end": "PROCESS_END",
            "split": "PROCESS_SPLIT",
            "going back": "PROCESS_RETURN",
            "continuation": "PROCESS_CONTINUE",
        }
        entity["entity_group"] = process_info_dict[data["labels"][0]]
        updated_entities.append(entity)

    return updated_entities


def extract_entities(type: str, data: list[dict], min_score: float) -> list[dict]:
    """
    Extracts all entities of a given type from the model output
    Args:
        type (str): the type of entity to extract
        data (list): the model output
        min_score (float): the minimum score an entity must have to be extracted
    Returns:
        list: a list of entities of the given type
    """
    return [
        entity
        for entity in data
        if entity["entity_group"] == type and entity["score"] > min_score
    ]


def create_agent_task_pairs(
    agents: list[dict],
    tasks: list[dict],
    sentence_data: list[dict],
) -> list[dict]:
    """
    Combines agents and tasks into agent-task pairs based on the sentence they appear in.
    Args:
        agents (list): a list of agents
        tasks (list): a list of tasks
        sentence_data (list): a list of sentence data containing the start and end indices of each sentence, and the sentence itself
    Returns:
        list: a list of agent-task pairs (each pair is a dictionary containing the agent, task and sentence index)
    """

    agents_in_sentences = [
        {"sentence_idx": i, "agent": agent}
        for agent in agents
        for i, sent in enumerate(sentence_data)
        if sent["start"] <= agent["start"] <= sent["end"]
    ]

    tasks_in_sentences = [
        {"sentence_idx": i, "task": task}
        for task in tasks
        for i, sent in enumerate(sentence_data)
        if sent["start"] <= task["start"] <= sent["end"]
    ]

    added_indices = set()

    multi_agent_sentences_idx = [
        agent["sentence_idx"]
        for i, agent in enumerate(agents_in_sentences[:-1])
        if agent["sentence_idx"] == agents_in_sentences[i + 1]["sentence_idx"]
        and agent["sentence_idx"] not in added_indices
        and not added_indices.add(agent["sentence_idx"])
    ]

    agent_task_pairs = [
        {
            "agent": agent["agent"],
            "task": task["task"],
            "sentence_idx": task["sentence_idx"],
        }
        for agent in agents_in_sentences
        for task in tasks_in_sentences
        if (
            agent["sentence_idx"] == task["sentence_idx"]
            and agent["sentence_idx"] not in multi_agent_sentences_idx
        )
    ]

    if len(multi_agent_sentences_idx) > 0:
        multi_agent_task_pairs = handle_multi_agent_sentences(
            agents_in_sentences, tasks_in_sentences, multi_agent_sentences_idx
        )
        agent_task_pairs.extend(multi_agent_task_pairs)

    agent_task_pairs = sorted(agent_task_pairs, key=lambda k: k["sentence_idx"])

    return agent_task_pairs


def handle_multi_agent_sentences(
    agents_in_sentences: list[dict],
    tasks_in_sentences: list[dict],
    multi_agent_sentences_idx: list[int],
) -> list[dict]:
    """
    Creates agent-task pairs for sentences that contain multiple agents.
    For example, if the sentence is "A does B and C does D", then the agent-task pairs are (A, B) and (C, D)
    Args:
        agents_in_sentences (list): a list of agents with their sentence indices
        tasks_in_sentences (list): a list of tasks with their sentence indices
        multi_agent_sentences_idx (list): a list of sentence indices that contain multiple agents
    Returns:
        list: a list of agent-task pairs (each pair is a dictionary containing the agent, task and sentence index)
    """

    multi_agent_sentences = [
        {
            "sentence_idx": idx,
            "agents": [
                agent for agent in agents_in_sentences if agent["sentence_idx"] == idx
            ],
            "tasks": [
                task for task in tasks_in_sentences if task["sentence_idx"] == idx
            ],
        }
        for idx in multi_agent_sentences_idx
    ]

    agent_task_pairs = [
        {
            "agent": agent["agent"],
            "task": sentence["tasks"][i]["task"],
            "sentence_idx": sentence["sentence_idx"],
        }
        for sentence in multi_agent_sentences
        for i, agent in enumerate(sentence["agents"])
    ]

    return agent_task_pairs


def add_process_end_events(
    agent_task_pairs: list[dict],
    sentences: list[dict],
    process_info_entities: list[dict],
) -> list[dict]:
    """
    Adds process end events to agent-task pairs
    Args:
        agent_task_pairs (list): a list of agent-task pairs
        sentences (list): list of sentence data
        process_info_entities (list): list of process info entities
    Returns:
        list: a list of agent-task pairs with process end events
    """

    process_end_events = [
        {
            "process_end_event": entity,
            "sentence_idx": sentences.index(sent),
        }
        for entity in process_info_entities
        if entity["entity_group"] == "PROCESS_END"
        for sent in sentences
        if sent["start"] <= entity["start"] <= sent["end"]
    ]

    for pair in agent_task_pairs:
        for event in process_end_events:
            if pair["sentence_idx"] == event["sentence_idx"]:
                pair["process_end_event"] = event["process_end_event"]

    return agent_task_pairs


def has_parallel_keywords(text: str) -> bool:
    """
    Checks if a text contains parallel keywords
    Args:
        text (str): the text to check
    Returns:
        bool: True if the text contains parallel keywords, False otherwise
    """

    nlp = spacy.load("en_core_web_md")
    matcher = Matcher(nlp.vocab)

    patterns = [
        [{"LOWER": "in"}, {"LOWER": "the"}, {"LOWER": "meantime"}],
        [{"LOWER": "at"}, {"LOWER": "the"}, {"LOWER": "same"}, {"LOWER": "time"}],
        [{"LOWER": "meanwhile"}],
        [{"LOWER": "while"}],
        [{"LOWER": "in"}, {"LOWER": "parallel"}],
        [{"LOWER": "parallel"}, {"LOWER": "paths"}],
        [{"LOWER": "concurrently"}],
        [{"LOWER": "simultaneously"}],
    ]

    matcher.add(
        "PARALLEL",
        patterns,
    )

    doc = nlp(text)

    matches = matcher(doc)

    return len(matches) > 0


def find_sentences_with_loop_keywords(sentences: list[dict]) -> list[dict]:
    """
    Returns sentences that contain loop keywords
    Args:
        sentences (list): list of sentence data
    Returns:
        list: list of sentences that contain loop keywords
    """

    nlp = spacy.load("en_core_web_md")
    matcher = Matcher(nlp.vocab)

    patterns = [
        [{"LOWER": "again"}],
    ]

    matcher.add("LOOP", patterns)

    detected_sentences = [
        sent for sent in sentences if len(matcher(nlp(sent["sentence"]))) > 0
    ]

    return detected_sentences


def add_task_ids(
    agent_task_pairs: list[dict], sentences: list[dict], loop_sentences: list[dict]
) -> list[dict]:
    """
    Adds task ids to tasks that are not in a sentence with a loop keyword.
    Args:
        agent_task_pairs (list): a list of agent-task pairs
        sentences (list): list of sentence data
        loop_sentences (list): list of sentences that contain loop keywords
    Returns:
        list: a list of agent-task pairs with task ids
    """

    id = 0

    for pair in agent_task_pairs:
        task = pair["task"]
        for sent in sentences:
            if sent["start"] <= task["start"] <= sent["end"]:
                if sent not in loop_sentences:
                    pair["task"]["task_id"] = f"T{id}"
                    id += 1

    return agent_task_pairs


def add_loops(
    agent_task_pairs: list[dict], sentences: list[dict], loop_sentences: list[dict]
) -> list[dict]:
    """
    Adds go_to keys to tasks that are in a sentence with a loop keyword.
    Args:
        agent_task_pairs (list): a list of agent-task pairs
        sentences (list): list of sentence data
        loop_sentences (list): list of sentences that contain loop keywords
    Returns:
        list: a list of agent-task pairs with go_to keys
    """
    previous_tasks = []

    for pair in agent_task_pairs:
        task = pair["task"]
        for sent in sentences:
            if task["start"] in range(sent["start"], sent["end"] + 1):
                if sent in loop_sentences:

                    previous_task = find_previous_task(previous_tasks, task)

                    pair["go_to"] = previous_task["task_id"]
                    pair["start"] = pair["task"]["start"]
                    del pair["task"]
                    del pair["agent"]
                else:
                    previous_tasks.append(task)

    return agent_task_pairs


def find_previous_task(previous_tasks: list[dict], task: dict) -> dict:
    """
    Finds the previous task in the list of previous tasks that has the highest similarity to the current task.
    Args:
        previous_tasks (list): list of previous tasks
        task (dict): the current task
    Returns:
        dict: the previous task with the highest similarity to the current task
    """

    previous_tasks_str = "\n".join(map(lambda task: task["word"], previous_tasks))
    previous_task = prompts.find_previous_task(task["word"], previous_tasks_str)

    highest_similarity_task = None
    highest_similarity = 0
    for task in previous_tasks:
        similarity = fuzz.ratio(previous_task, task["word"])
        if similarity > highest_similarity:
            highest_similarity = similarity
            highest_similarity_task = task

    return highest_similarity_task


def extract_exclusive_gateways(process_description: str, conditions: list) -> list:
    """
    Extracts the conditions for each exclusive gateway in the process description
    Args:
        process_description (str): the process description
        conditions (list): the list of condition entities found in the process description
    Returns:
        list: the list of exclusive gateways

    Example output:
    [
        {
            "id": "EG0",
            "conditions": [
                "if the customer is a new customer",
                "if the customer is an existing customer"
            ],
            "start": 54,
            "end": 251,
            "paths": [{'start': 54, 'end': 210}, {'start': 211, 'end': 321}]
        },
    ]
    """

    first_condition_start = conditions[0]["start"]
    exclusive_gateway_text = process_description[first_condition_start:]

    if len(conditions) == 2:
        response = prompts.extract_exclusive_gateways_2_conditions(
            exclusive_gateway_text
        )
    else:
        response = prompts.extract_exclusive_gateways(exclusive_gateway_text)

    pattern = r"Exclusive gateway \d+: (.+?)(?=(?:Exclusive gateway \d+:|$))"
    matches = re.findall(pattern, response, re.DOTALL)
    gateways = [s.strip() for s in matches]
    gateway_indices = get_indices(gateways, process_description)
    print("Exclusive gateway indices:", gateway_indices, "\n")

    exclusive_gateways = []

    if len(conditions) == 2 and len(gateways) == 1:
        conditions = [x["word"] for x in conditions]
        exclusive_gateways = [{"id": "EG0", "conditions": conditions}]
    else:
        condition_string = ""
        for condition in conditions:
            condition_text = condition["word"]
            if condition_string == "":
                condition_string += f"'{condition_text}'"
            else:
                condition_string += f", '{condition_text}'"
        response = prompts.extract_gateway_conditions(
            process_description, condition_string
        )
        pattern = r"Exclusive gateway \d+: (.+?)(?=(?:Exclusive gateway \d+:|$))"
        matches = re.findall(pattern, response, re.DOTALL)
        gateway_conditions = [s.strip() for s in matches]
        exclusive_gateways = [
            {"id": f"EG{i}", "conditions": [x.strip() for x in gateway.split("||")]}
            for i, gateway in enumerate(gateway_conditions)
        ]

    for gateway in exclusive_gateways:
        condition_indices = get_indices(gateway["conditions"], process_description)
        gateway["start"] = gateway_indices[exclusive_gateways.index(gateway)]["start"]
        gateway["end"] = gateway_indices[exclusive_gateways.index(gateway)]["end"]
        gateway["paths"] = condition_indices

    # Check for nested exclusive gateways
    for i, gateway in enumerate(exclusive_gateways):
        if i != len(exclusive_gateways) - 1:
            if gateway["end"] > exclusive_gateways[i + 1]["start"]:
                print("Nested exclusive gateway found\n")
                # To the nested gateway, add parent gateway id
                exclusive_gateways[i + 1]["parent_gateway_id"] = gateway["id"]

    # Update the start and end indices of the paths
    for eg_idx, exclusive_gateway in enumerate(exclusive_gateways):
        for i, path in enumerate(exclusive_gateway["paths"]):
            if i != len(exclusive_gateway["paths"]) - 1:
                path["end"] = exclusive_gateway["paths"][i + 1]["start"] - 1
            else:
                if "parent_gateway_id" in exclusive_gateway:
                    # Set the end index to the end of the parent gateway path
                    parent_gateway = next(
                        (
                            gateway
                            for gateway in exclusive_gateways
                            if gateway["id"] == exclusive_gateway["parent_gateway_id"]
                        ),
                        None,
                    )
                    assert parent_gateway is not None, "No parent gateway found"
                    for parent_path in parent_gateway["paths"]:
                        if path["start"] in range(
                            parent_path["start"], parent_path["end"]
                        ):
                            path["end"] = parent_path["end"] - 1
                else:
                    # If this is not the last gateway and the next gateway is not nested
                    if (
                        exclusive_gateway != exclusive_gateways[-1]
                        and "parent_gateway_id" not in exclusive_gateways[eg_idx + 1]
                    ):
                        # Set the end index to the start index of the next gateway
                        path["end"] = (
                            exclusive_gateways[
                                exclusive_gateways.index(exclusive_gateway) + 1
                            ]["start"]
                            - 1
                        )
                    # If this is not the last gateway and the next gateway is nested
                    elif (
                        exclusive_gateway != exclusive_gateways[-1]
                        and "parent_gateway_id" in exclusive_gateways[eg_idx + 1]
                    ):
                        # Find the next gateway that is not nested
                        # If there is no such gateway, set the end index to the end of the gateway
                        next_gateway = next(
                            (
                                gateway
                                for gateway in exclusive_gateways[
                                    exclusive_gateways.index(exclusive_gateway) + 1 :
                                ]
                                if "parent_gateway_id" not in gateway
                            ),
                            None,
                        )
                        if next_gateway is not None:
                            path["end"] = next_gateway["start"] - 1
                        else:
                            path["end"] = exclusive_gateway["end"]
                    # If this is the last gateway
                    elif exclusive_gateway == exclusive_gateways[-1]:
                        # Set the end index to the end of the gateway
                        path["end"] = exclusive_gateway["end"]

    # Add parent gateway path id to the nested gateways
    for eg_idx, exclusive_gateway in enumerate(exclusive_gateways):
        if "parent_gateway_id" in exclusive_gateway:
            parent_gateway = next(
                (
                    gateway
                    for gateway in exclusive_gateways
                    if gateway["id"] == exclusive_gateway["parent_gateway_id"]
                ),
                None,
            )
            assert parent_gateway is not None, "No parent gateway found"
            for path in parent_gateway["paths"]:
                if exclusive_gateway["start"] in range(path["start"], path["end"]):
                    exclusive_gateway["parent_gateway_path_id"] = parent_gateway[
                        "paths"
                    ].index(path)

    print("Exclusive gateways data:", exclusive_gateways, "\n")
    return exclusive_gateways


def add_conditions(conditions: list, agent_task_pairs: list, sentences: list) -> list:
    """
    Adds conditions and condition ids to agent-task pairs
    Args:
        conditions (list): a list of conditions
        agent_task_pairs (list): a list of agent-task pairs
        sentences (list): a list of sentences
    Returns:
        list: a list of agent-task pairs with conditions
    """

    condition_id = 0

    for pair in agent_task_pairs:

        for sent in sentences:

            if pair["task"]["start"] in range(sent["start"], sent["end"]):

                for condition in conditions:
                    if condition["start"] in range(sent["start"], sent["end"]):
                        pair["condition"] = condition
                        pair["condition"]["condition_id"] = f"C{condition_id}"
                        condition_id += 1
                        break

    return agent_task_pairs


def handle_text_with_conditions(
    agent_task_pairs: list, conditions: list, sents_data: list, process_desc: str
) -> list:
    """
    Adds conditions and exclusive gateway ids to agent-task pairs.
    Args:
        agent_task_pairs (list): the list of agent-task pairs
        conditions (list): the list of condition entities
        sents_data (list): the sentence data
        process_desc (str): the process description
    Returns:
        list: the list of agent-task pairs with conditions and exclusive gateway ids
    """

    updated_agent_task_pairs = add_conditions(conditions, agent_task_pairs, sents_data)

    exclusive_gateway_data = extract_exclusive_gateways(process_desc, conditions)

    return updated_agent_task_pairs, exclusive_gateway_data


def should_resolve_coreferences(text: str) -> bool:
    """
    Determines whether coreferences should be resolved.
    Args:
        text (str): the text
    Returns:
        bool: True if coreferences should be resolved, False otherwise
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    for token in doc:
        if token.lower_ in ["he", "she", "it", "they"]:
            return True
    return False


def extract_all_entities(data: list, min_score: float) -> tuple:
    """
    Extracts all entities from the model output.
    Args:
        data (list): model output
        min_score (float): the minimum score for an entity to be extracted
    Returns:
        tuple: a tuple of lists containing the extracted entities
    """

    print("Extracting entities...\n")
    agents = extract_entities("AGENT", data, min_score)
    tasks = extract_entities("TASK", data, min_score)
    conditions = extract_entities("CONDITION", data, min_score)
    process_info = extract_entities("PROCESS_INFO", data, min_score)
    return (agents, tasks, conditions, process_info)


def get_indices(strings: list, text: str) -> list:
    """
    Gets the start and end indices of the given strings in the text by using fuzzy string matching.
    Args:
        strings (list): the list of strings to be found in the text
        text (str): the text in which the strings are to be found
    Returns:
        list: the list of start and end indices
    """

    indices = []
    matches = []

    for str in strings:
        potential_matches = []
        length = len(str)
        first_word = str.split()[0].lower()
        first_word_indices = [
            i for i in range(len(text)) if text.lower().startswith(first_word, i)
        ]
        for idx in first_word_indices:
            potential_match = text[idx : idx + length + 1]
            prob = fuzz.ratio(str, potential_match)
            potential_matches.append(
                {"potential_match": potential_match, "probability": prob}
            )
        matches.append(max(potential_matches, key=lambda x: x["probability"]))

    for match in matches:
        txt = match["potential_match"]
        indices.append(
            {
                "start": text.find(txt),
                "end": text.find(txt) + len(txt),
            }
        )

    return indices


def get_parallel_paths(parallel_gateway: str, process_description: str) -> list[dict]:
    """
    Gets the start and end indices of the parallel paths in the process description.
    Args:
        parallel_gateway (str): the parallel gateway
        process_description (str): the process description
    Returns:
        list: the list of start and end indices
    """
    num = int(prompts.number_of_parallel_paths(parallel_gateway))
    assert num <= 3, "The maximum number of parallel paths is 3"
    paths = ""
    if num == 1:
        return None
    elif num == 2:
        paths = prompts.extract_2_parallel_paths(parallel_gateway)
    elif num == 3:
        paths = prompts.extract_3_parallel_paths(parallel_gateway)
    paths = paths.split("&&")
    paths = [s.strip() for s in paths]
    indices = get_indices(paths, process_description)
    print("Parallel path indices:", indices, "\n")
    return indices


def get_parallel_gateways(text: str) -> list[dict]:
    """
    Gets the indices of the parallel gateways in the text.
    Args:
        text (str): the text
    Returns:
        list: the list of start and end indices
    """
    response = prompts.extract_parallel_gateways(text)
    pattern = r"Parallel gateway \d+: (.+?)(?=(?:Parallel gateway \d+:|$))"
    matches = re.findall(pattern, response, re.DOTALL)
    gateways = [s.strip() for s in matches]
    indices = get_indices(gateways, text)
    print("Parallel gateway indices:", indices, "\n")
    return indices


def handle_text_with_parallel_keywords(
    process_description, agent_task_pairs, sents_data
):
    """
    Extracts parallel gateways and paths from the process description.
    Args:
        process_description (str): the process description
        agent_task_pairs (list): the list of agent-task pairs
        sents_data (list): the sentence data
    Returns:
        list: the list of parallel gateways

    Example output:
    [
        {
            "id": "PG0",
            "start": 0,
            "end": 100,
            "paths": [
                {
                    "start": 0,
                    "end": 50,
                },
                {
                    "start": 50,
                    "end": 100,
                },
            ],
        },
    ]
    """

    parallel_gateways = []
    parallel_gateway_id = 0

    gateway_indices = get_parallel_gateways(process_description)

    for indices in gateway_indices:

        num_of_sentences = count_sentences_spanned(sents_data, indices, 3)
        assert num_of_sentences != 0, "No sentences found in parallel gateway"

        num_of_atp = num_of_agent_task_pairs_in_range(agent_task_pairs, indices)
        assert num_of_atp != 0, "No agent-task pairs found in parallel gateway"

        if num_of_sentences == 1 and num_of_atp == 1:
            print("Parallel gateway is a single sentence and single agent-task pair\n")

            sentence_text = get_sentence_text(sents_data, indices)
            assert sentence_text is not None
            response = prompts.extract_parallel_tasks(sentence_text)
            tasks = extract_tasks(response)
            idx = get_agent_task_pair_index(agent_task_pairs, indices)
            assert idx is not None
            # Take all keys except "task" from the agent-task pair
            atp = {k: v for k, v in agent_task_pairs[idx].items() if k != "task"}
            task_start = agent_task_pairs[idx]["task"]["start"]

            pg_path_indices = []

            insert_idx = idx

            # Remove original agent-task pair
            agent_task_pairs.pop(idx)

            # Create agent task pairs for each task
            for task in tasks:
                atp_to_insert = {
                    **atp,
                    "task": {
                        "entity_group": "TASK",
                        "start": task_start,
                        "end": task_start + 1,
                        "word": task,
                    },
                }
                pg_path_indices.append(
                    {
                        "start": task_start,
                        "end": task_start + 1,
                    }
                )
                agent_task_pairs.insert(insert_idx, atp_to_insert)
                insert_idx += 1
                task_start += 2

            # Create parallel gateway
            gateway = {
                "id": f"PG{parallel_gateway_id}",
                "start": pg_path_indices[0]["start"],
                "end": pg_path_indices[-1]["end"],
                "paths": pg_path_indices,
            }
            parallel_gateway_id += 1
            parallel_gateways.append(gateway)

        else:
            gateway_text = process_description[indices["start"] : indices["end"]]
            gateway = {
                "id": f"PG{parallel_gateway_id}",
                "start": indices["start"],
                "end": indices["end"],
                "paths": get_parallel_paths(gateway_text, process_description),
            }
            parallel_gateway_id += 1
            parallel_gateways.append(gateway)

    for gateway in parallel_gateways.copy():
        for path in gateway["paths"]:
            path_text = process_description[path["start"] : path["end"]]
            if has_parallel_keywords(path_text):
                print("Parallel keywords detected in path:", path_text, "\n")
                indices = get_parallel_paths(path_text, process_description)
                gateway = {
                    "id": f"PG{parallel_gateway_id}",
                    "start": indices[0]["start"],
                    "end": indices[-1]["end"],
                    "paths": indices,
                    "parallel_parent": gateway["id"],
                    "parallel_parent_path": gateway["paths"].index(path),
                }
                parallel_gateway_id += 1
                parallel_gateways.append(gateway)

    print("Parallel gateway data:", parallel_gateways, "\n")

    return parallel_gateways


def num_of_agent_task_pairs_in_range(
    agent_task_pairs: list[dict], indices: dict[str, int]
) -> int:
    """
    Gets the number of agent-task pairs in a given range of indices.
    Args:
        agent_task_pairs (list): the list of agent-task pairs
        indices (dict): the start and end indices
    Returns:
        int: the number of agent-task pairs in the given range
    """
    num_agent_task_pairs = 0
    for pair in agent_task_pairs:
        if (
            pair["task"]["start"] >= indices["start"]
            and pair["task"]["end"] <= indices["end"]
        ):
            num_agent_task_pairs += 1
    return num_agent_task_pairs


def get_agent_task_pair_index(
    agent_task_pairs: list[dict], indices: dict[str, int]
) -> int:
    """
    Gets the index of the agent-task pair in a given range of indices (should only be one).
    Args:
        agent_task_pairs (list): the list of agent-task pairs
        indices (dict): the start and end indices
    Returns:
        int: the index of the agent-task pair in the given range
    """
    for idx, pair in enumerate(agent_task_pairs):
        if (
            pair["task"]["start"] >= indices["start"]
            and pair["task"]["end"] <= indices["end"]
        ):
            return idx
    return None


def count_sentences_spanned(
    sentence_data: list[dict], indices: dict[str, int], buffer: int
) -> int:
    """
    Counts the number of sentences spanned by a given range of indices.
    Args:
        sentence_data (list): the list of sentence data
        indices (dict): the start and end indices
        buffer (int): the buffer to remove from the start and end indices
    Returns:
        int: the number of sentences spanned by the given range
    """
    count = 0

    idx_start = indices["start"] + buffer
    idx_end = indices["end"] - buffer

    for sentence_info in sentence_data:
        if (sentence_info["start"] <= idx_start <= sentence_info["end"]) or (
            sentence_info["start"] <= idx_end <= sentence_info["end"]
        ):
            count += 1

    return count


def get_sentence_text(sentence_data: list[dict], indices: dict[str, int]) -> str:
    """
    Gets the text of the sentence that is inside a given range of indices (should only be one sentence)
    Args:
        sentence_data (list): the list of sentence data
        indices (dict): the start and end indices
    Returns:
        str: the text of the sentence that is inside the given range
    """
    for sentence_info in sentence_data:
        if (
            (sentence_info["start"] <= indices["start"] <= sentence_info["end"])
            or (sentence_info["start"] <= indices["end"] <= sentence_info["end"])
            or (
                indices["start"] <= sentence_info["start"]
                and indices["end"] >= sentence_info["end"]
            )
        ):
            return sentence_info["sentence"]
    raise None


def extract_tasks(model_response: str) -> list[str]:
    """
    Processes the model response to extract the tasks being done in parallel.
    Example model response:
        Task 1: prepare the entree
        Task 2: prepare the dessert dishes
    Example output:
        ["prepare the entree", "prepare the dessert dishes"]
    Args:
        model_response (str): the model response
    Returns:
        list: the list of tasks
    """
    pattern = r"Task \d+: (.+?)(?=(?:Task \d+:|$))"
    matches = re.findall(pattern, model_response, re.DOTALL)
    tasks = [s.strip() for s in matches]
    return tasks


def process_text(text: str) -> list[dict]:
    """
    Processes the text to create the BPMN structure (a list of dictionaries).
    Args:
        text (str): the text of the process description
    Returns:
        list: the list of dictionaries representing the BPMN structure
    """

    clear_folder("./output_logs")

    print("\nInput text:\n" + text + "\n")

    if should_resolve_coreferences(text):
        print("Resolving coreferences...\n")
        text = resolve_references(text)
        write_to_file("coref_output.txt", text)
    else:
        print("No coreferences to resolve\n")

    data = extract_bpmn_data(text)

    if not data:
        return

    data = fix_bpmn_data(data)

    agents, tasks, conditions, process_info = extract_all_entities(data, 0.6)
    parallel_gateway_data = []
    exclusive_gateway_data = []

    sents_data = create_sentence_data(text)

    agent_task_pairs = create_agent_task_pairs(agents, tasks, sents_data)

    write_to_file("agent_task_pairs_initial.json", agent_task_pairs)

    if has_parallel_keywords(text):
        parallel_gateway_data = handle_text_with_parallel_keywords(
            text, agent_task_pairs, sents_data
        )
        write_to_file("parallel_gateway_data.json", parallel_gateway_data)

    if len(conditions) > 0:
        agent_task_pairs, exclusive_gateway_data = handle_text_with_conditions(
            agent_task_pairs, conditions, sents_data, text
        )
        write_to_file("exclusive_gateway_data.json", exclusive_gateway_data)

    if len(process_info) > 0:
        process_info = batch_classify_process_info(process_info)
        agent_task_pairs = add_process_end_events(
            agent_task_pairs, sents_data, process_info
        )

    write_to_file("process_info_entities.json", process_info)

    loop_sentences = find_sentences_with_loop_keywords(sents_data)
    agent_task_pairs = add_task_ids(agent_task_pairs, sents_data, loop_sentences)
    agent_task_pairs = add_loops(agent_task_pairs, sents_data, loop_sentences)

    write_to_file("agent_task_pairs_final.json", agent_task_pairs)

    structure = create_bpmn_structure(
        agent_task_pairs, parallel_gateway_data, exclusive_gateway_data, process_info
    )

    write_to_file("bpmn_structure.json", structure)

    return structure


def generate_graph_pdf(input: list[dict], notebook: bool) -> None:
    """
    Generates a graph of the BPMN structure and saves it as a PDF.
    Args:
        input (list): the list of dictionaries representing the BPMN structure
        notebook (bool): whether the graph is being generated in a Jupyter notebook
    """
    bpmn = GraphGenerator(input, notebook=notebook)
    print("Generating graph...\n")
    bpmn.generate_graph()
    bpmn.show()


def generate_graph_image(input: list[dict]) -> None:
    """
    Generates a graph of the BPMN structure and saves it as an image.
    Args:
        input (list): the list of dictionaries representing the BPMN structure
    """
    bpmn = GraphGenerator(input, format="jpeg", notebook=False)
    print("Generating graph...\n")
    bpmn.generate_graph()
    bpmn.save_file()
