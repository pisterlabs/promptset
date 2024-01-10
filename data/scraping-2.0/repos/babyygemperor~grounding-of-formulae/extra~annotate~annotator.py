# The preprocess tool for MioGatto
from pathlib import Path

from docopt import docopt

from lib.logger import get_logger
from lib.version import VERSION

import openai
from bs4 import BeautifulSoup, NavigableString
from unidecode import unidecode
import re
import os
from datetime import datetime
import unicodedata
import tiktoken
import ast
import json

# meta
PROG_NAME = "extra.annotate.annotator"
HELP = """Annotation tool for MioGatto

Usage:
    {p} [options] HTML

Options:
    -d DIR, --data=DIR  Dir for data outputs [default: ./data]
    --sources=DIR       Dir for HTML outputs [default: ./sources]
    --model=MODEL       GPT Model for predictions [default: gpt-3.5-turbo]

    -D, --debug         Show debug messages
    -q, --quiet         Show less messages

    -h, --help          Show this screen and exit
    -V, --version       Show version
""".format(p=PROG_NAME)

logger = get_logger(PROG_NAME)

args = docopt(HELP, version=VERSION)
model = args['--model']
openai.api_key = os.environ["OPENAI_API_KEY"]

enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")


def num_tokens_from_messages(messages, string=True):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if string:
        num_tokens = 0
        num_tokens += len(encoding.encode(messages))
        return num_tokens
    else:
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens


def split_into_messages(text):
    return text.split("\n")


def split_into_chunks(text, max_tokens=2000):
    messages = split_into_messages(text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    for message in messages:
        message_tokens = num_tokens_from_messages([{'role': 'user', 'content': message}], False)
        if current_tokens + message_tokens > max_tokens:
            # If adding this message would exceed the max tokens, start a new chunk
            chunks.append('\n'.join(current_chunk))
            current_chunk = [message]
            current_tokens = message_tokens
        else:
            # Otherwise, add the message to the current chunk
            current_chunk.append(message)
            current_tokens += message_tokens
    # Don't forget the last chunk!
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    return chunks


def string_to_dict(my_string):
    # Split string into lines
    lines = my_string.strip().split('\n')

    # Process each line
    processed_lines = []
    for line in lines:
        # Remove leading/trailing spaces
        line = line.strip()

        # Escape backslashes
        line = line.replace("\\", "\\\\")
        line = re.sub("<|([^>]*)|>", r"\1", line)
        line = line.replace('<|', '').replace('|>', '')

        # Add processed line to list
        processed_lines.append(line)

    # Join processed lines into a single string
    processed_string = '\n'.join(processed_lines)

    # Remove "identifiers = " part
    dict_string = processed_string[processed_string.index("{"):]

    # Use ast.literal_eval() to convert string to dictionary
    my_dict = ast.literal_eval(dict_string)

    return my_dict


def merge_dictionaries(dict1, dict2):
    union_dict = dict1.copy()

    for key, value in dict2.items():
        if key in union_dict:
            if isinstance(union_dict[key], list):
                if value not in union_dict[key]:
                    union_dict[key].append(value)
            else:
                if union_dict[key] != value:
                    union_dict[key] = [union_dict[key], value]
        else:
            union_dict[key] = value

    return union_dict


def get_text_from_tags(element):
    if isinstance(element, NavigableString):
        return element
    if element.name == 'mi':
        return str(element)
    return ''.join(get_text_from_tags(child) for child in element.children)


def parse_html(file_path, clean=True):
    with open(file_path, 'r') as html_file:
        soup = BeautifulSoup(html_file, 'html.parser')

    texts = get_text_from_tags(soup)
    if clean:
        matches = re.findall(r'<mi(.*?)</mi>', texts)
        for match in matches:
            original_string = f'<mi{match}</mi>'
            replaced_string = re.sub(r'<.*?>(.*?)</.*?>', r'<|\1|>', original_string)
            texts = texts.replace(original_string, replaced_string)

    return texts


def flatten_list(input_list):
    output_list = []
    for i in input_list:
        if isinstance(i, list):
            output_list.extend(flatten_list(i))
        else:
            output_list.append(i)
    return output_list


def remove_duplicates(input_list):
    output_list = []
    for item in input_list:
        if item not in output_list:
            output_list.append(item)
    if len(output_list) == 1:
        return output_list[0]
    return output_list


def process_value(v):
    if isinstance(v, str):
        new_v = v.replace('$', '').replace('<|', '').replace('|>', '')
        new_v = re.sub("<|([^>]*)|>", r"\1", new_v)
        while '\\\\' in new_v:
            new_v = new_v.replace('\\\\', '\\').replace('\n', '')
    else:  # Assuming it's a list
        new_v = flatten_list([process_value(val) for val in v])

    return remove_duplicates(new_v) if isinstance(new_v, list) else new_v


def reduce_pairs(dictionary):
    new_dict = {}
    for k, v in dictionary.items():
        # reduce key backslashes
        new_k = k.replace('$', '').replace('<|', '').replace('|>', '')
        new_k = re.sub("<|([^>]*)|>", r"\1", new_k)
        while '\\\\' in new_k:
            new_k = new_k.replace('\\\\', '\\')

        # process value
        new_v = process_value(v)

        new_dict[new_k] = new_v

    return new_dict


def remove_duplicates_dict(d):
    for k, v in d.items():
        if isinstance(v, list):
            d[k] = list(set(v))
    return d


def get_hex_code(key):
    return key.encode().hex()


def find_mi_strings(text):
    pattern = r'(<mi.*?</mi>)'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def expand_string_to_tokens(message, index, num_tokens_right=25, num_tokens_left=75):
    words = message.split()  # Split the message into words

    # Start at the index where the center word is
    left_index = right_index = index

    tokens_counter_right = num_tokens_from_messages(words[right_index], True)
    tokens_counter_left = num_tokens_from_messages(words[left_index], True)

    # Expand to the left from the center index until you reach num_tokens_left
    while tokens_counter_left < num_tokens_left and left_index > 0:
        left_index -= 1
        tokens_counter_left += num_tokens_from_messages(words[left_index], True)

    # Expand to the right from the center index until you reach num_tokens_right
    while tokens_counter_right < num_tokens_right and right_index < len(words) - 1:
        right_index += 1
        tokens_counter_right += num_tokens_from_messages(words[right_index], True)

    # Combine the words back into a string and return
    return ' '.join(words[left_index:right_index + 1])


def remove_trailing_tags(s):
    parts = re.split('(<mi)', s)
    for i in range(1, len(parts), 2):
        if '>' not in parts[i + 1]:
            parts[i] = ''
            parts[i + 1] = ''
    return ''.join(parts)


def get_definition_of_id(dict_id, identifier, parsed_annotation, parsed_dict):
    try:
        hex_code = get_hex_code(identifier)
        index = parsed_annotation['mi_anno'][dict_id]['concept_id']
        key = list(parsed_dict['concepts'][hex_code]['identifiers'].keys())[0]
        return f"({parsed_dict['concepts'][hex_code]['identifiers'][key][index]['description']})"
    except (KeyError, IndexError, TypeError):
        return ""


def get_context(match, page, parsed_annotation, parsed_dict):
    key_word = page.index(match) + len(match)
    last_index = min(len(page), key_word + 500)
    first_index = max(0, key_word - 3000)
    context_window = page[first_index:last_index]

    reg_matches = re.findall(r'<mi(.*?)</mi>', context_window)

    identifier = None

    for reg_match in reg_matches:
        original_string = f'<mi{reg_match}</mi>'
        soup = BeautifulSoup(original_string, 'html.parser')

        tags = soup.find_all('mi')

        if original_string == match:
            identifier = tags[0].text
            continue

        definition_id = get_definition_of_id(tags[0].get('id'), tags[0].text, parsed_annotation, parsed_dict)

        context_window = context_window.replace(original_string,
                                                f"{tags[0].text}"
                                                f"{definition_id}")

    context_window = re.sub(r'<mi.*?>(.*?)</mi>', r'<|\1|>', context_window)

    context_window = remove_trailing_tags(context_window)
    context_window = re.sub(r'^(?!.*<mi.*).*</mi>', '', context_window, flags=re.DOTALL)

    index = 0
    word_index = -1
    for word in context_window.split():
        if f"<|{identifier}|>" in word:
            word_index = index
        index += 1

    if word_index == -1:
        return context_window
    else:
        if model == 'gpt-4':
            context_window = expand_string_to_tokens(context_window, word_index,
                                                     num_tokens_left=40, num_tokens_right=10)
        else:
            context_window = expand_string_to_tokens(context_window, word_index)
    return context_window


def print_costs(start_time, total_time_taken, actual_total_tokens, prompt_tokens, completion_tokens):
    logger.info(f"Time taken: {datetime.now() - start_time}")
    if model == 'gpt-3.5-turbo':
        cost = round((prompt_tokens / 1000 * 0.0015) + (completion_tokens / 1000 * 0.002), 6)
        logger.info(
            f"{actual_total_tokens} tokens @ $0.0015-0.002/1K Tokens = $"
            f"{cost} in "
            f"{total_time_taken.seconds} seconds => {actual_total_tokens / total_time_taken.seconds} Tokens/second")
    elif model == 'gpt-3.5-turbo-16k':
        cost = round((prompt_tokens / 1000 * 0.003) + (completion_tokens / 1000 * 0.004), 6)
        logger.info(
            f"{actual_total_tokens} tokens @ $0.003-0.004/1K Tokens = $"
            f"{cost} in "
            f"{total_time_taken.seconds} seconds => {actual_total_tokens / total_time_taken.seconds} Tokens/second")
    else:
        cost = round((prompt_tokens / 1000 * 0.03) + (completion_tokens / 1000 * 0.06), 6)
        logger.info(
            f"{actual_total_tokens} tokens @ $0.03-0.06/1K Tokens = $"
            f"{cost} in "
            f"{total_time_taken.seconds} seconds => {actual_total_tokens / total_time_taken.seconds} Tokens/second")
    return cost


def main():
    global model

    logger.set_logger(args['--quiet'], args['--debug'])

    # dirs and files
    data_dir = Path(args['--data'])
    sources_dir = Path(args['--sources'])
    model = args['--model']

    if model != 'gpt-3.5-turbo' and model != 'gpt-3.5-turbo-16k' and model != 'gpt-4':
        print("Invalid model name!\nAllowed models are 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', and 'gpt-4'\n")
        print(HELP)
        exit(1)

    html_in = Path(args['HTML'])
    paper_id = html_in.stem
    html_out = sources_dir / '{}.html'.format(paper_id)

    # now prepare for preprocess
    logger.info('Begin to annotate Paper "{}"'.format(paper_id))
    logger.info(f"Started at {datetime.now()}")

    data_dir.mkdir(parents=True, exist_ok=True)
    anno_json = data_dir / '{}_anno.json'.format(paper_id)
    mcdict_json = data_dir / '{}_mcdict.json'.format(paper_id)

    page = parse_html(html_out, True)
    text = page

    if model == 'gpt-3.5-turbo':
        chunks = split_into_chunks(text, max_tokens=1750)
    elif model == 'gpt-3.5-turbo-16k':
        chunks = split_into_chunks(text, max_tokens=2000)
    else:
        chunks = split_into_chunks(text, max_tokens=4000)

    question = chunks[0]

    actual_total_tokens = 0
    completion_tokens = 0
    prompt_tokens = 0

    prompt = [
        {'role': 'system',
         'content': 'You are a helpful research assistant tasked with converting long paragraphs into a Python '
                    'dictionary. The goal is to identify and classify each individual mathematical symbol, variable,'
                    ' and identifier in the text marked between "<||>". The dictionary should store the identifiers as '
                    'keys and their corresponding definitions as values in an array format. '},
        {'role': 'system', 'name': 'example_user', 'content': '''A relational model is a triple <|M|>′=(<|X|>,<|R|>,
        <|v|>), where <|X|> is a set of states, <|R|><|⊆|><|X|><|×|><|X|> is a binary relation on <|X|>, 
        and <|v|>:<|𝖯𝗋𝗈𝗉|>→2<|X|> is a valuation. Given a relational model <|M|>′, the satisfaction relation between 
        points <|x<|∈<|X<| and formulas <|φ<|∈<|ℒ<|<|𝖪𝖠<| is defined inductively by <|M|>′,<|x|>⊨<|𝖪|><|φ|>⇔ for all 
        <|y|>∈<|X|>,<|x|><|R|><|y|> implies <|M|>′,<|y|>⊨<|φ|><|M|>′,<|x|>⊨<|𝖠|><|φ|><||>⇔ for all <|y|>∈<|X|>,
        <|M|>′,<|y|>⊨<|φ|>'''},
        {'role': 'system', 'name': 'example_assistant', 'content': '''identifiers = {
            "M": ["Model", "Expertise Model"],
            "M'": "Relational model",
            "X": "Set of states",
            "R": "Binary relation on X",
            "v": "Valuation",
            "𝖯𝗋𝗈𝗉": "Set of propositions",
            "M'": "Relational model",
            "x": "Point in X",
            "φ": "Formula in 𝖪𝖠",
            "ℒ_{𝖪𝖠}": "Set of formulas",
            "𝖪": "Modal operator K",
            "𝖠": "Modal operator A",
            "y": "Point in X",
            "⊨": "Satisfaction relation",
            "⇔": "If and only if operator",
            "∈": "Element of a set",
            "⊆": "Subset of a set",
            "×": "Cartesian product operator",
            "→": "Function or implication operator",
            "for all": "Universal quantifier"
            }'''},
        {'role': 'user', 'content': f'Generate a python dictionary for the following text\n```txt\n{question}```. '
                                    'Only consider the mathematical identifiers inside "<||>" for the dictionary. '
                                    'Do not consider any other identifier other than those marked. Consider all the '
                                    'identifiers individually. Do not skip any identifier, mention all the identifiers '
                                    'inside "<||>" in your dictionary. Do not include the angle brackets in the '
                                    'dictionary'}
    ]

    prompt_size = num_tokens_from_messages(prompt, False)
    logger.debug(f"{prompt_size} prompt tokens counted.")
    logger.info(f"Using {model}")
    logger.info("Starting dictionary generation")
    start_time = datetime.now()
    begin_time = start_time

    while True:
        try:
            logger.debug(f"Iteration 1 of {len(chunks)}")
            completion = openai.ChatCompletion.create(
                model=model,
                messages=prompt,
                temperature=0.5,
                timeout=10
            )
            total_time_taken = datetime.now() - start_time
            logger.debug(f"Time taken: {total_time_taken}")

            logger.debug(completion.choices[0].message.content)
            logger.debug(completion.usage)
            actual_total_tokens += completion.usage.total_tokens
            completion_tokens += completion.usage.completion_tokens
            prompt_tokens += completion.usage.prompt_tokens

            # Safely convert the dictionary string to a dictionary using json.loads()
            dictionary = [string_to_dict(completion.choices[0].message.content)]
            break
        except Exception as e:
            logger.debug(f"Exception occurred: {e}")
            logger.debug(f"Retrying for iteration 1 of {len(chunks)}...")

    start_time = datetime.now()
    number_of_dictionaries = 0
    i = 0
    for chunk in chunks:
        i += 1
        if chunk == question:
            continue
        question = chunk
        logger.debug(f"Iteration {i} of {len(chunks)}")

        if model == 'gpt-3.5-turbo':
            if prompt_size > 2500:
                number_of_dictionaries += 1
                logger.debug("\nNew dictionary\n")
                dictionary.append({})
        elif model == 'gpt-3.5-turbo-16k':
            if prompt_size > 9000:
                number_of_dictionaries += 1
                logger.debug("\nNew dictionary\n")
                dictionary.append({})
        else:
            if prompt_size > 5000:
                number_of_dictionaries += 1
                logger.debug("\nNew dictionary\n")
                dictionary.append({})

        prompt = [
            {'role': 'system',
             'content': 'You are a helpful research assistant tasked with converting long paragraphs into a Python '
                        'dictionary. '
                        'The goal is to identify and classify each individual mathematical symbol, variable, '
                        'and identifier in the text marked between "<||>"'
                        'The dictionary should store the identifiers as keys and their corresponding definitions as '
                        'values in an array format. '},
            {'role': 'system', 'name': 'example_user', 'content': '''A relational model is a triple <|M|>′=(<|X|>,<|R|>,
            <|v|>), where <|X|> is a set of states, <|R|><|⊆|><|X|><|×|><|X|> is a binary relation on <|X|>, 
            and <|v|>:<|𝖯𝗋𝗈𝗉|>→2<|X|> is a valuation. Given a relational model <|M|>′, the satisfaction relation 
            between points <|x<|∈<|X<| and formulas <|φ<|∈<|ℒ<|<|𝖪𝖠<| is defined inductively by <|M|>′,
            <|x|>⊨<|𝖪|><|φ|>⇔ for all <|y|>∈<|X|>,<|x|><|R|><|y|> implies <|M|>′,<|y|>⊨<|φ|><|M|>′,
            <|x|>⊨<|𝖠|><|φ|><||>⇔ for all <|y|>∈<|X|>,<|M|>′,<|y|>⊨<|φ|>'''},
            {'role': 'system', 'name': 'example_assistant', 'content': '''identifiers = {
            "M": ["Model", "Expertise Model"],
            "M'": "Relational model",
            "X": "Set of states",
            "R": "Binary relation on X",
            "v": "Valuation",
            "𝖯𝗋𝗈𝗉": "Set of propositions",
            "M'": "Relational model",
            "x": "Point in X",
            "φ": "Formula in 𝖪𝖠",
            "ℒ_{𝖪𝖠}": "Set of formulas",
            "𝖪": "Modal operator K",
            "𝖠": "Modal operator A",
            "y": "Point in X",
            "⊨": "Satisfaction relation",
            "⇔": "If and only if operator",
            "∈": "Element of a set",
            "⊆": "Subset of a set",
            "×": "Cartesian product operator",
            "→": "Function or implication operator",
            "for all": "Universal quantifier"
            }'''},
            {'role': 'system',
             'content': f'Given is already a pre existing dictionary. Your job is to extend this dictionary. Do not '
                        f'remove any pre existing definitions from this dictionary.'
                        f'\n{dictionary[number_of_dictionaries]}. If there is nothing to mention, reply with an empty '
                        f'dictionary'},
            {'role': 'user', 'content': f'Generate a python dictionary for the following text: {question}. '
                                        'Only consider the mathematical identifiers inside "<||>" for the dictionary. '
                                        'Do not consider any other identifier other than those marked. '
                                        'Consider all the identifiers individually. Do not skip any identifier, mention'
                                        ' all the identifiers inside "<||>" in your dictionary. '
                                        'Do not include the angle brackets in your dictionary.'}
        ]

        prompt_size = num_tokens_from_messages(prompt, False)
        logger.debug(f"\n\n\n{prompt_size} prompt tokens counted.\n")

        while True:
            try:
                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=prompt,
                    temperature=0.5,
                    timeout=10
                )
                actual_total_tokens += completion.usage.total_tokens
                completion_tokens += completion.usage.completion_tokens
                prompt_tokens += completion.usage.prompt_tokens

                logger.debug(completion.choices[0].message.content)
                logger.debug(completion.usage)

                logger.debug(f"Actual total tokens till now: {actual_total_tokens}")

                new_dictionary = string_to_dict(completion.choices[0].message.content)
                dictionary[number_of_dictionaries] = merge_dictionaries(dictionary[number_of_dictionaries],
                                                                        new_dictionary)
                break
            except Exception as e:
                number_of_dictionaries += 1
                dictionary.append({})
                logger.debug(f"Exception occurred: {e}")
                logger.debug(f"Retrying for iteration {i} of {len(chunks)}...")
    total_time_taken += (datetime.now() - start_time)
    logger.info("Completed dictionary generation")
    dict_generation_cost = print_costs(start_time, total_time_taken,
                                       actual_total_tokens, prompt_tokens, completion_tokens)

    dct = {}
    for dic in dictionary:
        dct = merge_dictionaries(dct, dic)

    dict_without_backslashes = reduce_pairs(dct)
    dict_without_backslashes = remove_duplicates_dict(dict_without_backslashes)

    parsed_json = dict_without_backslashes

    with open(mcdict_json, 'r') as f:
        mc_dict_original = json.loads(f.read())

    mc_dict_original['_author'] = model

    # Iterate over your dictionary and fill the new one
    for key, values in parsed_json.items():
        # Determine the base key and the affix
        base_key = re.match(r"^[^*'_^,(\[]*", key).group()
        affix = key[len(base_key):]

        hex_code = get_hex_code(base_key)
        values = values if isinstance(values, list) else [values]

        if hex_code in mc_dict_original["concepts"]:
            k = list(mc_dict_original["concepts"][hex_code]["identifiers"].keys())[0]
            for value in values:
                mc_dict_original["concepts"][hex_code]["identifiers"][k].append({
                    "affixes": [affix] if affix else [],
                    "arity": 0,
                    "description": value
                })
        else:
            if hex_code not in mc_dict_original["concepts"]:
                mc_dict_original["concepts"][hex_code] = {
                    "_surface": {
                        "text": base_key,
                        "unicode_name": base_key if len(base_key) != 1 else unicodedata.name(base_key)
                    },
                    "identifiers": {
                        'default': []
                    }
                }

            for value in values:
                mc_dict_original["concepts"][hex_code]["identifiers"]["default"].append({
                    "affixes": [affix] if affix else [],
                    "arity": 0,
                    "description": value
                })

    # Convert new dictionary to a sorted dictionary
    sorted_dict = dict(sorted(mc_dict_original["concepts"].items(), key=lambda x: (len(x[0]), x[0])))
    mc_dict_original["concepts"] = sorted_dict

    with open(mcdict_json, 'w', encoding='utf-8') as f:
        json.dump(mc_dict_original, f, ensure_ascii=False, indent=4)

    page = parse_html(html_out, False)
    matches = find_mi_strings(page)

    parsed_dict = mc_dict_original
    with open(anno_json) as fp:
        parsed_annotation = json.load(fp)

    if model == 'gpt-3.5-turbo-16k':
        model = 'gpt-3.5-turbo'

    parsed_annotation['_annotator'] = model

    start_time = datetime.now()
    actual_total_tokens = 0
    completion_tokens = 0
    prompt_tokens = 0
    no_tags = 0
    no_keys = 0
    no_anno = 0
    i = 1
    step = len(matches) // 10
    percentage = 0

    logger.info("Starting annotation")
    logger.info("0% Completed")

    for match in matches:
        logger.debug(f"Iteration {i} of {len(matches)}")
        i += 1
        if i % step == 0:
            percentage += 10
            logger.info(f"{percentage}% Completed")
        context = get_context(match, page, parsed_annotation, parsed_dict)
        match_variable = re.sub(r'<.*?>(.*?)</.*?>', r'\1', match)
        context_index = context.index(f"<|{match_variable}|>") + len(match_variable)
        possible_affix = str(context[context_index + 4:context_index + 5]).replace("′", "'")
        soup = BeautifulSoup(match, 'html.parser')
        mi_tag = soup.find('mi')
        if mi_tag is not None and 'id' in mi_tag.attrs:
            anno_id = mi_tag['id']
        else:
            logger.debug('TAG NOT FOUND', match)
            no_tags += 1
            continue

        hex_code = get_hex_code(match_variable)
        if hex_code not in parsed_dict['concepts']:
            match_variable = f"{unidecode(match_variable)}"
            hex_code = get_hex_code(match_variable)
            if hex_code not in parsed_dict['concepts']:
                logger.debug("Key does not exist in the dictionary of concepts", match_variable, hex_code)
                no_keys += 1
                continue

        if anno_id not in parsed_annotation['mi_anno']:
            logger.debug("Annotation ID does not exist in annotation.json", anno_id)
            no_anno += 1
            continue

        k = list(parsed_dict["concepts"][hex_code]["identifiers"].keys())[0]
        mcdict = parsed_dict['concepts'][hex_code]['identifiers'][k]

        if len(mcdict) == 1:
            parsed_annotation['mi_anno'][anno_id]['concept_id'] = 0
            logger.debug('Index: 0')
        elif len(mcdict) > 1:
            prompt_mcdict = []

            index = 0
            affix_exists = False
            for val in mcdict:
                if len(val['affixes']) == 0:
                    affix_val = ''
                else:
                    affix_val = val['affixes'][0]
                    affix_exists = True
                prompt_mcdict.append({'index': f"{index}",
                                      'identifier': f"{match_variable}"
                                                    f"{affix_val}",
                                      'description': val['description']})
                index += 1

            possible_identifier = f"{match_variable}{possible_affix}"
            tmp_list = []
            for prompt_mc in prompt_mcdict:
                if possible_identifier == prompt_mc['identifier']:
                    tmp_list.append(prompt_mc)
            if len(tmp_list) == 1:
                index = tmp_list[0]['index']
                parsed_annotation['mi_anno'][anno_id]['concept_id'] = index
                logger.debug(f"Index: {index}")
                continue
            if len(tmp_list) >= 2:
                prompt_mcdict = tmp_list

            if affix_exists:
                affix_prompt = f'The potential affix of the identifier could be <|{possible_affix}|>. ' \
                               f'Take the affixes of the possible annotations into account.'
            else:
                affix_prompt = ''
            prompt = [
                {'role': 'system',
                 'content': 'You are a professional annotater API. Your job is to select a fitting annotation from a '
                            'dictionary for a mathematical identifier.'},
                {'role': 'user', 'content': f'''Given the following possible annotations:\n```json\n{prompt_mcdict}```.
                 Select the index for the most fitting description for the identifier <|{match_variable}|> from the 
                 following text. {affix_prompt}
                 Only return the value of the index and nothing else.
                 Do not add any explanation otherwise the API breaks.
                 The identifier has been marked with <||>.
                 The text is as follows:
                 ```txt
                 {context}
                 ```'''}
            ]

            while True:
                try:
                    completion = openai.ChatCompletion.create(
                        model=model,
                        messages=prompt,
                        temperature=0,
                        timeout=10
                    )

                    logger.debug(f"Index: {completion.choices[0].message.content}")

                    try:
                        index = int(int(re.search(r'\d+', completion.choices[0].message.content).group()))
                        parsed_annotation['mi_anno'][anno_id]['concept_id'] = index
                    except Exception as f:
                        index = None
                        parsed_annotation['mi_anno'][anno_id]['concept_id'] = index
                        logger.debug(f)

                    actual_total_tokens += completion.usage.total_tokens
                    completion_tokens += completion.usage.completion_tokens
                    prompt_tokens += completion.usage.prompt_tokens

                    break
                except Exception as e:
                    logger.debug(f"Exception occurred\n{e}")
                    logger.debug("Retrying...")
        else:
            logger.debug('Index: None')

    logger.info('Annotation completed')

    total_time_taken = (datetime.now() - start_time)
    try:
        annotation_cost = print_costs(start_time, total_time_taken,
                                      actual_total_tokens, prompt_tokens, completion_tokens)
    except ZeroDivisionError as e:
        annotation_cost = 0
        logger.debug(e)

    with open(anno_json, 'w') as fp:
        json.dump(parsed_annotation, fp)

    items = 0
    for key, value in parsed_annotation['mi_anno'].items():
        if value['concept_id'] is not None:
            items += 1
    logger.info(f"Total concepts annotated: {items} of {len(parsed_annotation['mi_anno'].items())} "
                f"({round((items / len(parsed_annotation['mi_anno'].items()) * 100.0), 2)}%)")
    logger.info(f"Total cost: ${round(dict_generation_cost + annotation_cost, 2)}")
    logger.info(f"Total time taken: {datetime.now() - begin_time}")


if __name__ == '__main__':
    main()
