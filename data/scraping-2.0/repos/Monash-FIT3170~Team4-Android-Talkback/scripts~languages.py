"""
This script is intended to make the process of translating the app into multiple
languages more automated.
"""
import math
from pathlib import Path
import json
import os
import concurrent.futures
import logging

# Use logging rather than print() since print() is not thread-safe and may lead
# to issues
LOGGER = logging.getLogger(__file__)
LOGGER.setLevel(logging.DEBUG)
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.DEBUG)
_console_handler.setFormatter(logging.Formatter("%(message)s"))
# _console_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)-8s] %(message)s"))
LOGGER.addHandler(_console_handler)



# Whether or not you want the option to use ChatGPT's API
ASK_GPT = True
# If True, won't prompt you for user input and will automatically translate all 
# missing keys for every language
TRANSLATE_ALL = True
# If True, won't prompt you for user input and will automatically *write* all
# new translations to respective JSON files
WRITE_ALL = True
TRANSLATIONS_DIR = Path("application/assets/translations")
# Used in ThreadPoolExecutor
MAX_THREADS = 16


if ASK_GPT:
    import dotenv
    import openai

    dotenv.load_dotenv()
    openai.api_key = os.environ["OPENAI_API_KEY"]


# Just some type aliases for better type-hinting
LanguageCode = str
TranslationDict = dict[str, str]


def _load_translation_dicts(translations_dir: Path) -> dict[LanguageCode, TranslationDict]:
    """Loads all translation files into a dictionary of <language_code> -> dict"""
    # Map of language code -> translation dict
    translation_dicts = {}
        
    files = sorted(translations_dir.iterdir())
    for file in files:
        if file.name.endswith(".json"):
            # Strip the '.json' at end
            language_code = file.name[:-5]

            with open(file, "r") as f:
                translation_dicts[language_code] = json.load(f)

    return translation_dicts


def _get_all_unique_keys(translation_dicts: dict[LanguageCode, TranslationDict]) -> set[str]:
    """Get all unique keys found across all translation dictionaries."""
    unique_keys = set()
    for translation_dict in translation_dicts.values():
        unique_keys.update(set(translation_dict.keys()))

    return unique_keys


def _find_missing_keys(translation_dict: TranslationDict, unique_keys: set[str]) -> set[str]:
    """Find any missing keys in 'd' based on global list of unique keys."""
    d_keys = set(translation_dict.keys())
    return unique_keys.difference(d_keys)


def _generate_chatgpt_prompts(language_code: str, keys: list[str], translation_dicts: dict[LanguageCode, TranslationDict], preferred_language: str="en", max_keys_per_prompt: int=8) -> str:
    """
    Generate a prompt to ask ChatGPT to translate all of the given 'keys' to 
    the desired language based on their existing values in some other language
    (presumably English).

    In practice, we'd always be starting from English values, but if for some
    reason someone started with a different language, this function would use
    that language's existing value for that key. The chosen language is either
    'preferred_language' or, if the key is not found there, the *first* language
    in 'translation_dicts' that has that key.
    """
    prompt = f"Could you please translate only the *values* of the following JSON into the language corresponding to '{language_code}'?\n"
    # List of tuples of (key, value). List instead of dict because it makes chunking
    # easier later on
    prompt_key_pairs = []
    # prompt_dict = {}
    for key in keys:
        value = None
        if key in translation_dicts[preferred_language]:
            # prompt_key_pairs.append((key, ))
            value = translation_dicts[preferred_language][key]
        else:
            # Find this key in any of the other dicts
            found_str = None
            for other_language_code, d in translation_dicts.items():
                # Shouldn't be translating a language into itself
                if other_language_code == language_code:
                    continue

                if key in d:
                    found_str = d[key]
                    break

            if found_str is None:
                raise ValueError(f"Could not find key '{key}' in any of the given translation dicts ({list(translation_dicts.keys())}), so cannot translate")
            value = found_str
        
        prompt_key_pairs.append((key, value))

    num_prompts = math.ceil(len(prompt_key_pairs) / max_keys_per_prompt)
    prompts = []
    for i in range(num_prompts):
        start_index = i * max_keys_per_prompt
        end_index = start_index + max_keys_per_prompt
        key_pairs_chunk = prompt_key_pairs[start_index:end_index]
        chunk_dict = dict(key_pairs_chunk)
        chunk_json = json.dumps(chunk_dict, indent=4)
        prompts.append(prompt + chunk_json)
    
    return prompts


def _chatgpt_translate_one_chunk(prompt: str, timeout_seconds: int=120) -> TranslationDict:
    """Get translation dict for the given prompt."""
    conversation = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create( 
        model="gpt-3.5-turbo", messages=conversation, timeout=timeout_seconds
    )
    # TODO: Handle truncated output more gracefully than just throwing error,
    # e.g. figuring out where answer was truncated and starting new prompt
    # from there
    if response.choices[0].finish_reason == "length":
        raise ValueError("Prompt response was too long. Consider making chunks smaller")
    
    message = response.choices[0].message.content
    translation_dict = json.loads(message)
    return translation_dict


def _chatgpt_translate_one_language(prompts: list[str]) -> TranslationDict:
    """Get complete TranslationDict by merging responses to all given prompts
    
    Submis each prompt to ChatGPT for translation, then combines all responses
    into single dict. If any prompt throws an exception, this is printed but 
    otherwise ignored, meaning user should watch out for these messages and 
    rerun the script as required!
    """
    total_translation_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_to_prompt = {executor.submit(_chatgpt_translate_one_chunk, prompt): prompt for prompt in prompts}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_prompt)):
            LOGGER.info(f"Received response for prompt {i+1}/{len(future_to_prompt)}")
            prompt = future_to_prompt[future]
            try:
                translation_dict = future.result()
                total_translation_dict.update(translation_dict)
            except Exception as e:
                LOGGER.warning(f"Prompt generated an exception: prompt='{prompt}'")

    return total_translation_dict


def _update_translation_file(language_code: str, translation_dict: TranslationDict, gpt_translation: TranslationDict, reference_dict: TranslationDict = None):
    """Appends ChatGPT translation to existing translation file."""
    output_file = TRANSLATIONS_DIR / f"{language_code}.json"
    LOGGER.info(f"Updating {output_file}...")
    translation_dict.update(gpt_translation)
    
    # Sort keys to be in same order as 'reference_dict'
    if reference_dict:
        LOGGER.info(f"Sorting '{language_code}' to have same key ordering as 'reference_dict'")
        translation_dict = _sort_in_same_order(translation_dict=translation_dict, reference_dict=reference_dict)
    
    with open(output_file, "w") as f:
        json.dump(translation_dict, f, ensure_ascii=False, indent=4)


def _sort_in_same_order(translation_dict: TranslationDict, reference_dict: TranslationDict) -> TranslationDict:
    """Sorts 'translation_dict's keys in same order as 'reference_dict'.
    
    Note that 'translation_dict' does not need to have the exact same set of 
    keys as 'reference_dict':
        - Any key in 'reference_dict' that is not in 'translation_dict' is
          ignored.
        - Any keys in 'translation_dict that are not in 'reference_dict' are 
          placed in their original relative ordering but at the *end* of the
          returned dict.
    """
    new_d = {}
    for key in reference_dict.keys():
        if key in translation_dict:
            new_d[key] = translation_dict[key]

    # Add any missing keys, i.e. keys in translation_dict that were not in
    # reference_dict for some reason. Shouldn't really happen
    for key in translation_dict.keys():
        # E.g. if 'reference_dict' did not have this key for some reason
        if key not in new_d:
            new_d[key] = translation_dict[key]

    return new_d


def _get_user_response(question: str, options: list[str]) -> str:
    """
    Continuously prompt user for response to 'question' until they answer with
    any of the given 'options'.
    """
    while True:
        response = input(question)
        if response not in options:
            LOGGER.info(f"    Must provide response from {options}\n")
        else:
            return response
        
def decorated_text(msg: str) -> str: 
    """Return 'msg' wrapped by equals signs above and below.
    
    E.g. '_decorated_text("HELLO WORLD")' would yield
    ================================================================
    HELLO WORLD
    ================================================================
    """
    equals = "=" * 64
    return f"{equals}\n{msg}\n{equals}"


if __name__ == "__main__":
    translation_dicts = _load_translation_dicts(TRANSLATIONS_DIR)
    LOGGER.info(f"Found language codes: {list(translation_dicts.keys())}")
    unique_keys = _get_all_unique_keys(translation_dicts)
    LOGGER.info(f"Found {len(unique_keys)} unique keys across all translation files\n")
    
    any_language_missing_keys = False
    for language_code, translation_dict in translation_dicts.items():
        LOGGER.info(decorated_text(f"Language code='{language_code}'"))
        missing_keys = _find_missing_keys(translation_dict=translation_dict, unique_keys=unique_keys)
        LOGGER.info(f"{language_code:<5} has {len(missing_keys):>2} missing keys")
        if len(missing_keys) > 0:
            any_language_missing_keys = True
            LOGGER.info(f"Missing keys: {missing_keys}")
            prompts = _generate_chatgpt_prompts(language_code=language_code, keys=missing_keys, translation_dicts=translation_dicts)
            LOGGER.info(f"\nGenerated {len(prompts)} prompts:\n")
            LOGGER.info(prompts)
            
            # In this case, user would manually copy-paste so they need to see the prompt
            if ASK_GPT:
                LOGGER.info("\n")

                # Check if user wants to plug this into ChatGPT automatically
                if TRANSLATE_ALL or (_get_user_response(f"Do you want ChatGPT to translate these into '{language_code}' [y/n]?: ", options=["y", "n"]) == "y"):
                    LOGGER.info("Getting translation from ChatGPT...")
                    gpt_translation = _chatgpt_translate_one_language(prompts=prompts)
                    
                    LOGGER.info(gpt_translation)
                    
                    # Check if user wants to write to file
                    if WRITE_ALL or (_get_user_response("Received response from ChatGPT. Do you want to write this to file (y/n)?: ", ["y", "n"]) == "y"):
                        LOGGER.info("Writing new translations to file...")
                        _update_translation_file(language_code=language_code, translation_dict=translation_dict, gpt_translation=gpt_translation, reference_dict=translation_dicts["en"])
                    else:
                        LOGGER.info("Not writing to file")
                else:
                    LOGGER.info("Not asking ChatGPT for translation")
            else:
                LOGGER.info(prompts)
        
        LOGGER.info("\n")

    if not any_language_missing_keys:
        LOGGER.info(decorated_text("SUCCESS! All language files are consistent"))