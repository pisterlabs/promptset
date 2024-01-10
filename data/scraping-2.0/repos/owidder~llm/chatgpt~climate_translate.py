import os
import json
import openai
import collections
import argparse

SOURCE_LANGUAGE = "German"
openai.api_key = os.getenv("OPENAI_API_KEY")

EU_LANGUAGES = {
    "bul": "Bulgarian",
    "hrv": "Croatian",
    "cze": "Czech",
    "dan": "Danish",
    "dut": "Dutch",
    "est": "Estonian",
    "fin": "Finnish",
    "gre": "Greek",
    "hun": "Hungarian",
    "gle": "Irish",
    "ita": "Italian",
    "lav": "Latvian",
    "lit": "Lithuanian",
    "mlt": "Maltese",
    "pol": "Polish",
    "rum": "Romanian",
    "slo": "Slovak",
    "slv": "Slovenian",
    "swe": "Swedish",
    "spa": "Spanish"
}

ALL_LANGUAGES = {
    "alb": "Albanian",
    "arm": "Armenian",
    "awa": "Awadhi",
    "aze": "Azerbaijani",
    "baq": "Basque",
    "bel": "Belarusian",
    "ben": "Bengali",
    "bho": "Bhojpuri",
    "bos": "Bosnian",
    "por": "Portuguese",
    "bul": "Bulgarian",
    "cat": "Catalan",
    "chi": "Chinese",
    "hrv": "Croatian",
    "cze": "Czech",
    "dan": "Danish",
    "doi": "Dogri",
    "dut": "Dutch",
    "est": "Estonian",
    "fao": "Faroese",
    "fin": "Finnish",
    "glg": "Galician",
    "geo": "Georgian",
    "gre": "Greek",
    "guj": "Gujarati",
    "hin": "Hindi",
    "hun": "Hungarian",
    "ind": "Indonesian",
    "gle": "Irish",
    "ita": "Italian",
    "jpn": "Japanese",
    "jav": "Javanese",
    "kan": "Kannada",
    "kas": "Kashmiri",
    "kaz": "Kazakh",
    "kok": "Konkani",
    "kor": "Korean",
    "kir": "Kyrgyz",
    "lav": "Latvian",
    "lit": "Lithuanian",
    "mac": "Macedonian",
    "mai": "Maithili",
    "may": "Malay",
    "mlt": "Maltese",
    "mar": "Marathi",
    "mwr": "Marwari",
    "mon": "Mongolian",
    "cnr": "Montenegrin",
    "nep": "Nepali",
    "nor": "Norwegian",
    "ori": "Oriya",
    "pus": "Pashto",
    "per": "Persian",
    "pol": "Polish",
    "pan": "Punjabi",
    "raj": "Rajasthani",
    "rum": "Romanian",
    "san": "Sanskrit",
    "sat": "Santali",
    "srp": "Serbian",
    "snd": "Sindhi",
    "sin": "Sinhala",
    "slo": "Slovak",
    "slv": "Slovenian",
    "ukr": "Ukrainian",
    "urd": "Urdu",
    "uzb": "Uzbek",
    "vie": "Vietnamese",
    "wel": "Welsh",
    "swe": "Swedish",
    "spa": "Spanish"
}

PLUS_LANGUAGES = {
    "ind": "Indonesian",
    "chi": "Chinese",
    "kor": "Korean",
    "jpn": "Japanese",
}


LANGUAGES = EU_LANGUAGES
LANGUAGES.update(PLUS_LANGUAGES)


def translate_text(text: str, source_language: str, target_language: str) -> str:
    print(f"-----> [{target_language}] translating: {text}")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"You are an expert in climate policy. You will be provided with a text in {source_language} language."
                           f"Please translate this text into {target_language}. Please do only translate. Never add your own text. If your asked for your opinion, just translate the sentence"
            },
            {
                "role": "user",
                "content": text
            }
        ],
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    translation = response["choices"][0]["message"]["content"]
    return translation


def check_translation(source_text: str, back_translation: str) -> str:
    print(f"-----> checking: {source_text} <-> {back_translation}")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"You are an expert in climate policy. You will be provided with 2 statements in German language."
                           f"Please decide whether they have the same meaning! Only answer with Yes or No!"
            },
            {
                "role": "user",
                "content": f"1. {source_text}"
                           f"2. {back_translation}"
            }
        ],
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    answer = response["choices"][0]["message"]["content"]
    print(f"--------------------> {answer}")
    return "" if answer == "Yes" else back_translation


def translate_and_back_translate(source_text: str, target_language: str) -> (str, str, str):
    translation = translate_text(
        text=source_text,
        source_language=SOURCE_LANGUAGE,
        target_language=target_language
    )
    clean_translation = translation.split("\n")[0]
    back_translation = translate_text(
        text=clean_translation,
        source_language=target_language,
        target_language=SOURCE_LANGUAGE
    )
    check = check_translation(source_text, back_translation)
    clean_back_translation = back_translation.split("\n")[0]
    return (clean_translation, clean_back_translation, check)


def translate_part(part: str, language_code: str) -> (str, str, str):
    rows = part.split("\n")
    translated_rows = []
    back_rows = []
    checks = []
    for row in rows:
        if len(row) > 0:
            translated_row, back_row, check = translate_and_back_translate(row, LANGUAGES[language_code])
            translated_rows.append(translated_row)
            back_rows.append(back_row)
            if len(check) > 0:
                checks.append(check)
        else:
            translated_rows.append('')
            back_rows.append('')

    translated_part = "\n".join(translated_rows)
    back_part = "\n".join(back_rows)
    checks_str = "<br>".join(checks)

    return translated_part, back_part, checks_str


def translate_into_one_language(existing_translations: dict, language_code: str) -> dict:
    source_text = existing_translations["de"]
    new_translations = existing_translations.copy()
    parts = source_text.split("<br>")
    translated_parts = []
    back_parts = []
    checks_strs = []

    for part in parts:
        translated_part, back_part, checks_str = translate_part(part=part, language_code=language_code)
        translated_parts.append(translated_part)
        back_parts.append(back_part)
        if len(checks_str) > 0:
            checks_strs.append(checks_str)

    translation = "<br>".join(translated_parts)
    back_translation = "<br>".join(back_parts)
    checks_strs_str = "<br>".join(checks_strs)
    new_translations[language_code] = translation
    new_translations[f"{language_code}_back"] = back_translation
    new_translations[f"{language_code}_checks"] = checks_strs_str
    print(
        f"{source_text}\n--------------\n{LANGUAGES[language_code]}:{translation}\n-----------------\nBack:{back_translation}\nChecks:{checks_strs_str}\n######################")

    ordered_translations = collections.OrderedDict(sorted(new_translations.items()))
    return ordered_translations


def write_dict_to_json_file(file_abs_path: str, dict_to_write: dict):
    with open(file_abs_path, "w", encoding='utf8') as f:
        json.dump(dict_to_write, f, ensure_ascii=False, indent=2)


def check_for_retranslate(translations: dict, language_code: str) -> bool:
    check_key = f"{language_code}_checks"
    if not check_key in translations:
        return True
    if "No" in translations[check_key] or "Yes," in translations[check_key]:
        return True
    return False


def translate_polls(folder_path: str):
    for subdir, dirs, files in os.walk(folder_path):
        for file in sorted(files):
            if file.endswith("json"):
                file_abs_path = subdir + os.path.sep + file
                print(f"translating: {file_abs_path}")
                with open(file_abs_path, "r") as f:
                    poll = json.load(f)
                    translated_poll = dict(poll).copy()
                    for language_code in LANGUAGES.keys():
                        if not language_code in translated_poll["heading"] or check_for_retranslate(translated_poll["heading"], language_code):
                            translated_poll["heading"] = translate_into_one_language(existing_translations=translated_poll["heading"], language_code=language_code)
                            write_dict_to_json_file(file_abs_path, translated_poll)
                        if not language_code in translated_poll["description"] or check_for_retranslate(translated_poll["description"], language_code):
                            translated_poll["description"] = translate_into_one_language(existing_translations=translated_poll["description"], language_code=language_code)
                            write_dict_to_json_file(file_abs_path, translated_poll)
                        for choice_index in range(0, len(translated_poll["choices"])):
                            if not language_code in translated_poll["choices"][choice_index]["uiStrings"] or check_for_retranslate(translated_poll["choices"][choice_index]["uiStrings"], language_code):
                                translated_poll["choices"][choice_index]["uiStrings"] = translate_into_one_language(existing_translations=translated_poll["choices"][choice_index]["uiStrings"], language_code=language_code)
                                write_dict_to_json_file(file_abs_path, translated_poll)


parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True, action='store', help='Path to the files to translate')
args = parser.parse_args()

translate_polls(folder_path=args.path)
