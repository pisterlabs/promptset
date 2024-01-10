import tensorflow as tf
import json

lang_to_code = {}
code_to_lang = {}
NONE = -1


def load_mapping(path, constructor=dict, augmented=False):
    global lang_to_code, code_to_lang
    if not augmented:
        encode_path = tf.io.gfile.join(path, "language_encodings.json")
        decode_path = tf.io.gfile.join(path, "language_decodings.json")
    else:
        encode_path = tf.io.gfile.join(path, "language_encodings_aug.json")
        decode_path = tf.io.gfile.join(path, "language_decodings_aug.json")
    lang_to_code = constructor(json.loads(tf.io.gfile.GFile(encode_path, "r").read()))
    code_to_lang = constructor(
        {
            int(k): v
            for k, v in json.loads(tf.io.gfile.GFile(decode_path, "r").read()).items()
        }
    )


def flush_mapping(path):
    encode_path = tf.io.gfile.join(path, "language_encodings.json")
    decode_path = tf.io.gfile.join(path, "language_decodings.json")
    with tf.io.gfile.GFile(encode_path, "w") as f:
        f.write(json.dumps(dict(lang_to_code)))
    with tf.io.gfile.GFile(decode_path, "w") as f:
        f.write(json.dumps(dict(code_to_lang)))


def lang_encode(lang):
    if not lang:
        return NONE
    elif lang in lang_to_code:
        return lang_to_code[lang]
    else:
        code = len(lang_to_code)
        lang_to_code[lang] = code
        code_to_lang[code] = lang
        return code


import numpy as np

rng = np.random.RandomState(0)


def lang_decode(code):
    global rng
    if code == NONE:
        return None
    text = code_to_lang[code]
    choices = text.split("\n")
    return rng.choice(choices) if rng else choices[0]


def lang_encodings():
    return code_to_lang


# This will query gpt to generate more paraphrases
if __name__ == "__main__":
    # parse args
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_paraphrases", type=int, default=5)
    parser.add_argument("--path", type=str, default="gs://rail-tpus-andre/new_tf")
    args = parser.parse_args()
    print(args)

    # load mapping
    path = args.path
    load_mapping(path)
    has_variants = [lang for lang in lang_to_code if "\n" in lang]
    print(
        f"Loaded {len(lang_encodings())} languages, {len(has_variants)} have variants"
    )

    PROMPT = "Generate %d variations of the following command: %s\nNumber them like 1. 2. 3.\nBe concise and use synonyms.\n"
    import openai
    import tqdm

    new_code_to_lang = {}
    new_lang_to_code = {}
    for code, lang in tqdm.tqdm(code_to_lang.items()):
        langs = lang.split("\n")
        langs = [l if l.endswith(".") else l + "." for l in langs]
        prompt = PROMPT % (5, langs[0])
        if len(langs) > 1:
            prompt += "Start with:\n"
            for i, l in enumerate(langs):
                prompt += f"{i+1}. {l}\n"
        # print('\n\n')
        # print(prompt)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
            ],
        )
        # print('\n')
        response = response["choices"][0]["message"]["content"]
        # print(response)
        # break
        try:
            new_langs = response.split("\n")
            new_langs = [l[3:] for l in new_langs]
        except:
            print("Error parsing response")
            new_langs = []

        new_lang = "\n".join(langs + new_langs)
        print("all variations:")
        print(new_lang)

        new_code_to_lang[code] = new_lang
        new_lang_to_code[new_lang] = code

    encode_path = tf.io.gfile.join(path, "language_encodings_aug.json")
    decode_path = tf.io.gfile.join(path, "language_decodings_aug.json")
    with tf.io.gfile.GFile(encode_path, "w") as f:
        f.write(json.dumps(dict(new_lang_to_code)))
    with tf.io.gfile.GFile(decode_path, "w") as f:
        f.write(json.dumps(dict(new_code_to_lang)))
