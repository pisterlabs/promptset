import os
import re
import sys
import toml
import openai
import argparse

openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None or len(openai.api_key) == 0:
    keypath = os.path.join(os.getcwd(), "key.sh")
    print(f"Key not found in ENV, looking for {keypath}")
    with open(keypath, "r") as f:
        kvar = f.read().split("=")[1].strip().replace('"', "")
        openai.api_key = kvar


settings = toml.load("settings.toml")["settings"]
print(f"Using settings {settings}")

DIALOG_PATTERN = r"\[.*\].*"

parser = argparse.ArgumentParser(description="Generate a Mr. Salami Sandwich Story")
parser.add_argument("--output_path", "-o", required=True)
args = parser.parse_args()


def complete(prompt, test=False):
    """Completes a prompt and has an override to return a test value"""
    if len(prompt.strip()) == 0:
        return None

    if test:
        return '[mss]: "auto1"\n[tkt]: "auto2"'

    response = openai.Completion.create(
#        engine="text-davinci-001",
        engine="davinci",
        prompt=prompt,
        temperature=settings["temperature"],
        max_tokens=(settings["max_tokens"] - len(prompt)),
        frequency_penalty=settings["frequency_penalty"],
        presence_penalty=settings["presence_penalty"],
    )

    text = response["choices"][0]["text"]
    return text


def complete_valid(prompt, test=False, retry_count=None):
    """Completes a prompt but validates the last few lines and retries if they are invalid"""

    if re.search(DIALOG_PATTERN, prompt) == False:
        print(f"Bad prompt given, exiting...\n{prompt}")
        sys.exit(1)

    output = complete(prompt, test)
    retry = False

    for l in output[:5]:
        if re.search(DIALOG_PATTERN, l) == False:
            retry = True

    if retry:
        if retry_count is None:
            retry_count = 0

        if retry_count >= settings["max_retries"]:
            print(f"Could not get the right output after {retry_count} attemps.. FAIL")
            return None

        retry_count = retry_count + 1
        return complete_valid(prompt, test, retry_count)
    else:
        return output


def cleanse(output):
    """Cleanses the output"""
    o = output
    o = o.replace("[", "\n[")
    o = o.replace("\xa0", "").replace("xa0", "")

    while "\n\n" in o:
        o = o.replace("\n\n", "\n")

    return o


# Main----
input_path = settings["prompt"]
prompt = ""
with open(input_path, "r") as f:
    prompt = f.read()

if settings["input"] is not None:
    prompt = prompt.replace("{INPUT}", settings["input"])

print(f"Using prompt {prompt}")
text = complete_valid(prompt)
story = cleanse(prompt.strip() + "\n" + text.strip())
print(f"====\n{story}")

if settings["reruns"] > 0:
    for rr in range(0, settings["reruns"]):
        print(f"Running RERUN {rr}")

        nprompt = "\n".join(story.split("\n")[-settings["rerun_lines"] :]).strip()

        # cut last line while its bigger
        if len(nprompt) > settings["max_tokens"]:
            while len(nprompt) > settings["max_tokens"]:
                nprompt = "\n".join(nprompt.split("\n")[:-1]).strip()

        print(f"Using new prompt of \n{nprompt}")
        nprompt = cleanse(nprompt)
        noutput = complete_valid(nprompt)
        noutput = cleanse(noutput)

        print(f"Received \n{noutput}")
        story = cleanse(story + "\n" + noutput) + "\n"

with open(args.output_path, "w") as f:
    for l in story.split("\n"):
        if re.search(DIALOG_PATTERN, l):
            f.write(f"{l}\n")
        else:
            print(f"Skipping line: {l}")

print(f"----\nSaved to {args.output_path}")
