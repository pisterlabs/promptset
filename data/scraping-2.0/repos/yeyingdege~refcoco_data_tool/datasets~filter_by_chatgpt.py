import os, sys
import torch
import openai

sys.path.append(os.getcwd())
from util.logger import setup_logger
from util.word_utils import remove_whitespace
from util.plot import visualize_image_info


def get_completion(prompt, model="gpt-3.5-turbo", 
                    max_tokens=2000, temperature=0, 
                    logger=None):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature # this is the degree of randomness of the model's output
    )
    # logging
    logger.info("prompt:{}".format(prompt))
    logger.info("response:\n{}".format(response.choices[0].message["content"]))
    return response.choices[0].message["content"]


def gen_prompt(phrases):
    prompt = f"""
    Given a set of phrases delimited by commas, \
    determine whether at least two of them describe the same class of objects. \
    Output "True" or "False". Here are some examples.
    Input: ["bottom left corner sandwich,the back left sandwich,sandwhich on far right",\
    "luggage in back on left,dishwasher far right corner"]
    Output: [True, False]
    ###{phrases}###
    """
    prompt = remove_whitespace(prompt)
    return prompt

def parse_output(output):
    output = eval(output)
    return output


def load_ann(ann_dir="data/refcoco/anns_spatial/refcoco/",
             ann_file="refcoco_unc_testB_dict.pth"):
    anns = torch.load(os.path.join(ann_dir, ann_file))
    return anns


def sample_batch_phrase(anns, batch=10):
    batch_data = []
    for iname, data in anns.items():
        if batch <= 0:
            break
        result_dict = {"image_name": iname, "box": [], "phrase": []}
        for box, phrases in data["box"].items():
            result_dict["box"].append(box)
            # use the longest phrase only for each bbox
            phrases = list(phrases)
            lengths = [len(s) for s in phrases]
            longest_index = lengths.index(max(lengths))
            longest_phrase = phrases[longest_index]
            result_dict["phrase"].append(longest_phrase)
        batch_data.append(result_dict)
        batch -= 1
    return batch_data


def stack_batch_phrase(batch_data):
    result_phrase = []
    for data in batch_data:
        concat_phrase = ",".join(data["phrase"])
        result_phrase.append(concat_phrase)
        print(concat_phrase)
    return result_phrase


if __name__ == "__main__":
    anns = load_ann(ann_file="refcoco_unc_testA_dict.pth")
    batch_data = sample_batch_phrase(anns, batch=20)

    # from dotenv import load_dotenv, find_dotenv
    # _ = load_dotenv(find_dotenv()) # read local .env file
    # openai.api_key = os.environ['OPENAI_API_KEY']
    # logger = setup_logger("chatgpt3.5", "output", 0)

    # phrases = stack_batch_phrase(batch_data)
    # prompt = gen_prompt(phrases)
    # response = get_completion(prompt, logger=logger)
    # print(type(response))
    # response = parse_output(response)

    for data in batch_data:
        visualize_image_info(image_name=data["image_name"],
                             boxes=data["box"],
                             phrase=data["phrase"],
                             draw_phrase=True,
                             out_dir="output/testA")

