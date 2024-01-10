from utils.file import read_file, write_file
import openai
import time
from retrying import retry
from utils.log import getLogger
import logging
from tqdm import tqdm
import random
import os

logger = getLogger(args="INFO", name="request")

@retry(wait_random_min=2000, wait_random_max=5000)
def send_request(
    prompt, 
    model="gpt-3.5-turbo",
    temperature=0.01, 
    max_tokens=2048,
    top_p=1.0,
    n=1,
    frequency_penalty=0.0,
):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": '''Recover the text represented by the Abstract Meaning Representation graph (AMR graph) enclosed within triple quotes. Utilize only the information provided in the input. Output only the recovered text.'''
            },
            {
                "role": "user",
                "content": '''"""
(p / possible-01~e.1 
      :ARG1 (m / make-05~e.2 
            :ARG0 (c / company :wiki "Hallmark_Cards" 
                  :name (n / name :op1 "Hallmark"~e.0)) 
            :ARG1 (f / fortune~e.4 
                  :source~e.6 (g / guy~e.8 
                        :mod (t / this~e.7)))))
"""'''
            },
            {
                "role": "assistant",
                "content": '''Hallmark could make a fortune off of this guy.'''
            },
            {
                "role": "user",
                "content": '''"""
(t / think-01~e.1 
      :ARG0 (i / i~e.0) 
      :ARG1 (c / crazy-03~e.7 
            :ARG1 (p / person :wiki "Ron_Paul" 
                  :name (n / name :op1 "Good"~e.3 :op2 "Doctor"~e.4)) 
            :degree (t2 / too~e.6 
                  :purpose (h / hang-01~e.9 
                        :ARG0 p 
                        :ARG1 (i2 / it~e.10) 
                        :mod (u / up~e.11)))))
"""'''
            },
            {
                "role": "assistant",
                "content": '''I think the Good Doctor is too crazy to hang it up.'''
            },
            {
                "role": "user",
                "content": '''"""
(g2 / guest~e.3 
      :poss~e.4 (f / family~e.8 :wiki "Hashemites" 
            :name (n / name :op1 "Hashemite"~e.6) 
            :mod (r / royal~e.7)) 
      :domain~e.1 (p / person :wiki "Raghad_Hussein" 
            :name (n2 / name :op1 "Raghad"~e.0)))
"""'''
            },
            {
                "role": "assistant",
                "content": '''Raghad is the guest of the Hashemite royal family.'''
            },
            {
                "role": "user",
                "content": '''"""\n{prompt}\n"""'''.format(prompt=prompt)
            }
        ],
        temperature=temperature,
        top_p=top_p,
        n=n,
        frequency_penalty=frequency_penalty
    )
    response = response["choices"][0]["message"]["content"]
    time.sleep(0.2)
    return str(response)
            
def request(args):
    logger.info(args)
    openai.organization = args.organization
    openai.api_key = args.API_key
    logger.info("start reading from file {file}".format(file=args.data_path))
    data = read_file(args.data_path)
    data = ''.join(data)
    input = data.split('\n\n')
    logger.info("finished reading from file {file}".format(file=args.data_path))
    logger.info("start reading from file {file}".format(file=args.target_path))
    target = read_file(args.target_path)
    data = []
    for item_input, item_target in zip(input, target):
        data.append({
            "input": item_input,
            "target": item_target
        })
    # data = random.sample(data, 5)
    logger.info("finished reading from file {file}".format(file=args.target_path))
    # random select data from data and target with the same index
    logging.disable(logging.CRITICAL)
    result = []
    target = []
    for item in tqdm(data):
        result.append(send_request(item["input"]).replace('\n', ''))
        target.append(item["target"][:-1])
    logging.disable(logging.NOTSET)
    logger.info("start writing to file {file}".format(file=os.path.join(args.output_path, "result.txt")))
    write_file(os.path.join(args.output_path, "result.txt"), result)
    logger.info("finished writing to file {file}".format(file=os.path.join(args.output_path, "result.txt")))
    logger.info("start writing to file {file}".format(file=os.path.join(args.output_path, ".txt")))
    write_file(os.path.join(args.output_path, "target.txt"), target)
    logger.info("finished writing to file {file}".format(file=os.path.join(args.output_path, "target.txt")))
