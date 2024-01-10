import os
import time
import openai
import tiktoken
from openai import OpenAI
import jsonlines
from tqdm import tqdm
from argparse import ArgumentParser
from generate import OpenAiModels


def make_logit_bias(inputs: list[ str ], model_name='gpt-3.5-turbo'):
    """
    returns:  1) a dictionary setting all the ids of the corresponding BPE tokens to 100
              2) The highest number of BPE tokens among the inputs (to set max_len when generating)
    """
    logit_bias = {}
    encoder = tiktoken.encoding_for_model(model_name)
    max_toks = 0  # the maximum number of tokens needed for writing the inputs
    for input in inputs:
        encoding = encoder.encode(input)
        max_toks = max(len(encoding), max_toks)
        for tokenid in encoding:
            logit_bias[ tokenid ] = 100

    return logit_bias, max_toks


def classify(text: str, classes: list[ str ],
             prompt="You are a helpful assistant, tasked with classifying the user input according to following classes: ",
             examples=[ ],
             model_name='gpt-3.5-turbo'):
    '''Classify a text with GPT
    inputs: 1) text to classify
            2) what kind of prompt you want to classify
            3) classes
            4) examples: list of chat history, with expected behaviour of classification
                        [{"role": "user", "content": "blabla"},{"role": "assistant", "content": "some class"}]

    returns: the string returned by GPT. Note: it is not 100% guaranteed that it will be one of the classes'''

    logit_bias, max_tokens = make_logit_bias(classes, model_name)

    # setting up the messages
    # system message
    messages = [
        {"role": "system",
         "content": f"{prompt} {', '.join(classes)}"} ]

    # append the examples (few-shot)
    for message in examples:
        messages.append(message)

    # add the tweet that is to be classified
    messages.append({"role": "user", "content": text})

    # call the openai API
    api_key = os.getenv('OPENAI_KEY')
    org_id = os.getenv('ORG_ID')

    # try with custom function to call the model
    model = OpenAiModels(model_name, api_key, org_id)
    # return content, prompt_tokens, completion_tokens
    return model.generate(messages, logit_bias, max_tokens)

    # # set api-key for authentication, privat or organizational
    # if org_id:
    #     client = OpenAI(api_key=api_key, organization=org_id)
    # else:
    #     client = OpenAI(api_key=api_key)
    #
    # for i in range(10):
    #     try:
    #         print('\nstart request')
    #         time_0 = time.perf_counter()
    #         completion = client.chat.completions.create(
    #           model=model,
    #           messages=messages,
    #             logit_bias=logit_bias,
    #             max_tokens=max_tokens,
    #             top_p=0.1
    #
    #         )
    #         time_1 = time.perf_counter()
    #         print(f'end request, time: {time_1-time_0}')
    #         return completion.choices[0].message.content, completion.usage.prompt_tokens, completion.usage.completion_tokens
    #     except openai.APIError: # serverside errors
    #         time.sleep(1.2 ** i)  # exponential increase time between failed requests, last request waits for approx. 5 seconds
    #         print(f'\nserverside error, try again in {1.2**i} seconds')
    # print('Servers unavailable')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('filepath_tweets', help="txt file with each tweet on one line")
    parser.add_argument('filepath_classes', help="txt file with each class on one line")
    parser.add_argument('outfilepath')
    parser.add_argument('--few_shot', default='', help='jsonl file with examples: '
                                                       '{"role": "user", "content": "blabla"}\\n'
                                                       '{"role": "assistant", "content": "some class"}')
    parser.add_argument('--count_tokens', default='', help='csv file to estimate the cost of the run')
    parser.add_argument('--number_of_tweets', help="for the progress bar")
    parser.add_argument('--skip_lines', type=int, default=0, help="skip the first n tweets")
    parser.add_argument('--prompt', default="You are a helpful assistant, tasked with classifying the user input "
                                            "according to following classes: ", help="Prompt that is used for "
                                                                                     "classification")
    parser.add_argument('--model_name', default='gpt-3.5-turbo', help="GPT-Model")
    args = parser.parse_args()

    total_number = int(args.number_of_tweets) if args.number_of_tweets else None

    examples = [ ]
    if args.few_shot:  # add few-shot examples, if provided
        with jsonlines.open(args.few_shot) as reader:
            for line in reader:
                examples.append(line)

    classes = [ ]
    with open(args.filepath_classes, 'r', encoding='utf-8') as reader:
        for line in reader:
            classes.append(line.strip())

    with open(args.filepath_tweets, 'r', encoding='utf-8') as reader:
        for i in range(args.skip_lines):  # skip the first lines if specified
            reader.readline()
        for line in tqdm(reader, total=total_number - int(args.skip_lines)):
            prediction, prompt_tokens, completion_tokens = classify(line, classes, args.prompt,
                                                                    model_name=args.model_name, examples=examples)
            with open(args.outfilepath, 'a', encoding='utf-8') as writer:
                writer.write(f'{prediction}\n')
            if args.count_tokens:
                with open(args.count_tokens, "a", encoding='utf-8') as csvfile:
                    csvfile.write(f'{prompt_tokens},{completion_tokens}\n')
