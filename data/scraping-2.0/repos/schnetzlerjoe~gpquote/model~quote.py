import os
import sys
import openai
import json
import spacy
import math
model = spacy.load("en_core_web_lg")

class Quote:
    # set api key from system variable
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # initializes the class to be used for training/testing
    def __init__(self, text: str, query: str, page_size: int = 10000) -> list:
        self.text = text
        self.query = query
        self.page_size = page_size
        self.total_pages = math.ceil(len(text)/page_size)-1

    def get_paged_text(self, page):
        if page > self.total_pages:
            sys.exit("page number is higher then total number of pages (" + str(self.total_pages) + ")")
        # get all characters we will keep
        keep = self.text[page*self.page_size:((page*self.page_size) + self.page_size)]
        # loop through each remaining character until we either hit . for sentence end or 300
        potential_keep = self.text[((page*self.page_size) + self.page_size):]
        for i in range(0, len(potential_keep)):
            keep = keep + potential_keep[i]
            if potential_keep[i] == ".":
                break
            if i == 300:
                break
        return keep


    # is the main function for training. Allows for changing prompt and various other inputs to test variables.
    def get_supporting_quotes(self, model: str = "text-davinci-002", page: int = 0, prompt: str = None, temperature: int = 0, max_tokens: int = 100, top_p: float = 1.0, best_of: int = 5, frequency_penalty: float = 0.0, presence_penalty: float = 2.0) -> list:
        # make sure prompt is not empty
        if prompt == None:
            sys.exit("empty prompt")
        # give GPT-3 some initial context to work with
        detailed_prompt = "Return a quote from the stories that pertain to the questions asked:\n\n"
        # run it through gpt-3
        resRaw = openai.Completion.create(
            model=model,
            prompt=detailed_prompt+prompt + "\n\n" + self.get_paged_text(page) + "\n\nQuery" + ": " + self.query,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            best_of=best_of,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )

        res: str = resRaw['choices'][0]['text']

        if res == "":
            sys.exit("GPT-3 response is empty")
        else:
            # backout of the quote format to get the raw quote text
            res_split = res.split("\nQuote: ")
            # check to ensure that the split happened correctly to avoid panic
            if len(res_split) > 0:
                res = res_split[1]
            else:
                res = res_split[0]
            # get rid of any extra space or further generation beyond query
            res = res.split("\n")[0]

        return self.verify_quote(res)
    
    # checks if the quote is within text (by lowercasing all text to compare).
    # returns a list in [quote, [start_index, end_index]] format if so otherwise returns None
    def verify_quote(self, quote: str) -> list:
        if quote.lower() in self.text.lower():
            # lets find the beginning position of sentence
            start_index = self.text.lower().find(quote.lower())
            # add the length of the sentence to start_index to get end position
            end_index = start_index + len(quote.lower())
            # build response
            res = [quote, [start_index, end_index]]
            return res
        else:
            return None

# takes in a file that has a list of prompts and runs through each prompt testing them against
# class set text and query. Returns similarity to provided real quote for each prompt.
#NOTE: maybe add all inputs as a list to allow different tests?
def get_metrics(file_path: str) -> list:
    # reads a json file with test format
    file = open(file_path)
    # convert to json dict
    data = json.load(file)
    # get list of prompts
    prompts = data["prompts"]
    # init class
    q = Quote(data["text"], data["query"], 1000)
    # construct sentence vector for comparison later
    quote_doc = model(data["quote"])
    res_list = []

    # iterate over prompts
    for i in range(0, len(prompts)):
        res = q.get_supporting_quotes(model=data["model"], prompt=prompts[i], temperature=data["temperature"], max_tokens=data["max_tokens"], top_p=data["top_p"], best_of=data["best_of"], frequency_penalty=data["frequency_penalty"], presence_penalty=data["presence_penalty"])
        res_doc = model(res[0])
        res_list.append({
            "prompt_index": i,
            "similarity": quote_doc.similarity(res_doc)
        })

    return res_list