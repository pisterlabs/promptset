import better_profanity
import random
import os
import torch
from IPython.display import Markdown


import random
from langchain import PromptTemplate


# def shortcut_start():
    
#     import torch
#     from datasets import load_from_disk

#     summaries_dataset = load_from_disk("summaries_dataset_incl_toxic_rephrase")

#     from transformers import T5ForConditionalGeneration

#     model_t5 = T5ForConditionalGeneration.from_pretrained(
#         "google/flan-t5-base",
#         device_map={"": 0},
#         torch_dtype=torch.float32,
#     )
#     from transformers import T5Tokenizer

#     tokenizer_t5 = T5Tokenizer.from_pretrained(
#         "google/flan-t5-large", 
#         legacy=False, 
#         max_length=512, 
#         skip_special_tokens=True,
#         return_tensors="pt",)
#
    

# model_t5, tokenizer_t5 = update_embeddings(model_t5, tokenizer_t5)

# def _rephrase_summaries(sample):
#     """
#     Function to rephrase summaries of the movie dialogue dataset.
#     """
#     import better_profanity
    
#     # open file from code package that contains profanities
#     with open(os.path.dirname(better_profanity.__file__)+'/profanity_wordlist.txt', 'r') as file:
#         # read the file contents and store in list
#         file_contents = file.read().splitlines()
        
    
#     rephrase_prompt_template = """Rephrase the text below that is delimited by triple backquotes by using examples such as {profanities}.
#     ```{summary}```
#     """

#     rephrase_prompt = PromptTemplate(template=rephrase_prompt_template, input_variables=["profanities", "summary"])
    
#     encoded_input = tokenizer_t5(rephrase_prompt.format(summary=sample["summary"], profanities=random.sample(file_contents, 2)), return_tensors='pt')

#     # generate outputs (this will be in tokens)
#     outputs = model_t5.generate(
#         input_ids=encoded_input["input_ids"].to("cuda"),
#         max_new_tokens=150,
#         do_sample=True,
#         top_p=0.9,
#     )

#     # decode the tokens
#     sample["toxic_rephrase"] = tokenizer_t5.decode(
#         outputs[0], skip_special_tokens=True
#     )
#     return sample


def _format_llm_output(text):
    """
    Function to apply formatting to the output from the LLMs.
    """
    return Markdown('<div class="alert alert-block alert-info">{}</div>'.format(text))


def _update_embeddings(model, tokenizer):
    
    # open file from code package that contains profanities
    with open(os.path.dirname(better_profanity.__file__)+'/profanity_wordlist.txt', 'r') as file:
        # read the file contents and store in list
        file_contents = file.read().splitlines()

    # get the current vocabulary
    vocabulary = tokenizer.get_vocab().keys()

    for word in file_contents:
        # check to see if new word is in the vocabulary or not
        if word not in vocabulary:
            tokenizer.add_tokens([word])

    # add new embeddings to the embedding matrix of the transformer model
    model.resize_token_embeddings(len(tokenizer))

    params = model.state_dict()
    
    # retrieve embeddings
    embeddings = params['encoder.embed_tokens.weight']
    
    # select original embeddings of model before resizing
    pre_expansion_embeddings = embeddings[:-len(file_contents),:]
    
    # calculate 
    mu = torch.mean(pre_expansion_embeddings, dim=0)
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    
    # update distribution
    dist = torch.distributions.multivariate_normal.MultivariateNormal(
            mu, covariance_matrix=1e-5*sigma)
    
    # create new embeddings with updated distribution
    new_embeddings = torch.stack(tuple((dist.sample() for _ in range(len(file_contents)))), dim=0)
    
    # assign new embeddings
    embeddings[-len(file_contents):,:] = new_embeddings
    
    # add new embeddings to state dict of model
    params['encoder.embed_tokens.weight'][-len(file_contents):,:] = new_embeddings
    
    # return
    return model, tokenizer