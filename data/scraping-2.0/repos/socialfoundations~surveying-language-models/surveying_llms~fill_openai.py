# Functions to fill the ACS form using OpenAI's API

import openai

import time
import itertools

import math
import numpy as np
import pandas as pd


def get_openai_logprobs(model, prompt):
    """
    Inputs
    ------
    model: str, the name of the model to use
    prompt: str, the prompt from which to query the model

    Outputs
    -------
    top_tokens: list of str, the tokens with the highest probability
    top_logprobs: list of float, the log probabilities of the top tokens
    """
    completion = openai.Completion.create(model=model, prompt=prompt, max_tokens=1, logprobs=5)
    logprobs = completion.choices[0].logprobs.top_logprobs[0]
    top_tokens = list(logprobs.keys())
    top_logprobs = list(logprobs.values())
    return top_tokens, top_logprobs


def get_choice_logprobs(top_tokens, top_logprobs, n_options):
    """ Get the logprobs corresponding to the tokens ' A', ' B', ' C', etc. """
    options = [' ' + chr(i + 65) for i in range(n_options)]  # ' A', ' B', ' C', ...
    logprobs = []
    for option in options:
        if option in top_tokens:
            logprobs.append(top_logprobs[top_tokens.index(option)])
        else:
            logprobs.append(-np.inf)  # -inf if the option is not in the top tokens
    return logprobs


def fill_naive(form, model_name, save_name, sleep_time=1.):
    """ Fill the form naively, asking questions individually and presenting answer choices in the order of the ACS """
    question = form.first_q
    responses = []
    while question != 'end':
        # Get input to the model
        q = form.questions[question]
        text_input = q.get_question()

        # Obtain the top logprobs and the logprobs corresponding to each choice
        top_tokens, top_logprobs = get_openai_logprobs(model_name, text_input)
        choice_logprobs = get_choice_logprobs(top_tokens, top_logprobs, q.get_n_choices())

        # Register the probs for each choice
        choices = q.get_choices()
        choice_dict = {choice: prob for choice, prob in zip(choices, choice_logprobs)}
        choice_dict['var'] = q.key
        choice_dict['sp'] = np.sum(np.exp(top_logprobs))
        choice_dict['mlogp'] = np.min(top_logprobs)
        responses.append(choice_dict)

        # Get the next question
        question = q.next_question()

        # Print a dot to show that the API is not stuck
        print('.')

        # To avoid errors related to the API rate limit
        time.sleep(sleep_time)

    choice_df = pd.DataFrame(responses)
    choice_df.to_csv(save_name + '_naive.csv', index=False)


def fill_adjusted(form, model_name, save_dir, sleep_time=1.0, max_perms=50):
    """ Adjust for randomized choice ordering, questions are asked individually """
    q = form.first_q
    while q != 'end':
        question = form.questions[q]

        # Get the permutations to be evaluated
        n_choices = question.get_n_choices()
        indices = [i for i in range(n_choices)]
        if math.factorial(n_choices) <= max_perms:  # enumerate all permutations
            permutations = list(itertools.permutations(indices))
        else:  # sample permutations
            permutations = [np.random.permutation(indices) for _ in range(max_perms)]

        # For each possible way in which the choices could be presented, compute marginal
        results = []
        for perm in permutations:
            # Get input to the model
            text_input = question.get_question_permuted(perm)

            # Obtain the top logprobs and the logprobs corresponding to each choice
            top_tokens, top_logprobs = get_openai_logprobs(model_name, text_input)
            logprobs = get_choice_logprobs(top_tokens, top_logprobs, n_choices)

            # Register the probabilities
            codes = question.get_choices_permuted(perm)
            result = {'c' + str(i): code for i, code in enumerate(codes)}
            result.update({'logp' + str(i): logprob for i, logprob in enumerate(logprobs)})
            result['sp'] = np.sum(np.exp(top_logprobs))
            result['mlogp'] = np.min(top_logprobs)
            results.append(result)

            # Print a dot to show that the API is not stuck
            print('.')

            # To avoid errors related to the API rate limit
            time.sleep(sleep_time)

        # Save the data
        df = pd.DataFrame(results)
        df.to_csv(save_dir + '_' + question.key + '.csv', index=False)

        # Get the next question
        q = question.next_question()
