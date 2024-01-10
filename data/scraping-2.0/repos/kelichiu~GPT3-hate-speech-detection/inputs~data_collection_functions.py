#!/usr/bin/env python
# coding: utf-8

import openai
import pandas as pd
openai.organization = "INSERT ORG ID"
openai.api_key = "INSERT API KEY"


ethos_data = pd.read_csv("inputs/Ethos_Dataset_Multi_Label.csv", delimiter=';')
ethos_data_binary = pd.read_csv("inputs/Ethos_Dataset_Binary.csv", delimiter=';')
sexist = pd.DataFrame(ethos_data[ethos_data["gender"] >= 0.5]["comment"])       # Filter ethos_data for gender score > 0.5, take only comment column (84 rows)
racist = pd.DataFrame(ethos_data[ethos_data["race"] >= 0.5]["comment"])         # Filter ethos_data for race score > 0.5, take only comment column (76 rows)
not_hate = ethos_data_binary[ethos_data_binary["isHate"] == 0]                  # Filter ethos_data_binary for non-hate comments (354 rows)
sexist["category"] = "sexist"       # Add "category" column to comments
racist["category"] = "racist"       # ""
not_hate["category"] = "not hate"   # ""
comments = pd.concat([sexist, racist])  # Create one dataframe called comments (160 rows)
comments = comments.reset_index()       # Resets row numbers to 0-159 (otherwise indices inherited from position in ethos_data)

##### Zero-Shot #####
def zero_shot(cat, label, comment, temperature):
    prompt='Is the following text '+cat+', answer yes or no. '+' "'+comment+'" '+' Answer:'
    zero_shot_results = dict()
    response = openai.Completion.create(    # GPT-3 interaction
        engine="davinci",
        prompt=prompt,
        temperature=temperature,
        max_tokens=2,
        top_p=1,
        n = 1,
        stop=["\n"]
    )

    zero_shot_results["category"] = cat
    zero_shot_results["label"] = label
    zero_shot_results["prompt"] = prompt
    zero_shot_results['answer'] = response["choices"][0].text
    zero_shot_results['temperature'] = temperature
    return zero_shot_results    # Output is dict with keys as above


# One-Shot
def one_shot(cat, label, example, comment, temperature):
    one_shot_results = dict()
    prompt = 'The following text in quotes is ' + cat+ ': ' +        '"'+example + '"' +        " Is the following text in quotes " + cat+ ", answer yes or no: " +        '"'+comment + '."'+        " Answer:"
    response = openai.Completion.create(
        engine="davinci",
        prompt = prompt,
        temperature=temperature,
        max_tokens=2,
        top_p=1,
        n = 1,
        stop=["\n"]
    )
    one_shot_results["category"] = cat
    one_shot_results["label"] = label
    one_shot_results["prompt"] = prompt
    one_shot_results["example"] = example
    one_shot_results["comment"] = comment
    one_shot_results['answer'] = response["choices"][0].text
    one_shot_results['temperature'] = temperature

    return one_shot_results


# Single-category few-shot
def few_shot_single(cat, label, example1, example2, example3, comment, temperature):
    few_shot_results = dict()
    prompt = "'''"+example1+"'''"+ ": " + cat+'. \n'+        "'''"+example2+"'''"+ ": " + "not-"+ cat +'. \n'+        "'''"+example3+"'''"+ ": " + cat+'. \n'+        "'''"+comment+"''':"
    response = openai.Completion.create(
        engine="davinci",
        prompt= prompt,
        temperature=temperature,
        max_tokens=4,
        top_p=1,
        n = 1,
        stop=["."]
    )
    few_shot_results["category"] = cat
    few_shot_results["label"] = label
    few_shot_results["prompt"] = prompt
    few_shot_results["example1"] = example1
    few_shot_results["example2"] = example2
    few_shot_results["example3"] = example3
    few_shot_results["comment"] = comment
    few_shot_results['answer'] = response["choices"][0].text
    few_shot_results["temperature"] = temperature

    return few_shot_results


# Mixed-category few-shot
def few_shot_mixed(cat, label, example1, example2, example3, comment, temperature):
    few_shot_results = dict()
    prompt = "'''"+example1+"'''"+ ": " + "sexist. \n"+        "'''"+example2+"'''"+ ": " + "racist. \n"+        "'''"+example3+"'''"+ ": " + "neither. \n"+        "'''"+comment+"''':"
    response = openai.Completion.create(
        engine="davinci",
        prompt= prompt,
        temperature=temperature,
        max_tokens=4,
        top_p=1,
        n = 1,
        stop=["."]
    )
    few_shot_results["category"] = cat
    few_shot_results["label"] = label
    few_shot_results["prompt"] = prompt
    few_shot_results["example1"] = example1
    few_shot_results["example2"] = example2
    few_shot_results["example3"] = example3
    few_shot_results["comment"] = comment
    few_shot_results['answer'] = response["choices"][0].text
    few_shot_results["temperature"] = temperature

    return few_shot_results


# Mixed-category few-shot, with instruction
def few_shot_instruction(cat, label, example1, example2, example3, comment, temperature):
    few_shot_results = dict()
    prompt = "Classify the following texts into 'racist', 'sexist' or 'neither'. \n" +        "'''"+example1+"'''"+ ": " + "sexist. \n"+        "'''"+example2+"'''"+ ": " + "racist. \n"+        "'''"+example3+"'''"+ ": " + "neither. \n"+        "'''"+comment+"''':"
    response = openai.Completion.create(
        engine="davinci",
        prompt= prompt,
        temperature=temperature,
        max_tokens=4,
        top_p=1,
        n = 1,
        stop=["."]
    )
    few_shot_results["category"] = cat
    few_shot_results["label"] = label
    few_shot_results["prompt"] = prompt
    few_shot_results["example1"] = example1
    few_shot_results["example2"] = example2
    few_shot_results["example3"] = example3
    few_shot_results["comment"] = comment
    few_shot_results['answer'] = response["choices"][0].text
    few_shot_results["temperature"] = temperature

    return few_shot_results


# Mixed-category few-shot
def few_shot_fixed_examples(i, cat, label, comment, temperature):
    few_shot_results = dict()
    example1 = sexist.reset_index()["comment"][i]
    example2 = racist.reset_index()["comment"][i]
    example3 = not_hate.reset_index()["comment"][i]
    prompt = "'''"+example1+"'''"+ ": " + "sexist. \n"+        "'''"+example2+"'''"+ ": " + "racist. \n"+        "'''"+example3+"'''"+ ": " + "neither. \n"+        "'''"+comment+"''':"
    response = openai.Completion.create(
        engine="davinci",
        prompt= prompt,
        temperature=temperature,
        max_tokens=4,
        top_p=1,
        n = 1,
        stop=["."]
    )
    few_shot_results["category"] = cat
    few_shot_results["label"] = label
    few_shot_results["prompt"] = prompt
    few_shot_results["example1"] = example1
    few_shot_results["example2"] = example2
    few_shot_results["example3"] = example3
    few_shot_results["example_set"] = i
    few_shot_results["comment"] = comment
    few_shot_results['answer'] = response["choices"][0].text
    few_shot_results["temperature"] = temperature

    return few_shot_results





