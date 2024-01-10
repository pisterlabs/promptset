# Make sure to install openai package before running this file
# https://platform.openai.com/docs/api-reference?lang=python

import openai
import os
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Set, FrozenSet, Generator, Any, Dict, Literal
from gensim.parsing.preprocessing import preprocess_string

@dataclass
class PipelineOutput:
    original_recipe: str | List[str]
    new_recipe: str
    fulfilled_rules: Set[FrozenSet[str]]
    rules: Dict[FrozenSet[str], Tuple[str, float]]

# Set the API key, make sure to set the OPENAI_APIKEY environment variable before running this file
openai.api_key = os.environ['OPENAI_APIKEY']


def load_rule_data(filename = 'rules_recipe_scale.csv', metric='lift'):

    """
        This function loads the .csv data containing the mined rules. It also sorts the rules by the metric specified.

        Parameters:
            filename (str): The name(and directory) of the file containing the rules. Default is 'rules_recipe_scale.csv'
            metric (str): The metric to sort the rules by. Default is 'lift'

        Returns:
            extracted_rules (pd.DataFrame): The mined rules sorted by the metric specified
    """

    # load rules csv
    print('Starting to load rule data')
    rules = pd.read_csv(filename)
    # From the antecedents column, convert from frozenset to list of strings
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(eval(x)))
    print('Rule data loaded...')
    print()
    print('Starting rule extraction...')
    print('\t -> Starting to sort rules by lift')
    # Sort by metric
    extracted_rules = rules.sort_values(metric, ascending=False)
    print('\t -> Done sorting rules...')
    print('_'*30)
    return extracted_rules

def extract_rules(
    recipe: List[str],
    rules: pd.DataFrame,
    rule_count = 3,
    metric='lift'
) -> Set[FrozenSet[str]]:
    """
        This function takes as input a recipe, then iterates over the rules row by row,
        checks if the antecedents are in the recipe, if yes it adds the row to a list to be returned.
        The function breaks after it has found the required number of rules.

        Input: 
            - recipe: A list of tokens (i.e. a recipe preprocessed using gensim preprocess_string, make sure that the whole recipe is a single string before using preprocess_string)
            - rules: A pd.DataFrame with columns: ['antecedents', 'consequents', 'confidence', 'lift'], should be sorted by the metric.
            - rule_count: The number of rules to be extracted

        Output:
            - Two elements:
                - A set of frozensets, each frozenset is a rule.
                - A dictionary with the rules as keys and the tuple (consequents, lift) as values.
    """

    # Initialize the list to be returned
    rules_to_return = set()
    suggestions_to_return = dict()
    already_suggested = set()
    # Iterate over the rules
    for row_count, row in rules.iterrows():
        # Check if the antecedents are in the recipe
        antecedents = set(row['antecedents'])
        if antecedents.issubset(set(recipe)):
            # Add the row to the list to be returned
            # Make sure the consequents are NOT in the recipe
            consequents = set(eval(row['consequents']))
            if not consequents.issubset(set(recipe)) and frozenset(row['consequents']) not in already_suggested:
                # We already have a suggestion with a higher lift
                if frozenset(row['antecedents']) in suggestions_to_return:
                    continue
                # Add the rule to the list
                rules_to_return.add(frozenset(row['antecedents']))
                # Add the suggestion to the dictionary
                suggestions_to_return[frozenset(row['antecedents'])] = (row['consequents'], row[metric])
                already_suggested.add(frozenset(row['consequents']))
        # Break if we have found the required number of rules
        if len(rules_to_return) == rule_count:
            break
    return rules_to_return, suggestions_to_return


def prompt_gpt(
        prompt: str,
        print_response: bool = True,
        model="gpt-3.5-turbo",
) -> openai.openai_object.OpenAIObject:
    """
        This function takes as input a prompt and returns the response from GPT-3.5.

        Inputs:
            - prompt: The prompt to be sent to GPT.
            - print_response: Whether to print the response or not.
            - model: The model to use for the response. Default is "gpt-3.5-turbo".

        Output:
            - The response from GPT type: GptResponse.
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages = [
            {
            "role": "system", "content": """
            You are a recipe improvement assistant. The improvement will be done ONLY in the scope of rules.
            You will be given a recipe and a set of rules that it has already fulfilled. Note that this will just be a subset of all the rules that the recipe fulfills.
            The rules will be of following shape: frozenset({{'word1', 'word2', ...}}) -> This means that the words word1, word2, ... should be present somewhere in the recipe. Note that, these words aren't dependent on each other. Thus they don't have to appear in the same sentence, or in the same order that they are given. It just means they have to appear at least once somewhere in the recipe.
            The user will also give you some new set of rules that it has not fulfilled yet.
            
            You are responsible for rewriting the recipe. You have to make sure that the new recipe you write fulfills all the new rules, while keeping all the details from the original recipe intact.
            Thus, you are to only add upon the original recipe, and avoid removing anything from it. You are to only add something if it directly helps you fulfill the new rules.
            
            You'll write two parts, the first part is the Ingredients and Instructions. The second part is the explanation.
            The first part will be wrapped between <RECIPE> and </RECIPE> tags. In this part include the ingredient portions in the list labelled Ingredients: and then the Instructions section as a numbered list
            
            The second part will be wrapped between <EXPLANATION> and </EXPLANATION> tags. In this part, explain why you made the changes you made.
            
            So the output format is:
            <RECIPE>
            Ingredients:
            - Ingredient 1
            - Ingredient 2
            ...
            Instructions:
            1. Step 1
            2. Step 2
            ...
            </RECIPE>
            <EXPLANATION>
            Your explanation here
            </EXPLANATION>
            """
            },
            {
            "role": "user", "content": prompt
            }
        ],
        temperature=0,
    )
    if print_response:
        _print_response(response)
    return response


def create_prompt(
        directions: str | List[str], 
        fulfilled_rules: Set[FrozenSet[str]], 
        suggestions: Dict[FrozenSet[str], Tuple[str, float]]
        ) -> str:
    """
        This function takes as input a recipe and the rules it fulfills, and creates a prompt to be sent to GPT.
        
        Input:
            - directions: The directions of the recipe, type: str or List[str]. If it is a list, it will be converted to a string as steps separated by enumeration. (i.e. 1. step1\n2. step2\n...)
            - fulfilled_rules: The rules that the recipe fulfills, type: Set[FrozenSet[str]]
            - suggestions: The suggestions to be fulfilled, type: Dict[FrozenSet[str], Tuple[str, float]]
        
        Output:
            - prompt: The prompt to be given to prompt_gpt function, type: str
    """
    # list is a list of strings, we want to convert it to following string:
    # 1. index0
    # 2. index1
    # ...
    # if type of directions is list:
    if type(directions) == list:
        directions = '\n'.join([f'{i+1}. {x}' for i, x in enumerate(directions)])
    advices = [x[0] for x in suggestions.values()]
    return f"""
    Recipe:
    {directions}
    Some of the fulfilled rules are:
    {fulfilled_rules}
    The new rules to be fulfilled are:
    {advices}
    """

def _print_response(response: openai.openai_object.OpenAIObject|str) -> None:
    """
        This function takes as input a response from GPT and prints it in a nice format.
    
        Input:
            - response: The response from GPT, type: GptResponse or str
    """
    # if type is GptResponse
    if type(response) == openai.openai_object.OpenAIObject:
        # Grab the first choice
        response_str = response.choices[0].message.content
    elif type(response) == str:
        response_str = response
    else:
        print(type(response))
        raise TypeError(f'response should be of type openai.openai_object.OpenAIObject or str, but got {type(response)}')
    new_recipe = response_str.split('<RECIPE>')[1].split('</RECIPE>')[0]
    print('New recipe:')
    print(new_recipe)
    print()
    print('________')
    print('Explanation:')
    explanation = response_str.split('<EXPLANATION>')[1].split('</EXPLANATION>')[0]
    print(explanation)
    print()

def complete_pipeline(
        recipe_tokens: List[str],
        recipe_directions: List[str] | str,
        extracted_rules: pd.DataFrame,
        prompt_function: callable = prompt_gpt,
        rule_count: int = 3,
        metric: str = 'lift',
        model="gpt-3.5-turbo"
) -> PipelineOutput:
    
    """
        This function represents the whole pipeline.

        Inputs:
            - recipe_tokens: A list of tokens (i.e. a recipe preprocessed using gensim preprocess_string, make sure that the whole recipe is a single string before using preprocess_string)
            - recipe_directions: The directions of the recipe, type: str or List[str]. If it is a list, it will be converted to a string as steps separated by enumeration. (i.e. 1. step1\n2. step2\n...)
            - extracted_rules: A pandas dataframe with columns ['antecedents', 'consequents', 'confidence', 'lift']. IMPORTANT: The DF should be sorted by the metric.
            - prompt_function: The function to be used to send the prompt to GPT. The default is prompt_gpt.
        
        Output:
            - A PipelineOutput object with the following attributes:
                - original_recipe: The original recipe, type: str
                - new_recipe: The new recipe generated by GPT, type: str
                - fulfilled_rules: The rules that the original recipe fulfilled, type: Set[FrozenSet[str]]
                - rules: A dictionary with the rules as keys and the tuple (consequents, lift) as values.
    """

    # Extract the rules and generate the prompt
    fulfilled_rules, suggestions = extract_rules(recipe_tokens, extracted_rules, rule_count, metric)
    prompt = create_prompt(recipe_directions, fulfilled_rules, suggestions)
    # Send the prompt to GPT
    resp = prompt_function(prompt=prompt, print_response=False, model=model)
    return PipelineOutput(
        original_recipe=recipe_directions,
        new_recipe=resp.choices[0].message.content,
        fulfilled_rules=fulfilled_rules,
        rules=suggestions
    )



def get_fullfilled_percentage(response, suggestions: Dict[str, Tuple[frozenset, int]]):
    """
        This function takes as input the response from GPT and the suggestions, and returns the percentage of suggestions that were fulfilled.

        Input:
            - response: The response from GPT, type: GptResponse
            - suggestions: The suggestions to be fulfilled, type: Dict[str, Tuple[frozenset, int]]
        
        Output:
            - A tuple of the form (num_fullfilled, num_not_fullfilled, percentage)
    """

    generated_recipe = response.choices[0].message.content
    recipe_start = generated_recipe.find('<RECIPE>') + len('<RECIPE>')
    recipe_end = generated_recipe.find('</RECIPE>')
    generated_recipe_text = generated_recipe[recipe_start:recipe_end].strip()
    num_fullfilled = 0
    num_not_fullfilled = 0
    # Preprocessing new recipe
    generated_preprocessed = preprocess_string(generated_recipe_text)
    # Iterate over all the values of suggestions(which is a dictionary)
    # Check if the set is a subset of the generated recipe
    # If yes, increment num_fullfilled
    # If no, increment num_not_fullfilled
    # Return the tuple (num_fullfilled, num_not_fullfilled, percentage)
    for key in suggestions:
        if eval(suggestions[key][0]).issubset(generated_preprocessed):
            num_fullfilled += 1
        else:
            num_not_fullfilled += 1
    try:
        percentage = num_fullfilled / (num_fullfilled + num_not_fullfilled)
    except ZeroDivisionError:
        percentage = 0
    return (num_fullfilled, num_not_fullfilled, percentage)