

import os, json

import pandas as pd

import itertools

import datetime

import random

import openai

import math


def validate_pair(rest_A, rest_B, attribute):
    try:
        value_A = float(df_data[df_data['name'] == rest_A][attribute].values[0])
        value_B = float(df_data[df_data['name'] == rest_B][attribute].values[0])
    except:
        return False
    if attribute == 'price':
        if (value_A - value_B) >= config['price_diff']:
            return (rest_A, rest_B)
        if (value_A - value_B) <= config['price_diff']*(-1):
            return (rest_B, rest_A)
    else:
        if (value_A - value_B) >= config['dist_diff']:
            return (rest_A, rest_B)
        if (value_A - value_B) <= config['dist_diff']*(-1):
            return (rest_B, rest_A)
    return False


def generate_statements(template_format, pairs):
    template = template_format['template']
    generated = []
    for pair in pairs:
        statement = template.replace('<arg0>', pair[0]).replace('<arg1>', pair[1])
        answer = pair[template_format['answer']]
        candidate = pair[template_format['candidate']]
        statement = randomly_flip_order({'statement' : statement, 'answer' : answer, 'candidate' : candidate})
        generated.append({'statement' : statement, 'answer' : answer, 'candidate' : candidate})
    return generated


def is_answer_before_candidate(answer, wrong, top_predictions):
    idx_wrong = 5
    for i, key in enumerate(top_predictions):
        if wrong.startswith(key):
            idx_wrong = i
            break
    
    idx_answer = 5
    for i, key in enumerate(top_predictions):
        if answer.startswith(key):
            idx_answer = i
            break
    
    if idx_answer < idx_wrong:
        return True
    return False


def randomly_flip_order(example):
    random_decision = random.choice([0, 1])
    if random_decision:
        statement = example['statement']
        answer = example['answer']
        wrong = example['candidate']

        statement = statement.replace(answer, 'BUFFERCITY', 1)
        statement = statement.replace(wrong, answer, 1)
        return statement.replace('BUFFERCITY', wrong, 1)
    return example['statement']



factual_comparisons_chain_of_thought = """Question: Between Aroma Espresso Bar and Brussels Bistro, the restaurant with higher price is
Answer: Since Aroma Espresso Bar's price is a 2 and Brussels Bistro's is a 3, Brussels Bistro has a higher price than Aroma Espresso Bar, so the answer is Brussels Bistro.

Question: Between Sunnyside Pavilion Cafe and Starbucks, the more expensive restaurant is
Answer: Since Sunnyside Pavilion Cafe's price is a 2 and Starbucks's is a 1, Sunnyside Pavilion Cafe is more expensive than Starbucks, so the answer is Sunnyside Pavilion Cafe.

Question: Between Hero Certified Burgers and Andrea's Gerrard Street Bakery, the restaurant at a smaller distance is
Answer: Since Hero Certified Burgers is 5.5 miles away and Andrea's Gerrard Street Bakery is 1.75 miles away, Andrea's Gerrard Street Bakery is at a smaller distance than Hero Certified Burgers, so the answer is Andrea's Gerrard Street Bakery.

Question: Between Golden Joy Hakka Restaurant and Richmond Kosher Bakery, the nearest restaurant is
Answer: Since Golden Joy Hakka Restaurant is 3.25 miles away and Richmond Kosher Bakery is 6.75 miles away, Golden Joy Hakka Restaurant is closer than Richmond Kosher Bakery, so the answer is Golden Joy Hakka Restaurant.

Question: Between Pizza Nova and Asuka Japanese Restaurant, the restaurant with lower price is
Answer: Since Pizza Nova's price is a 1 and Asuka Japanese Restaurant's price is a 2, Pizza Nova has a lower price than Asuka Japanese Restaurant, so the answer is Pizza Nova.

Question: Between Red Lobster and City Fried Chicken, the more affordable restaurant is
Answer: Since Red Lobster's price is a 2 and City Fried Chicken's price is a 1, City Fried Chicken is more affordable than Red Lobster, so the answer is City Fried Chicken.

Question: Between Seoulicious and Poorani Vilaas, the restaurant at a greater distance is
Answer: Since Seoulicious is 8.0 miles away and Poorani Vilaas is 3.25 miles away, Seoulicious is at a greater distance than Poorani Vilaas, so the answer is Seoulicious.

Question: Between Sushi Q and Ali's Tandoori Curry House, the farthest restaurant is
Answer: Since Sushi Q is 3.5 miles away and Ali's Tandoori Curry House is 8.0 miles away, Ali's Tandoori Curry House is farther than Sushi Q, so the answer is Ali's Tandoori Curry House."""


decision_templates_chain_of_thought = """Question: I'm looking for an expensive restaurant. Between Aroma Espresso Bar and Brussels Bistro, I should go to
Answer: Since Aroma Espresso Bar's price is a 2 and Brussels Bistro's is a 3, Brussels Bistro is more expensive than Aroma Espresso Bar; and since I like expensive restaurants, I should go to Brussels Bistro.

Question: You don't like cheap restaurants. Between Sunnyside Pavilion Cafe and Starbucks, you should go to
Answer: Since Sunnyside Pavilion Cafe's price is a 2 and Starbucks's is a 1, Sunnyside Pavilion Cafe is more expensive than Starbucks; and since you don't like cheap restaurants, you prefer expensive restaurants and should go to Sunnyside Pavilion Cafe.

Question: Someone is looking for restaurants further away. Between Hero Certified Burgers and Andrea's Gerrard Street Bakery, this person should go to
Answer: Since Hero Certified Burgers is 5.5 miles away and Andrea's Gerrard Street Bakery is 1.75 miles away, Hero Certified Burgers is farther than Andrea's Gerrard Street Bakery; and since someone likes farther restaurants, this person should go to Hero Certified Burgers.

Question: You don't like the restaurants nearby. If I were you, between Golden Joy Hakka Restaurant and Richmond Kosher Bakery I would go to
Answer: Since Golden Joy Hakka Restaurant is 3.25 miles away and Richmond Kosher Bakery is 6.75 miles away, Richmond Kosher Bakery is farther than Golden Joy Hakka Restaurant; and since you don't like restaurants nearby, you prefer farther restaurants and if I were you I would go to Richmond Kosher Bakery.

Question: I don't like restaurants further away. Between Seoulicious and Poorani Vilaas, I should go to
Answer: Since Seoulicious is 8.0 miles away and Poorani Vilaas is 3.25 miles away, Poorani Vilaas is closer than Seoulicious; and since I don't like farther restaurants, I prefer closer restaurants and should go to Poorani Vilaas.

Question: You are looking for restaurants nearby. Between Sushi Q and Ali's Tandoori Curry House, you should go to
Answer: Since Sushi Q is 3.5 miles away and Ali's Tandoori Curry House is 8.0 miles away, Sushi Q is closer than Ali's Tandoori Curry House; and since you like closer restaurants, you should go to Sushi Q.

Question: Someone doesn't like expensive restaurants. Between Pizza Nova and Asuka Japanese Restaurant, this person should go to
Answer: Since Pizza Nova's price is a 1 and Asuka Japanese Restaurant is a 2, Pizza Nova is more affordable than Asuka Japanese Restaurant; and since someone doesn't like expensive restaurants, this person prefers affordable restaurants and should go to Pizza Nova.

Question: You are looking for an affordable restaurant. If I were you, between Red Lobster and City Fried Chicken I would go to
Answer: Since Red Lobster's price is a 2 and City Fried Chicken's is a 1, City Fried Chicken is more affordable than Red Lobster; and since you like affordable restaurants, if I were you I would go to City Fried Chicken."""


def generate_examples(is_factual_comparison):
    global factual_comparisons_chain_of_thought
    global decision_templates_chain_of_thought

    if is_factual_comparison:
        return factual_comparisons_chain_of_thought
    return decision_templates_chain_of_thought
    


def test_GPT3(test_cases, is_factual_comparison):
    tests = []
    for case in test_cases:
        statement = case['statement']
        answer = case['answer']
        wrong = case['candidate']
        prompt = f"{generate_examples(is_factual_comparison)}\n\nQuestion: {answer.join(statement.split(answer)[:-1]).strip()}\nAnswer:"
        
        response = openai.Completion.create(
            engine=config['model'],
            prompt=prompt,
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        generation = response['choices'][0]['text']

        sep = 'go to'
        if is_factual_comparison:
            sep = 'answer is'
        new_prompt = f"{prompt}{generation.split(sep)[0] + sep}"

        response = openai.Completion.create(
            engine=config['model'],
            prompt=new_prompt,
            temperature=0,
            max_tokens=2,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            logprobs=5
        )

        top_logits = response['choices'][0]['logprobs']['top_logprobs'][0].to_dict()
        top_percents = { key[1:] : math.exp(top_logits[key]) for key in top_logits }
        top_predictions = sorted(top_percents, key=top_percents.get, reverse=True)
        
        tests.append({'prompt' : prompt, 'answer' : answer, 'candidate' : wrong, 'predictions' : top_percents,\
                      'top_prediction' : top_predictions[0],\
                      'predicts_answer_before_candidate' : is_answer_before_candidate(answer, wrong, top_predictions)
                     })
    return tests


config_file = 'exp_config_chain_of_thought_restaurants.json'

config = None

with open(os.path.join(os.getcwd(), config_file), 'r') as f_config:
    config = json.load(f_config)

if not config:
    print(f"Couldn't load config file: {config_file}")
    exit()


openai.api_key = config['openai_key']


df_data = pd.read_csv(os.path.join(os.getcwd(), '..', '..', 'data', 'testing', 'restaurants_domain', 'test_data_restaurants.csv'), index_col=0)
restaurants = list(df_data['name'].values)
print('Facts table loaded')

price_comparisons = [validate_pair(pair[0], pair[1], 'price') for pair in itertools.combinations(restaurants, 2)\
                    if validate_pair(pair[0], pair[1], 'price')]
print(f"Generated {len(price_comparisons)} pairs of restaurants for price comparison")

dist_comparisons = [validate_pair(pair[0], pair[1], 'dist') for pair in itertools.combinations(restaurants, 2)\
                    if validate_pair(pair[0], pair[1], 'dist')]
print(f"Generated {len(dist_comparisons)} pairs of restaurants for distance comparison")

samples = {}
samples['price'] = random.sample(price_comparisons, config['sample'])
samples['price_counter'] = samples['price'] 
samples['dist'] = random.sample(dist_comparisons, config['sample'])
samples['dist_counter'] = samples['dist']
print(f"Generated samples of N={config['sample']}")


successful_cases = {}
successful_cases['price'] = {}
successful_cases['price_counter'] = {}
successful_cases['dist'] = {}
successful_cases['dist_counter'] = {}

failed_cases = {}
failed_cases['price'] = {}
failed_cases['price_counter'] = {}
failed_cases['dist'] = {}
failed_cases['dist_counter'] = {}

segmented_cases = {}
segmented_cases['price'] = {}
segmented_cases['price_counter'] = {}
segmented_cases['dist'] = {}
segmented_cases['dist_counter'] = {}


comparison_scores = []
decision_scores = []
decision_scores_easy = []
decision_scores_hard = []


exp_namespace = f"{config['exp_name']} at {datetime.datetime.now()}"
os.mkdir(exp_namespace)

for template_name in config['factual']:
    # Select and save sample:
    dimension = template_name.split('_')[0]
    if template_name.endswith('_counter'):
        dimension = dimension + '_counter'
    successful_cases[dimension][template_name] = []
    failed_cases[dimension][template_name] = []
    
    test_cases = generate_statements(config['templates'][template_name], samples[dimension])
    
    os.mkdir(os.path.join(os.getcwd(), exp_namespace, template_name))
    with open(os.path.join(os.getcwd(), exp_namespace, template_name, 'sample.txt'), 'w') as f_sample:
        for statement in test_cases:
            f_sample.write(f"{statement}\n")
    print(f"Generated {config['sample']} statements with '{dimension}' data following template '{template_name}'")
    
    # Run experiment on GPT3:
    tests = test_GPT3(test_cases, (template_name in config['factual']))
    
    df_tests = pd.DataFrame(tests)
    df_tests.to_csv(os.path.join(os.getcwd(), exp_namespace, template_name, 'results.csv'))
    
    score = sum(df_tests['predicts_answer_before_candidate'].values) / len(df_tests)
    with open(os.path.join(os.getcwd(), exp_namespace, template_name, 'score.txt'), 'w') as f_result:
        f_result.write(f"{score}")
    print(f"For '{template_name}', GPT3 scored: {score}")
    comparison_scores.append(score)
    
    for i, test in enumerate(tests):
        if test['predicts_answer_before_candidate']:
            successful_cases[dimension][template_name].append(i)
        else:
            failed_cases[dimension][template_name].append(i)

# Split the data between cases where factual templates ALL succeeded and where they ALL failed
for dimension in successful_cases:
    sets = [set(successful_cases[dimension][template]) for template in successful_cases[dimension]]
    segmented_cases[dimension]['all_successful'] = list(sets[0].intersection(*sets))
    sets = [set(failed_cases[dimension][template]) for template in failed_cases[dimension]]
    segmented_cases[dimension]['all_failed'] = list(sets[0].intersection(*sets))

with open(os.path.join(os.getcwd(), exp_namespace, 'segments.txt'), 'w') as f_segments:
    f_segments.write(f"{segmented_cases}")


for template_name in config['templates']:
    if template_name in config['factual']:
        continue
    
    # Select and save sample:
    dimension = template_name.split('_')[0]
    if template_name.endswith('_counter'):
        dimension = dimension + '_counter'
    test_cases = generate_statements(config['templates'][template_name], samples[dimension])

    os.mkdir(os.path.join(os.getcwd(), exp_namespace, template_name))
    with open(os.path.join(os.getcwd(), exp_namespace, template_name, 'sample.txt'), 'w') as f_sample:
        for statement in test_cases:
            f_sample.write(f"{statement}\n")
    print(f"Generated {config['sample']} statements with '{dimension}' data following template '{template_name}'")

    # Run experiment on GPT3:
    tests = test_GPT3(test_cases, config['shots'])
    df_tests = pd.DataFrame(tests)
    df_tests.to_csv(os.path.join(os.getcwd(), exp_namespace, template_name, 'results.csv'))
    
    # Score the experiment on the entire data:
    score = sum(df_tests['predicts_answer_before_candidate'].values) / len(df_tests)
    print(f"For '{template_name}', GPT3 scored overall: {score}")
    decision_scores.append(score)
    
    # As well as on the data segments:
    segments = segmented_cases[dimension]
    segmented_scores = {}
    segmented_scores['factual_successful'] = 0
    segmented_scores['factual_failed'] = 0
    
    for i, test in enumerate(tests):
        if i in segments['all_successful']:
            if test['predicts_answer_before_candidate']:
                segmented_scores['factual_successful'] += 1
        if i in segments['all_failed']:
            if test['predicts_answer_before_candidate']:
                segmented_scores['factual_failed'] += 1
    
    factual_successful_N = len(segments['all_successful']) if len(segments['all_successful']) else 1
    factual_successful_score = segmented_scores['factual_successful'] / factual_successful_N
    print(f"On the subset of cases where factual templates ALL succeded (N={factual_successful_N}): {factual_successful_score}")
    decision_scores_easy.append(factual_successful_score)
    
    factual_failed_N = len(segments['all_failed']) if len(segments['all_failed']) else 1
    factual_failed_score = segmented_scores['factual_failed'] / factual_failed_N
    print(f"On the subset of cases where factual templates ALL failed (N={factual_failed_N}): {factual_failed_score}")
    decision_scores_hard.append(factual_failed_score)
    
    with open(os.path.join(os.getcwd(), exp_namespace, template_name, 'scores.txt'), 'w') as f_result:
        f_result.write(f"Overall: {score}\n")
        f_result.write(f"Subset of cases where factual templates ALL succeeded: {factual_successful_score}\n")
        f_result.write(f"Subset of cases where factual templates ALL failed: {factual_failed_score}")


print("\n")
with open(os.path.join(os.getcwd(), exp_namespace, 'final_scores.txt'), 'w') as f_result:
    series = pd.Series(data=comparison_scores)
    output = f"Score on factual comparisons: {round(series.mean(), 3)} ± {round(series.std(), 3)}"
    print(output)
    f_result.write(f"{output}\n")

    series = pd.Series(data=decision_scores)
    output = f"Total score on decision templates: {round(series.mean(), 3)} ± {round(series.std(), 3)}"
    print(output)
    f_result.write(f"{output}\n")

    series = pd.Series(data=decision_scores_easy)
    output = f"Score on decision templates (EASY): {round(series.mean(), 3)} ± {round(series.std(), 3)}"
    print(output)
    f_result.write(f"{output}\n")

    series = pd.Series(data=decision_scores_hard)
    output = f"Score on decision templates (HARD): {round(series.mean(), 3)} ± {round(series.std(), 3)}"
    print(output)
    f_result.write(output)
print("\n")

