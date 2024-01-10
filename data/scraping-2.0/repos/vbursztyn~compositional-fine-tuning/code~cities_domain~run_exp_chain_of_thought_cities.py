

import os, json

import pandas as pd

import itertools

import datetime

import random

import openai

import math


def validate_city_pair(city_A, city_B, attribute):
    value_A = df_data[df_data['city'] == city_A][attribute].values[0]
    value_B = df_data[df_data['city'] == city_B][attribute].values[0]
    if attribute == 'avg_temp':
        value_A = float(value_A.split('(')[0].replace('−','-'))
        value_B = float(value_B.split('(')[0].replace('−','-'))
        if (value_A - value_B) >= config['temp_diff']:
            return (city_A, city_B)
        if (value_A - value_B) <= config['temp_diff']*(-1):
            return (city_B, city_A)
    else:
        if (value_A - value_B) >= config['pop_diff']:
            return (city_A, city_B)
        if (value_A - value_B) <= config['pop_diff']*(-1):
            return (city_B, city_A)
    return False


def generate_statements(template_format, pairs):
    template = template_format['question_template']
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



factual_comparisons_chain_of_thought = """Question: Between Seattle and Rio de Janeiro, the city with higher average temperature is
Answer: Since Seattle's average temperature is 11.3C and Rio de Janeiro's is 23.8C, Rio de Janeiro has a higher average temperature than Seattle, so the answer is Rio de Janeiro.

Question: Between Accra and Sochi, the city with warmer weather is
Answer: Since Accra's average temperature is 26.4C and Sochi's is 14.2C, Accra is warmer than Sochi, so the answer is Accra.

Question: Between Barcelona and Bissau, the city with a smaller population is
Answer: Since Barcelona's population is 4.6M people and Bissau's is 0.4M people, Bissau has a smaller population than Barcelona, so the answer is Bissau.

Question: Between Oulu and Singapore, the smaller city is 
Answer: Since Oulu's population is 0.2M people and Singapore's is 5.7M people, Oulu is smaller than Singapore, so the answer is Oulu.

Question: Between Mar del Plata and Havana, the city with lower average temperature is
Answer: Since Mar del Plata's average temperature is 13.9C and Havana's is 25.2C, Mar del Plata has a lower average temperature than Havana, so the answer is Mar del Plata.

Question: Between Taipei and Cusco, the city with colder weather is
Answer: Since Taipei's average temperature is 23.0C and Cusco's is 12.5C, Cusco is colder than Taipei, so the answer is Cusco.

Question: Between Tokyo and Livingstone, the city with a larger population is
Answer: Since Tokyo's population is 38.0M people and Livingstone's is 0.1M people, Tokyo has a bigger population than Livingstone, so the answer is Tokyo.

Question: Between Vancouver and São Paulo, the bigger city is
Answer: Since Vancouver's population is 2.3M people and São Paulo's is 22.0M people, São Paulo is bigger than Vancouver, so the answer is São Paulo."""


decision_templates_chain_of_thought = """Question: I'm looking for a city with warm weather. Between Seattle and Rio de Janeiro, I should visit
Answer: Since Seattles's average temperature is 11.3C and Rio de Janeiro's is 23.8C, Rio de Janeiro is warmer than Seattle; and since I like warm cities, I should visit Rio de Janeiro.

Question: You don't like the cold weather. Between Accra and Sochi, you should visit
Answer: Since Accra's average temperature is 26.4C and Sochi's is 14.2C, Accra is warmer than Sochi; and since you don't like cold cities, you prefer warmer cities and should visit Accra.

Question: Someone is looking for a big city to visit. Between Tokyo and Livingstone, this person should visit
Answer: Since Tokyo's population is 38.0M people and Livingstone's is 0.1M people, Tokyo is bigger than Livingstone; and since someone likes big cities, this person should visit Tokyo.

Question: You don't like small cities. If I were you, between Vancouver and São Paulo I would visit
Answer: Since Vancouver's population is 2.3M people and São Paulo's is 22.0M people, São Paulo is bigger than Vancouver; and since you don't like small cities, you prefer bigger cities and if I were you I would visit São Paulo.

Question: I don't like big cities. Between Barcelona and Bissau, I should visit
Answer: Since Barcelona's population is 4.6M people and Bissau's is 0.4M people, Bissau is smaller than Barcelona; and since I don't like big cities, I prefer smaller cities and should visit Bissau.

Question: You are looking for a small city to visit. Between Oulu and Singapore, you should visit
Answer: Since Oulu's population is 0.2M people and Singapore's is 5.7M people, Oulu is smaller than Singapore; and since you like small cities, you should visit Oulu.

Question: Someone doesn't like hot weather. Between Mar del Plata and Havana, this person should visit
Answer: Since Mar del Plata's average temperature is 13.9C and Havana's is 25.2C, Mar del Plata is colder than Havana; and since someone doesn't like hot cities, this person prefers colder cities and should visit Mar del Plata.

Question: You are looking for a city with cold weather. If I were you, between Taipei and Cusco I would
Answer: Since Taipei's average temperature is 23.0C and Cusco's is 12.5C, Cusco is colder than Taipei; and since you like cold cities, if I were you I would visit Cusco."""


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

        sep = 'visit'
        if is_factual_comparison:
            sep = 'answer is'
        new_prompt = f"{prompt}{generation.split(sep)[0] + sep}"

        print(new_prompt)

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


config_file = 'exp_config_chain_of_thought_cities.json'

config = None

with open(os.path.join(os.getcwd(), config_file), 'r') as f_config:
    config = json.load(f_config)

if not config:
    print(f"Couldn't load config file: {config_file}")
    exit()


openai.api_key = config['openai_key']


df_data = pd.read_csv(os.path.join(os.getcwd(), '..', '..', 'data', 'testing', 'cities_domain', 'test_data_cities.csv'), index_col=0)
cities = list(df_data['city'].values)
print('Facts table loaded')

temp_comparisons = [validate_city_pair(pair[0], pair[1], 'avg_temp') for pair in itertools.combinations(cities, 2)\
                    if validate_city_pair(pair[0], pair[1], 'avg_temp')]
print(f"Generated {len(temp_comparisons)} pairs of cities for temperature comparison")

pop_comparisons = [validate_city_pair(pair[0], pair[1], 'pop') for pair in itertools.combinations(cities, 2)\
                    if validate_city_pair(pair[0], pair[1], 'pop')]
print(f"Generated {len(pop_comparisons)} pairs of cities for population comparison")

samples = {}
samples['temp'] = random.sample(temp_comparisons, config['sample'])
samples['temp_counter'] = samples['temp'] 
samples['pop'] = random.sample(pop_comparisons, config['sample'])
samples['pop_counter'] = samples['pop']
print(f"Generated samples of N={config['sample']}")


successful_cases = {}
successful_cases['temp'] = {}
successful_cases['temp_counter'] = {}
successful_cases['pop'] = {}
successful_cases['pop_counter'] = {}

failed_cases = {}
failed_cases['temp'] = {}
failed_cases['temp_counter'] = {}
failed_cases['pop'] = {}
failed_cases['pop_counter'] = {}

segmented_cases = {}
segmented_cases['temp'] = {}
segmented_cases['temp_counter'] = {}
segmented_cases['pop'] = {}
segmented_cases['pop_counter'] = {}


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

