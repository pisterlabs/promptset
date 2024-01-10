from probing_experiments.prompts import *
from probing_experiments.wvs_prompts import \
    COUNTRIES_WVS_W7
from src.probing_experiments.prompts import get_pew_moral_df,get_wvs_df, get_user_study_scores
from src.probing_experiments.wvs_prompts import wvs_gpt3_prompts_ratings, RATING_OPTIONS_WVS
import time
import pickle
import pandas as pd
import numpy as np
import openai
from probing_experiments.gpt3_api import get_gpt3_response_logprobs, get_gpt3_response, API_KEY

PERIOD = '.'
GPT3_REP = 5


def access_gp3(question, question_prompt, choices, country_responses, repeat_num):

    for i in range(repeat_num):
        lm_response = get_gpt3_response(question_prompt)
        while lm_response not in choices:
            lm_response = get_gpt3_response(question_prompt)

        lm_score = float(lm_response[0]) - 2
        country_responses[question].append(lm_score)

    question_var = np.var(country_responses[question])
    question_mean = np.mean(country_responses[question])
    return question_var, question_mean

def compare_gpt3_pew(repeat = GPT3_REP, cultures : list = None):


    pew_df = get_pew_moral_df()
    if cultures == None:
        cultures = list(pew_df['COUNTRY'].unique())
    gpt3_all = []
    for culture in cultures:
        prompts = pew_gpt3_prompts_ratings(pew_df, culture)


        country_rows = []
        country_responses = {}
        for question,ratings in prompts.items():
            country_responses[question] = []
            pew_score = ratings[2]
            question_prompt = ratings[1]
            question_var,question_scores = access_gp3(question, question_prompt,
                                                     RATING_OPTIONS_PEW, country_responses, repeat)

            row = {'country': culture, 'prompt': question_prompt, 'topic': question, 'pew_rating': ratings[0],
                   'pew_score': pew_score, 'gpt3_score_mean':question_scores, 'gpt_var':question_var,
                   'repeat' : repeat}
            gpt3_all.append(row)
            country_rows.append(row)
            time.sleep(30)


    df = pd.DataFrame(gpt3_all)
    df.to_csv('data/pew_gpt3.csv', index = False )


def universal_gpt3_pew(repeat = GPT3_REP, cultures : list = None):
    pew_df = get_pew_moral_df()

    if cultures == None:
        cultures = list(pew_df['COUNTRY'].unique())
    gpt3_all = []
    responses = {}
    universal_prompts = get_universal_prompts_for_gpt3()
    for question, question_prompt in universal_prompts.items():

        question_scores = []
        responses[question] = question_scores
        question_var, question_scores = access_gp3(question, question_prompt, RATING_OPTIONS_PEW,
                                                   responses, repeat)

        for culture in cultures:
            prompts = pew_gpt3_prompts_ratings(pew_df, culture)
            rating_scores = []
            ratings = prompts[question]

            pew_score = ratings[2] #rating_answers['Morally acceptable'] - rating_answers['Morally unacceptable']
            rating_scores.append(pew_score)

            row = {'country': culture, 'prompt': question_prompt, 'topic': question, 'pew_rating': ratings[0],
                   'pew_score': pew_score, 'gpt3_score_mean':question_scores, 'gpt_var':question_var,
                   'repeat' : repeat}
            gpt3_all.append(row)


    df = pd.DataFrame(gpt3_all)
    df.to_csv('data/universal_pew_gpt3.csv', index = False )



def compare_gpt3_wvs(repeat = GPT3_REP, cultures : list = None, wave = 7,
                     extend = False, rating_type = ''):

    wvs_df = get_wvs_df(wave)
    cultures = COUNTRIES_WVS_W7 if cultures == None else cultures
    extended_rating = True if rating_type == '_extended' else False

    gpt3_all = []

    for culture in cultures:
        country_responses = {}
        if extend:
            country_responses = pickle.load(open(f'data/WVS/{culture}_wvs_w{wave}_gpt3.p', 'rb'))
        prompts = wvs_gpt3_prompts_ratings(wvs_df, culture, wave=wave,extended_rating=extended_rating)

        country_rows = []
        for question,ratings in prompts.items():
            if not extend:
                country_responses[question] = []
                repeat_num = repeat
            else:
                repeat_num = repeat - len(country_responses[question])

            wvs_score = ratings[0]
            question_prompt = ratings[1]

            question_var,question_scores= access_gp3(question, question_prompt,
                                                     RATING_OPTIONS_WVS, country_responses, repeat_num)

            row = {'country': culture, 'prompt': question_prompt, 'topic': question,
                   'wvs_score': wvs_score, 'gpt3_score_mean':question_scores, 'gpt_var':question_var,
                   'repeat' : repeat}

            gpt3_all.append(row)
            country_rows.append(row)
            time.sleep(30)

        pickle.dump(country_responses, open(f'data/WVS/{culture}_wvs_w{wave}_gpt3{rating_type}.p', 'wb'))

    df = pd.DataFrame(gpt3_all)
    df.to_csv(f'data/wvs_w{wave}_gpt3{rating_type}.csv', index = False )

def get_gpt3_log_prob_difference(moral_prompts, nonmoral_prompts, rating_pairs):
    all_log_probs_m = get_gpt3_response_logprobs(moral_prompts)
    all_log_probs_nm = get_gpt3_response_logprobs(nonmoral_prompts)
    question_lm_scores = []
    for i, rating in enumerate(rating_pairs):
        tokens_m, log_probs_m = all_log_probs_m[i][0], all_log_probs_m[i][1]
        tokens_nm, log_probs_nm = all_log_probs_nm[i][0], all_log_probs_nm[i][1]

        log_prob_m = log_probs_m[tokens_m.index(PERIOD) - 1]
        log_prob_nm = log_probs_nm[tokens_nm.index(PERIOD) - 1]

        lm_score = log_prob_m - log_prob_nm
        question_lm_scores.append(lm_score)


    lm_score = np.mean(question_lm_scores)
    return lm_score


def compare_gpt3_pew_token_pairs(cultures : list = None,
                                 excluding_topics : list = [],
                                 prompt_mode = 'in'):

    from src.probing_experiments.compare_prompt_responses.compare_gpt2 import \
        pew_gpt2_prompts_ratings_multiple_tokens as gpt3_pew_prompts

    openai.api_key  = API_KEY


    pew_df = get_pew_moral_df()
    if cultures == None:
        cultures = list(pew_df['COUNTRY'].unique())

    cultures.append('')
    gpt3_all = []

    for culture in cultures:
        prompts = gpt3_pew_prompts(pew_df, culture, prompt_mode)
        culture = culture if culture != '' else 'universal'
        rating_scores,text_questions,country_rows = [],[],[]


        for question,rating_pairs in prompts.items():
            question_lm_scores = []
            moral_log_probs = []
            nonmoral_log_probs = []
            if any([x in question for x in excluding_topics]):
                continue

            moral_prompts = [rating[0] for rating in rating_pairs]
            nonmoral_prompts = [rating[1] for rating in rating_pairs]

            all_log_probs_m = get_gpt3_response_logprobs(moral_prompts)
            all_log_probs_nm = get_gpt3_response_logprobs(nonmoral_prompts)

            for i, rating in enumerate(rating_pairs):
                pew_score = rating[2]
                tokens_m, log_probs_m = all_log_probs_m[i][0], all_log_probs_m[i][1]
                tokens_nm, log_probs_nm = all_log_probs_nm[i][0], all_log_probs_nm[i][1]

                log_prob_m = log_probs_m[tokens_m.index(PERIOD) - 1]
                log_prob_nm = log_probs_nm[tokens_nm.index(PERIOD) - 1]

                lm_score = log_prob_m - log_prob_nm
                question_lm_scores.append(lm_score)
                moral_log_probs.append(log_prob_m)
                nonmoral_log_probs.append(log_prob_nm)

            lm_score = np.mean(question_lm_scores)
            moral_log_probs = np.mean(moral_log_probs)
            nonmoral_log_probs = np.mean(nonmoral_log_probs)

            rating_scores.append(pew_score)
            text_questions.append(question)
            row = {'country': culture,
                   'topic': question,  'pew_score': pew_score, 'moral log prob' : moral_log_probs,
                   'non moral log probs': nonmoral_log_probs, 'log prob difference' : lm_score}
            gpt3_all.append(row)
            country_rows.append(row)

    df = pd.DataFrame(gpt3_all)

    save_dir = f'data/pew_gpt3_token_pairs_{prompt_mode}.csv'
    df.to_csv(save_dir, index = False )


def compare_gpt3_wvs_token_pairs(cultures : list = None,
                                 wave = 7,
                                 excluding_topics = [],
                                 excluding_cultures = [],
                                 prompt_mode = 'in'):

    from src.probing_experiments.compare_prompt_responses.compare_gpt2 \
        import wvs_gpt2_prompts_ratings_multiple_tokens as wvs_gpt3_prompts


    wvs_df = get_wvs_df(wave)
    if cultures == None:
        cultures = COUNTRIES_WVS_W7
    cultures.append('')
    gpt3_all = []
    for culture in cultures:

        if culture in excluding_cultures:
            continue
        prompts = wvs_gpt3_prompts(wvs_df, culture, prompt_mode=prompt_mode)
        culture = culture if culture != '' else 'universal'

        rating_scores, text_questions, country_rows = [],[],[]


        for question,rating_pairs in prompts.items():
            if any([x in question for x in excluding_topics]):
                continue

            question_average_lm_score = []
            moral_log_probs = []
            nonmoral_log_probs = []

            moral_prompts = [rating[0] for rating in rating_pairs]
            nonmoral_prompts = [rating[1] for rating in rating_pairs]

            all_log_probs_m = get_gpt3_response_logprobs(moral_prompts)
            all_log_probs_nm = get_gpt3_response_logprobs(nonmoral_prompts)

            for i, rating in enumerate(rating_pairs):
                wvs_score = rating[2] #rating_answers['Morally acceptable'] - rating_answers['Morally unacceptable']
                tokens_m, log_probs_m = all_log_probs_m[i][0], all_log_probs_m[i][1]
                tokens_nm, log_probs_nm = all_log_probs_nm[i][0], all_log_probs_nm[i][1]

                log_prob_m = log_probs_m[tokens_m.index(PERIOD) - 1]
                log_prob_nm = log_probs_nm[tokens_nm.index(PERIOD) - 1]

                lm_score = log_prob_m - log_prob_nm
                question_average_lm_score.append(lm_score)
                moral_log_probs.append(log_prob_m)
                nonmoral_log_probs.append(log_prob_nm)

            lm_score = np.mean(question_average_lm_score)

            rating_scores.append(wvs_score)
            text_questions.append(question)
            row = {'country': culture, 'topic': question,  'wvs_score': wvs_score,
                   'moral log prob' : np.mean(moral_log_probs),'non moral log probs': np.mean(nonmoral_log_probs),
                   'log prob difference' : lm_score}

            gpt3_all.append(row)
            country_rows.append(row)

    df = pd.DataFrame(gpt3_all)

    save_dir = f'data/wvs_w{wave}_gpt3_token_pairs_{prompt_mode}.csv'
    df.to_csv(save_dir, index = False)



def universal_gpt3_wvs(repeat = GPT3_REP, cultures : list = None, wave = 7):

    wvs_df = get_wvs_df(wave)

    if cultures == None:
        cultures = COUNTRIES_WVS_W7
    gpt3_all = []
    responses = {}
    universal_prompts = wvs_gpt3_prompts_ratings(wvs_df)
    for question, question_prompt in universal_prompts.items():
        question_scores = []
        responses[question] = question_scores
        question_var, question_scores = access_gp3(question, question_prompt, RATING_OPTIONS_WVS,
                                                   responses, repeat)
        time.sleep(30)

        for culture in cultures:
            prompts = wvs_gpt3_prompts_ratings(wvs_df, culture)
            rating_scores = []
            ratings = prompts[question]

            wvs_score = ratings[0]
            rating_scores.append(wvs_score)

            row = {'country': culture, 'prompt': question_prompt, 'topic': question,
                   'wvs_score': wvs_score, 'gpt3_score_mean':question_scores, 'gpt_var':question_var,
                   'repeat' : repeat}
            gpt3_all.append(row)

    df = pd.DataFrame(gpt3_all)
    df.to_csv(f'data/universal_wvs_w{wave}_gpt3.csv', index = False )


def compare_gpt3_prompts_mort_user_study(repeat = GPT3_REP, user_study = 'globalAMT'):

    universal_mort_prompts = mort_prompts_pew_style(include_atoms = True)
    user_df = pd.read_csv(f'data/MoRT_actions/userStudy_scores_{user_study}.csv')
    list_rows = []
    responses = {}
    for question_prompt,aa, aci in universal_mort_prompts:
        question_scores = []
        responses[question_prompt] = question_scores

        if aci == '':
            user_score = user_df.loc[user_df.action == aa]['score']
        else:
            user_score = user_df.loc[user_df.action.str.contains(aa)].loc[user_df.action.str.contains(aci)]['score']
        if len(user_score) == 0:
            continue

        user_score = (float(list(user_score)[0]) - 0.5) / 0.5
        question_var, question_scores = access_gp3(question_prompt, question_prompt,
                                                   RATING_OPTIONS_PEW, responses, repeat)

        row = {'prompt': aa + ' ' + aci,'gpt3_score_mean':question_scores, 'gpt_var':question_var,
               'repeat' : repeat, 'user_score' : user_score}
        list_rows.append(row)
        df = pd.DataFrame(list_rows)
        df.to_csv(f'data/MoRT_actions/gpt3_mort_userstudy_{user_study}.csv', index = False)
        time.sleep(30)


def compare_paired_moral_non_moral_probs(prompts):
    for row in prompts:
        moral_prompts = row['moral prompt']
        nonmoral_prompts = row['nonmoral prompt']
        rating_pairs = [(mp , non_mp) for (mp , non_mp) in zip(moral_prompts,nonmoral_prompts)]
        q_lm_scores = get_gpt3_log_prob_difference(moral_prompts, nonmoral_prompts, rating_pairs)
        row['log prob difference'] = q_lm_scores

    return prompts

def compare_gpt3_prompts_mort_user_study_token_pairs(user_study = 'globalAMT', style = 'mv_at_end'):

    from src.probing_experiments.compare_prompt_responses.compare_gpt2 import\
        gpt2_mort_prompts_multiple_tokens as gpt3_mort_prompts_multiple_tokens

    prompts = gpt3_mort_prompts_multiple_tokens(include_atoms= True, style = style)
    prompts = get_user_study_scores(prompts, user_study = 'globalAMT')
    new_prompts = compare_paired_moral_non_moral_probs(prompts)
    df = pd.DataFrame(new_prompts)
    df.to_csv(f'data/MoRT_actions/gpt3_mort_userstudy_{user_study}_token_pairs.csv', index = False)


def compare_gpt3_pew_memorization(topic, repeat = 3):


    prompts = gpt3_pew_memorization_prompts(topic)
    ratings = {}
    for country, question in prompts.items():
        ratings[country] = []
        for r in range(repeat):
            response = get_gpt3_response(question)
            if '%' in response:

                rate = (response[response.index('%') - 2: response.index('%')])
                if len(rate) not in [1, 2]:
                    continue

                ratings[country].append(rate)
            time.sleep(30)

    return ratings

def compare_gpt3():
    store_small_pew()
    #GPT3-PROBS
    compare_gpt3_prompts_mort_user_study_token_pairs()
    compare_gpt3_pew_token_pairs(prompt_mode = 'in')
    compare_gpt3_wvs_token_pairs(prompt_mode = 'in')

    #GPT3-QA
    universal_gpt3_pew(repeat=GPT3_REP)
    compare_gpt3_pew(repeat = GPT3_REP, cultures=PEW_COUNTRIES)
    compare_gpt3_wvs(repeat=GPT3_REP, cultures=COUNTRIES_WVS_W7)
    compare_gpt3_prompts_mort_user_study(repeat= GPT3_REP, user_study='globalAMT')

if __name__ =='__main__':
    compare_gpt3()