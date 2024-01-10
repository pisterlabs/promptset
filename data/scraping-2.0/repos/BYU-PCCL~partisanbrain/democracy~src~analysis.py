import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from matplotlib import pyplot as plt
from pdb import set_trace as breakpoint
import pickle
import openai
import os

def read_data(path):
    '''
    Given a path to a csv, read the data into a pandas dataframe.
    Columns:
        prompt (str): the prompt
        possible_answers (list of strings): potential responses
        api_resp (dict): the response from the api
    '''
    data = pd.read_csv(path)
    # convert possible_answers to a list
    data['possible_answers'] = data['possible_answers'].apply(lambda x: eval(x))
    # convert api_resp to a dict
    data['api_resp'] = data['api_resp'].apply(lambda x: eval(x))
    # TODO - best idea? Change?
    # add party column, 'democrat' if the path contains 'democrat', 'republican' if the prompt contains 'republican', else 'other'
    party = 'other'
    if 'democrat' in path:
        party = 'democrat'
    elif 'republican' in path:
        party = 'republican'
    data['party'] = party
    # map dv_idx to dv
    name_map = {
        0: '0: reduce polling stations',
        1: '1: ignore court rulings',
        2: '2: prosecute journalists',
        3: '3: reject election results',
        4: '4: threatening messages',
        5: '5: harrassment',
        6: '6: violence for political goals',
        7: '7: violence if win more races',
        8: '8: temperature',
        9: '9: money split',
    }
    # map using name_map
    data['dv'] = data['dv_idx'].map(name_map)
    # map dv_idx to category
    cat_map = {
        0: 'anti-democratic',
        1: 'anti-democratic',
        2: 'anti-democratic',
        3: 'anti-democratic',
        4: 'violence',
        5: 'violence',
        6: 'violence',
        7: 'violence',
        8: 'polarization',
        9: 'polarization',
    }
    # map using cat_map
    data['category'] = data['dv_idx'].map(cat_map)
    # parse
    data = parse_responses(data)
    def treatment(row):
        # if 'treatment' is 'composite' or 'kalmoe'
        if row['treatment'] in ['composite', 'kalmoe']:
            possible_quotes = [
                'McConnell',
                'Cruz',
                'Biden',
                'Pelosi',
            ]
            for person in possible_quotes:
                if person in row['prompt']:
                    return row['treatment'] + '-' + person
        else:
            return row['treatment']
    data['treatment'] = data.apply(treatment, axis=1)
    return data

def read_many_data(paths):
    '''
    Read data from each path and combine into a single dataframe.
    '''
    dfs = []
    for path in paths:
        dfs.append(read_data(path))
    return pd.concat(dfs)

def make_good_bad_dict():
    """Returns a dictionary which maps from the dichotomous 
    outcomes across all DVs to the words "good" and "bad" to make them easier.

    Note, it does NOT map from TOKENS, but rather outcomes, so you want to
    do this mapping after you've already collapsed probabilities from all 
    tokens into a single outcome, which should be represented by the keys of
    this dictionary.
    """
    return {
        'reduce':'bad',
        'maintain':'good',
        'ignore':'bad',
        'heed':'good',
        'prosecute':'bad',
        'tolerate':'good',
        'accept':'good',
        'deny':'bad',
        'always':'bad',
        'never':'good',
        'warm':'good',
        'cold':'bad',
        'all of it':'good',
        'none of it':'bad',
    }

def get_logprobs(response):
    '''
    Given an instance of gpt3 response, return the logprobs of the first sampled token.
    Returns a sorted list of tuples of (token, logprob)
    '''
    # logprobs = response['choices'][0].logprobs.top_logprobs[0]
    logprobs = response['choices'][0]['logprobs']['top_logprobs'][0]
    # sort by logprob
    logprobs = sorted(logprobs.items(), key=lambda x: x[1], reverse=True)
    return logprobs

def matches(token, category, matching_strategy):
    '''
    Returns True if token matches category according to the matching strategy.
    Arguments:
        token: the token to match
        category: the category to match
        matching_strategy: the matching strategy to use
            Accepted values:
                'starts_with': token starts with category
                'substr': token is a substring of category
                'exact': token is the exact category
    '''
    # strip token and category and change to lowercase
    token = token.lower().strip()
    category = category.lower().strip()

    # check matching strategy
    if matching_strategy == 'starts_with':
        return category.startswith(token)
    elif matching_strategy == 'substr':
        return token in category
    elif matching_strategy == 'exact':
        return category == token
    else:
        raise ValueError('Invalid matching strategy')

def parse_response(response, candidates, matching_strategy='starts_with'):
    '''
    Given the response, measure the total probability mass
    on each candidate.
    '''
    # strip and lowercase all candidates
    candidates = [c.lower().strip() for c in candidates]
    logprobs = get_logprobs(response)
    # get category probabilities
    cand_probs = {cand: 0 for cand in candidates}

    for token, logprob in logprobs:
        # see if lower and strip is a candidate
        token = token.lower().strip()
        for cand in candidates:
            if matches(token, cand, matching_strategy):
                cand_probs[cand] += np.exp(logprob)
    # normalize, and store coverage
    coverage = sum(cand_probs.values())
    cand_probs = {cand: prob/coverage for cand, prob in cand_probs.items()}
    cand_probs['coverage'] = coverage
    return cand_probs

def parse_responses(df):
    '''
    Given a dataframe of responses, parse the responses and add columns to df.
    Get the response from the 'api_resp' column, and the candidates from the 'possible_answers' column.
    '''
    df['parsed_resp'] = df.apply(lambda x: parse_response(x['api_resp'], x['possible_answers']), axis=1)
    # make 'score' column which contains the probability of 'good' outcome
    good_dict = make_good_bad_dict()
    def get_score(row):
        resp = row['parsed_resp']
        for key in resp.keys():
            if good_dict[key] == 'good':
                return resp[key]
        raise ValueError('No good outcome found')
    def get_dv(row):
        resp = row['parsed_resp']
        for key in resp.keys():
            if good_dict[key] == 'good':
                return key
        raise ValueError('No good outcome found')
    def get_coverage(row):
        resp = row['parsed_resp']
        return resp['coverage']
    df['score'] = df.apply(lambda x: get_score(x), axis=1)
    # df['dv'] = df.apply(lambda x: get_dv(x), axis=1)
    df['coverage'] = df.apply(lambda x: get_coverage(x), axis=1)
    return df

def make_dict(df, key, value):
    '''
    Given a dataframe, make a dictionary where the keys are the unique values of the key column,
    and the values are all the values of the value column for that key.
    '''
    return {k: df[value][df[key] == k] for k in df[key].unique()}

def plot_bar(treatment_dict, y_label='', x_label='', title='', save_path=''):
    '''
    Given a dictionary of results, plot a bar chart.
    Arguments:
        treatment_dict: a dictionary of the treatment results. The keys are the treatment names,
            and the values are the treatment results (array-like).
    '''
    treatments = list(treatment_dict.keys())
    means, stds = [], []
    for treatment in treatments:
        results = treatment_dict[treatment]
        mean = np.mean(results)
        std = np.std(results)
        means.append(mean)
        stds.append(std)
    plt.figure(figsize=(10,7))
    plt.bar(treatments, means, yerr=stds)
    # rotate x labels
    plt.xticks(rotation=15)
    if y_label:
        plt.ylabel(y_label)
    if x_label:
        plt.xlabel(x_label)
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def grouped_bar(means, stds, x_label='', y_label='', title='', save_path=''):
    '''
    Given a 2d dataframe of means and stds, plot a grouped bar chart.
    Arguments:
        means: a 2d dataframe of means. Index is the x-axis, columns are the groups (colors).
        stds: a 2d dataframe of stds. Index is the x-axis, columns are the groups (colors).
        x_label: the x-axis label
        y_label: the y-axis label
        save_path: the path to save the plot to.
    '''
    n_x, n_groups = means.shape
    x = np.arange(n_x)
    width = 1/(n_groups+1)

    x_labels = means.index.values
    groups = means.columns.values

    plt.figure(figsize=(10,7))
    for i, group in enumerate(groups):
        # plt.bar(x + (i - n_groups/2)*width, means[group], width, yerr=stds[group], label=group[:10], color=f'C{i}', alpha=.8)
        plt.bar(x + (i - n_groups/2)*width, means[group], width, yerr=stds[group], label=group, color=f'C{i}', alpha=.8)

    plt.xticks(x - width/2, x_labels)
    # rotate x labels
    plt.xticks(rotation=15, ha='right')

    plt.legend()
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def get_means_and_stds(df, index, col, value, scale_by_n=False):
    '''
    Groupby the index and col, and get the mean and std of the value column.
    '''
    agg = df.groupby([index, col]).agg({value: ['mean', 'std', 'count']})
    means, std = agg[value]['mean'], agg[value]['std']
    # unstack the index and col
    means = means.unstack(level=1)
    stds = std.unstack(level=1)
    counts = agg[value]['count'].unstack(level=1)
    if scale_by_n:
        stds = stds/np.sqrt(counts)
    return means, stds



def t_test(treatment_dict):
    '''
    Given a dictionary of treatment results, perform a t-test on each treatment.
    Arguments:
        treatment_dict: a dictionary of the treatment results. The keys are the treatment names,
            and the values are the treatment results (array-like).
    '''
    treatments = list(treatment_dict.keys())
    p_values = pd.DataFrame(index=treatments, columns=treatments)
    for treatment1 in treatments:
        for treatment2 in treatments:
            if treatment1 == treatment2:
                continue
            else:
                p_value = ttest_ind(treatment_dict[treatment1], treatment_dict[treatment2])[1]
                p_values.loc[treatment1, treatment2] = p_value
    return p_values




if __name__ == '__main__':
    # prompt = ''' Would you say that you agree or disagree that mexican food is good?
    # I would say that I'''

    # response = openai.Completion.create(
        # prompt=prompt,
        # max_tokens=1,
        # logprobs=100,
        # engine='ada',
    # )

    # candidates = ['agree', 'disagree']
    # print(parse_response(response, candidates))

    # treatments = {
        # 'a': 3 * np.random.rand(100) + 1,
        # 'b': 2 * np.random.rand(100) + 1.4,
    # }
    # # plot_treatments(treatments, y_label='Mean', x_label='Treatments', save_path='test.png')
    # print(t_test(treatments))

    ###########
    # # read in data from data/kalmoe_republican.csv and data/passive_republican.csv
    # # merge in both into one dataframe
    # data1, data2 = read_data('data/kalmoe_republican.csv'), read_data('data/passive_republican.csv')
    # data1, data2 = parse_responses(data1), parse_responses(data2)

    # breakpoint()
    # # make a dictionary of the dataframes
    # dict1 = make_dict(data1, 'dv', 'score')
    # dict2 = make_dict(data2, 'dv', 'score')
    # # plot
    # plot_bar(dict1, y_label='Score', x_label='dv', save_path='test.png')

    ###########
    # # read in many dataframes
    # paths = [
    #     'data/passive_democrat.csv',
    #     'data/passive_republican.csv',
    #     # 'data/kalmoe_democrat.csv',
    #     'data/kalmoe_republican.csv',
    # ]
    # data = read_many_data(paths)
    # # data = parse_responses(data)
    # # plot 'dv' againts 'score'
    # dict1 = make_dict(data, 'dv', 'score')
    # plot_bar(dict1, y_label='Score', x_label='dv', save_path='test.pdf')
    # # plot 'dv' againts 'coverage'
    # dict2 = make_dict(data, 'dv', 'coverage')
    # plot_bar(dict2, y_label='Coverage', x_label='dv', save_path='coverage.pdf')
    # # plot 'education' against 'score'
    # dict3 = make_dict(data, 'education', 'score')
    # plot_bar(dict3, y_label='Score', x_label='education', save_path='education.pdf')
    # # plot 'treatment' against 'score'
    # dict4 = make_dict(data, 'treatment', 'score')
    # plot_bar(dict4, y_label='Score', x_label='treatment', save_path='treatment.pdf')

    ###########
    # test grouped bar
    # means = pd.DataFrame([
    #     ['A', 10, 20, 10, 30],
    #     ['B', 20, 25, 15, 25],
    #     ['C', 12, 15, 19, 6],
    #     ['D', 10, 29, 13, 19]],
    #     columns=['Team', 'Round 1', 'Round 2', 'Round 3', 'Round 4']
    # )
    # # make 'Team' index
    # means.set_index('Team', inplace=True)
    # stds = pd.DataFrame([
    #     ['A', 1, 2, 1, 3],
    #     ['B', 2, 2, 2, 2],
    #     ['C', 1, 1, 1, 1],
    #     ['D', 1, 3, 1, 1]],
    #     columns=['Team', 'Round 1', 'Round 2', 'Round 3', 'Round 4']
    # )
    # # make 'Team' index
    # stds.set_index('Team', inplace=True)
    # grouped_bar(means, stds, save_path='bar.pdf')

    ###########
    # get means and std for 'dv' and 'treatment', averaging over 'score'
    # paths = [
    #     'data/passive_democrat.csv',
    #     'data/passive_republican.csv',
    #     # 'data/kalmoe_democrat.csv',
    #     'data/kalmoe_republican.csv',
    # ]
    # data = read_many_data(paths)
    # means, stds = get_means_and_stds(data, 'dv', 'treatment', 'score')
    # grouped_bar(means, stds, x_label='dv', y_label='score', save_path='dv_treatment_score.pdf')

    ###########
    # get means and std for 'dv' and 'party', averaging over 'score'
    # paths = [
    #     'data/passive_democrat.csv',
    #     'data/passive_republican.csv',
    #     # 'data/kalmoe_democrat.csv',
    #     'data/kalmoe_republican.csv',
    # ]
    # data = read_many_data(paths)
    # means, stds = get_means_and_stds(data, 'dv', 'party', 'score')
    # grouped_bar(means, stds, x_label='dv', y_label='score', save_path='dv_party_score.pdf')

    ###########
    # paths = [
    #     'data/kalmoe_republican.csv',
    # ]
    # data = read_many_data(paths)
    # # make a new column called 'quote' that is the 'prompt' field, split by \n\n, 1th element
    # data['quote'] = data['prompt'].apply(lambda x: x.split('\n\n')[1])

    # # # plot 'dv' againts 'score'
    # # dict1 = make_dict(data, 'dv_idx', 'score')
    # # plot_bar(dict1, y_label='Score', x_label='dv', save_path='test.pdf')
    # # geat means and std for 'quote' and 'dv_idx', averaging over 'score'
    # # means, stds = get_means_and_stds(data, 'dv_idx', 'quote', 'score')
    # means, stds = get_means_and_stds(data, 'quote', 'dv_idx', 'score')
    # grouped_bar(means, stds, x_label='dv', y_label='score', save_path='dv_quote_score.pdf')
    # breakpoint()

    # # group by 'quote' and average score
    # means = data.groupby('quote').agg({'score': 'mean'})
    # # sort by 'score'
    # means.sort_values('score', inplace=True)

    paths = [
        'data/composite_democrat.csv',
        'data/composite_republican.csv',
        'data/passive_democrat.csv',
        'data/passive_republican.csv',
        'data/mixed_affect_republican.csv',
        'data/mixed_affect_democrat.csv',
        'data/kalmoe_democrat.csv',
        'data/kalmoe_republican.csv',
    ]
    data = read_many_data(paths)

    # plot 'dv' and 'treatment' against 'score'
    means, stds = get_means_and_stds(data, 'dv', 'treatment', 'score')
    grouped_bar(means, stds, x_label='Measure', y_label='Score', save_path='measure_treatment_score.pdf')

    # plot 'category' and 'treatment' against 'score'
    means, stds = get_means_and_stds(data, 'category', 'treatment', 'score')
    grouped_bar(means, stds, x_label='DV', y_label='Score', save_path='dv_treatment_score.pdf')

    # plot 'treatment' against 'score'
    dict1 = make_dict(data, 'treatment', 'score')
    plot_bar(dict1, y_label='Score', x_label='Treatment', save_path='treatment_score.pdf')

    # plot democrats vs. republicans
    for party in ['democrat', 'republican']:
        data_party = data[data['party'] == party]
        means, stds = get_means_and_stds(data_party, 'dv', 'treatment', 'score')
        grouped_bar(means, stds, x_label='Measure', y_label='Score', title=party, save_path=f'{party}_measure_treatment_score.pdf')

        means, stds = get_means_and_stds(data_party, 'category', 'treatment', 'score')
        grouped_bar(means, stds, x_label='DV', y_label='Score', title=party, save_path=f'{party}_dv_treatment_score.pdf')

        dict1 = make_dict(data_party, 'treatment', 'score')
        plot_bar(dict1, y_label='Score', x_label='Treatment', title=party, save_path=f'{party}_treatment_score.pdf')


    pass