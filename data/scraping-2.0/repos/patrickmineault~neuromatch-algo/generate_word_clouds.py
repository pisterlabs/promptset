import itertools
import matplotlib.pyplot as plt
import numpy as np
import openai
import os
import pandas as pd
import pickle
import re

from sqlitedict import SqliteDict
from wordcloud import WordCloud

def generate_html(matches):
    html = """
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>CCN 2022 matches</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-gH2yIJqKdNHPEq0n4Mqa/HGKIhSkIHeL5AyhkYV8i59U5AR6csBvApHHNl/vI1Bx" crossorigin="anonymous">
    </head>
    <body>
    <main>
        <div class='container'>
    """

    for i, row in matches.iterrows():
        btn_template = """<li class="nav-item" role="presentation">
            <button class="nav-link {status}" id="tab-{item}-{sub}" data-bs-toggle="tab" data-bs-target="#tabpane-{item}-{sub}" type="button" role="tab" aria-controls="tabpane-{item}-{sub}" aria-selected="{selected}">Abstract {sub}</button>
        </li>"""

        def status(x):
            if x == 0:
                return "show active"
            return ""

        selected = lambda x: x == 0
        btns = '\n'.join(
            [btn_template.format(item=i, sub=x, selected=selected, status=status(x)) for x in range(len(row['all_abstracts']))]
        )
        tab_template = """<div class="tab-pane {status}" id="tabpane-{item}-{sub}" role="tabpanel" aria-labelledby="tab-{item}-{sub}" tabindex="{sub}">{content}</div>"""
        tabs = '\n'.join([tab_template.format(status=status(j), item=i, sub=j, content=x) for j, x in enumerate(row['all_abstracts'])])

        people = '\n'.join(
            f"<li>{x} ({y})</li>" for x, y in zip(row['user_names'], row['user_affiliations'])
        )

        has_common_coauthors = ""
        if row['has_indirect_coauthors'] > 0:
            has_common_coauthors = f"<div>Indirect coauthors: {', '.join(row['indirect_coauthors'])}</div>"

        panels = f"""
        <div class='px-5 py-5 my-5mx-auto"'>
        <h1>Round {row['round']}, Group {row['group']}</h1>
        <p>Score (smaller is better): <b>{row['goodness']:.3f}</b></p>
        {has_common_coauthors}
        <ul>
            {people}
        </ul>
        <ul class="nav nav-tabs" id="myTab{i}" role="tablist">
            {btns}
        </ul>
        <div class="tab-content" id="myTabContent{i}">
            {tabs}
        </div>
        <div style="margin-left:auto; margin-right: auto; ">
            <img src="output/keywords_{row["round"]}_{row["group"]}.png"  class="mx-auto d-block" style='padding-top: 50px;padding-bottom: 50px' width='600'>
        </div>
        </div>
        """

        html += "\n" + panels

    html += """
        </div>
    </main>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0/dist/js/bootstrap.bundle.min.js" integrity="sha384-A3rJD856KowSb7dwlZdYEkO39Gagi7vIsF0jrRAoQmDKKtQBHUuLZ9AsSv4jD4Xa" crossorigin="anonymous"></script>
    </body>
    </html>
    """

    with open('html/index.html', 'w') as f:
        f.write(html)

def make_single_prompt(abstract):
    # Magic incantation for GPT-3
    return ("Read this scientific abstract and describe up to 10 key phrases associated with it. "
            "Assume the reader is a computational neuroscientist. "
            "Be as specific as possible.\n\nAbstract:" + abstract + "\n\Key phrases:\n1.")

def clean_bullet_points(bp):
    bp = ("-" + bp).split('\n')
    bps = []
    for x in bp:
        x = x.strip()
        if not x:
            continue
            
        if not x.startswith('-'):
            continue
            
        x = x[1:].strip()
            
        if x.endswith('.') or x.endswith(','):
            x = x[:-1]
        x = x[0].upper() + x[1:]
        bps.append(x)
    return bps

def summarize_one_abstract(i, text):
    # First label: summarize one abstract.
    full_prompt = make_single_prompt(text)
    abstracts = SqliteDict('sample.sqlite')
    if full_prompt in abstracts:
        print("Summary cached")
        return abstracts[full_prompt]
    else:
        print("Fetching summary from OpenAI")
        openai.api_key_path = '.openai-key'
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=full_prompt,
            temperature=0.2,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        r = response['choices'][0]['text']
        abstracts[full_prompt] = r
        abstracts.commit()
        return r

def generate_top_keywords(prompt):
    # Second level: summarize many abstracts together.
    prompt = prompt + "\n\nWrite 20 key phrases relevant to these bullet points, starting with the most prominent down to the least prominent. \n\n1."

    abstracts = SqliteDict('sample.sqlite')
    if prompt in abstracts:
        print("Top keywords cached")
        return abstracts[prompt]
    
    print("Fetching top keywords from OpenAI")
    openai.api_key_path = '.openai-key'
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    r = response['choices'][0]['text']
    abstracts[prompt] = r
    abstracts.commit()
    return r

def clean_keywords(top_keywords):
    r = re.compile('[0-9]+\.')
    top_keywords = [x.strip() for x in re.split(r, top_keywords)]
    top_keywords = [x.replace('"', "").replace("'", "") for x in top_keywords]
    r = re.compile('[0-9]')
    top_keywords = [x for x in top_keywords if not r.search(x)]
    kws = []
    for kw in top_keywords:
        if kw not in kws:
            kws.append(kw)
    kws = [x for x in kws if x.lower() not in ("brain", "computational neuroscience", "big data", "machine learning")]
    return kws

def main():
    users = pd.read_pickle('data/transformed/users_w_semantic_scholar.pkl')
    df_idxmax = pd.read_pickle('data/transformed/abstract_indices.pkl')
    df_matches = pd.read_pickle('data/output/matches.pkl')


    with open('data/transformed/all_abstracts.pkl', 'rb') as f:
        all_abstracts = pickle.load(f)

    first_abstract = df_idxmax.first_abstract
    second_abstract = df_idxmax.second_abstract

    top_keywordss = []
    all_abstractss = []

    total_num = 200
    for i, row in df_matches.iterrows():
        if i >= total_num:
            break

        # Look at all the pairs
        user_ids = row['user_ids']

        # Find the indices of these users in the user data frame.
        user_index = users[users.user_id.isin(user_ids)].index

        referenced_abstracts = []
        for j, n in enumerate(user_index[:-1]):
            for n2 in user_index[j+1:]:
                # Add to the stack
                referenced_abstracts += [first_abstract.loc[(float(n), float(n2))], 
                                        second_abstract.loc[(float(n), float(n2))]]

        # Sorting is important for stability and caching.      
        referenced_abstracts = sorted(list(set(referenced_abstracts)))
        summarized_abstracts = []
        for r in referenced_abstracts:
            summarized_abstracts.append(
                ('-' + summarize_one_abstract(0, all_abstracts[r])).replace('2.', '-').replace('3.', '-')
            )

        # Now summarize the abstracts together
        all_abstractss.append([all_abstracts[r] for r in referenced_abstracts])
        concatenated_summaries = '\n'.join(summarized_abstracts)
        top_keywords = generate_top_keywords(concatenated_summaries)
        kws = clean_keywords(top_keywords)

        # And now generate a nice word cloud from this
        
        words = {x: (20 - i) for i, x in enumerate(kws)}

        def color_func(word, font_size, position, orientation, font_path, random_state):
            pos = kws.index(word)
            g = 255 - int((1 - pos / 20) * 255)
            return (g, g, g)

        plt.figure(figsize=(16, 8))
        print_scale = 2
        wordcloud = WordCloud(font_path='/usr/local/share/fonts/FiraSansCondensed-Regular.ttf',
                            colormap='Greys_r', 
                            max_font_size=160, 
                            background_color='white',
                            relative_scaling=0,
                            color_func=color_func, 
                            width=1200 * print_scale,
                            height=600 * print_scale).generate_from_frequencies(words)

        k = row['round']
        fname = f'keywords_{k}_{row["group"]}.png'
        wordcloud.to_file(fname)
        os.rename(fname, f'html/output/{fname}')
        top_keywordss.append(kws)

    total_num = len(top_keywordss)
    # And now generate the HTML for this.
    df_matches.loc[:total_num-1, 'top_keywords'] = [', '.join(x) for x in top_keywordss]
    df_matches.loc[:total_num-1, 'all_abstracts'] = all_abstractss

    name_map = {x['user_id'] : x['NameFirst'] + ' ' + x['NameLast'] for i, x in users.iterrows()}
    df_matches['user_names'] = df_matches.user_ids.map(
        lambda x: [name_map[y] for y in x]
    )
    name_map = {x['user_id'] : x['Affiliation'] for i, x in users.iterrows()}
    df_matches['user_affiliations'] = df_matches.user_ids.map(
        lambda x: [name_map[y] for y in x]
    )

    generate_html(df_matches.loc[:total_num-1])

    df_matches = df_matches.drop(columns=['all_abstracts', 'user_affiliations'])
    
    # Create smaller or larger matches, depending.
    biggest_num = max(df_matches.user_ids.map(lambda x: len(x)))

    for i in range(biggest_num + 1):
        df_matches[f'user_id_{i}'] = 0
        df_matches[f'user_name_{i}'] = ""

    for j, row in df_matches.iterrows():
        for i, (user_id, name) in enumerate(zip(row['user_ids'], row['user_names'])):
            df_matches.loc[j, f'user_id_{i}'] = int(user_id)
            df_matches.loc[j, f'user_name_{i}'] = name
    df_matches['wordcloud'] = 'keywords_' + df_matches['round'].astype(str) + '_' + df_matches.group.astype(str) + '.png'

    df_matches.to_json('data/output/matches_with_annotations.json')
    df_matches.indirect_coauthors = df_matches.indirect_coauthors.map(lambda x: ', '.join(x))
    df_matches = df_matches.drop(columns=['user_ids', 'user_names', 'has_indirect_coauthors'])
    df_matches = df_matches.rename(columns={'goodness': 'mismatch_loss'})

    df_matches = df_matches[[
        'round','group','mismatch_loss','user_id_0','user_name_0','user_id_1','user_name_1','user_id_2','user_name_2','user_id_3','user_name_3','user_id_4','user_name_4','indirect_coauthors','top_keywords','wordcloud'
    ]]

    df_matches.to_csv('data/output/matches_with_annotations.csv')

if __name__ == '__main__':
    main()
