"""
Pre-process the Yelp point of interest recommendations dataset.
"""
import argparse
import hashlib
import json, codecs
import collections
import os
import random
import re
import statistics
import sys
import time
import pandas as pd
import numpy as np
import spacy
import openai
from sklearn import neighbors
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from rank_bm25 import BM25Okapi

from . import yelppoi_prompts
from . import data_utils
from . import pre_proc_buildreps

openai.api_key = os.getenv("OPENAI_API_KEY")
spacy_model = spacy.load("en_core_web_sm")
spacy_model.add_pipe('sentencizer')


def print_dataset(in_path):
    """
    - Read the reviews, and for each user print out their reviews and
    the metadata for every place they reviewed.
    
    Filtering data based on: http://www.vldb.org/pvldb/vol10/p1010-liu.pdf
    - "We eliminate those users with fewer than 10 check-in POIs
    as well as those POIs with fewer than 10 visitors. This yields a
    dataset with 30,887 users, 18,995 POIs and 860,888 reviews"
    """
    user_file = codecs.open(os.path.join(in_path, 'raw_dataset', 'yelp_academic_dataset_user.json'), 'r', 'utf-8')
    item_file = codecs.open(os.path.join(in_path, 'raw_dataset', 'yelp_academic_dataset_business.json'), 'r', 'utf-8')
    review_file = codecs.open(os.path.join(in_path, 'raw_dataset', 'yelp_academic_dataset_review.json'), 'r', 'utf-8')
    
    # Eliminate users with fewer than 10 reviews.
    uid2user = dict()
    for i, line in enumerate(user_file):
        userd = json.loads(line.strip())
        if userd['review_count'] < 10:
            continue
        if i % 100000 == 0:
            print(f'Processing user: {i}')
        uid2user[userd['user_id']] = userd
    print('Valid users: {:}'.format(len(uid2user)))
    print(list(uid2user.keys())[:5])
    
    # Eliminate POIs with fewer than 10 reviews.
    itemid2item = dict()
    for i, line in enumerate(item_file):
        itemd = json.loads(line.strip())
        if itemd['review_count'] < 10:
            continue
        if i % 20000 == 0:
            print(f'Processing item: {i}')
        itemid2item[itemd['business_id']] = itemd
    print('Valid items: {:}'.format(len(uid2user)))
    print(list(itemid2item.keys())[:5])
    
    # Read the reviews from valid users
    uid2reviews = collections.defaultdict(list)
    review_count = 0
    for i, line in enumerate(review_file):
        reviewd = json.loads(line.strip())
        user_id = reviewd['user_id']
        item_id = reviewd['business_id']
        if i % 200000 == 0:
            print(f'Processing review: {i}')
        if user_id in uid2user and item_id in itemid2item:
            reviewd = json.loads(line.strip())
            uid2reviews[user_id].append(reviewd)
            review_count += 1
    print('Valid reviews: {:}'.format(review_count))
    
    # Print out the users reviews.
    out_file = codecs.open(os.path.join(in_path, 'user_business_reviews.txt'), 'w', 'utf-8')
    wrote_count = 0
    for uid, user_reviews in uid2reviews.items():
        if len(user_reviews) < 10:
            continue
        # Sort by rating.
        out_file.write(f'user_id: {uid}\n')
        for review in sorted(user_reviews, key=lambda d: d['stars'], reverse=True):
            itemd = itemid2item[review['business_id']]
            business_str = 'ID: {:}\nName: {:}\nCity: {:}\nState: {:}\nCategories: {:}'.\
                format(itemd['business_id'], itemd['name'], itemd['city'], itemd['state'],
                       ' - '.join(itemd['categories'].split(',')))
            out_file.write(business_str+'\n')
            review_str = 'Stars: {:}\nReview:{:}'.format(review['stars'], review['text'])
            out_file.write(review_str + '\n\n')
        out_file.write(f'===============\n\n')
        wrote_count += 1
        if wrote_count > 1000:
            break
    print(f'Wrote: {out_file.name}')
    out_file.close()
    

def write_jsons(in_path):
    """
    - Apply the filtering of: "We eliminate those users with fewer than 10 check-in POIs
        as well as those POIs with fewer than 10 visitors. This yields a
        dataset with 30,887 users, 18,995 POIs and 860,888 reviews"
    - Then write out the items in a normalized json file.
    - Write out the user reviews in a jsonl file.
    - Print stats for the dataset.
    """
    user_file = codecs.open(os.path.join(in_path, 'raw_dataset', 'yelp_academic_dataset_user.json'), 'r', 'utf-8')
    print(f'Read: {user_file.name}')
    item_file = codecs.open(os.path.join(in_path, 'raw_dataset', 'yelp_academic_dataset_business.json'), 'r', 'utf-8')
    print(f'Read: {item_file.name}')
    review_file = codecs.open(os.path.join(in_path, 'raw_dataset', 'yelp_academic_dataset_review.json'), 'r', 'utf-8')
    print(f'Read: {review_file.name}')

    # Eliminate users with fewer than 10 reviews.
    uid2user = dict()
    for i, line in enumerate(user_file):
        userd = json.loads(line.strip())
        if userd['review_count'] < 10:
            continue
        if i % 100000 == 0:
            print(f'Processing user: {i}')
        uid2user[userd['user_id']] = userd
    print('First stage filter users: {:}'.format(len(uid2user)))

    # Eliminate POIs with fewer than 10 reviews.
    itemid2item = dict()
    for i, line in enumerate(item_file):
        itemd = json.loads(line.strip())
        if itemd['review_count'] < 10:
            continue
        if i % 20000 == 0:
            print(f'Processing item: {i}')
        itemid2item[itemd['business_id']] = itemd
    print('First stage filter items: {:}'.format(len(itemid2item)))

    # Read the reviews from first stage filtered users
    uid2reviews = collections.defaultdict(list)
    review_count = 0
    for i, line in enumerate(review_file):
        reviewd = json.loads(line.strip())
        user_id = reviewd['user_id']
        item_id = reviewd['business_id']
        if i % 200000 == 0:
            print(f'Processing review: {i}')
        if user_id in uid2user and item_id in itemid2item:
            reviewd = json.loads(line.strip())
            uid2reviews[user_id].append(reviewd)
            review_count += 1
    print('Valid reviews: {:}'.format(review_count))
    
    # Now actually filter the users and businesses and minimally process them.
    itemid2item_proc = {}
    out_file = codecs.open(os.path.join(in_path, 'user_reviews.jsonl'), 'w', 'utf-8')
    total_users = 0
    total_reviews = 0
    all_review_categories = collections.Counter()
    all_review_cities = collections.Counter()
    reviews_per_user = []
    for uid, user_reviews in uid2reviews.items():
        if len(user_reviews) < 10:
            continue
        user_categories = collections.Counter()
        user_cities = collections.defaultdict(int)
        user_ratings = []
        user_reviews_proc = list()
        for review in sorted(user_reviews, key=lambda d: d['date']):
            # Save the business.
            itemd = itemid2item[review['business_id']]
            if itemd['categories']:
                item_categories = [c.strip() for c in itemd['categories'].split(',')]
            else:
                item_categories = []
            itemid2item_proc[review['business_id']] = {
                'business_id': itemd['business_id'],
                'name': itemd['name'],
                'city': itemd['city'],
                'state': itemd['state'],
                'country': 'US',  # all the items in the dataset are USA, im p sure.
                'category': item_categories
            }
            user_categories.update(item_categories)
            user_cities[itemd['city']] += 1
            # Save the user reviews to a jsonl file.
            rsentences = spacy_model(review['text'],
                                     disable=['tok2vec', 'tagger', 'attribute_ruler',
                                              'lemmatizer', 'parser', 'ner'])
            rsentences = [sent.text.strip() for sent in rsentences.sents]
            reviewd = {
                'review_id': review['review_id'],
                'user_id': review['user_id'],
                'business_id': review['business_id'],
                'date': review['date'],
                'review': rsentences,
                'rating': review['stars']
            }
            user_ratings.append(review['stars'])
            user_reviews_proc.append(reviewd)
            total_reviews += 1
        if total_users % 10000 == 0:
            print('Wrote: {:d}'.format(total_users))
        reviews_per_user.append(len(user_ratings))
        average_rating = statistics.mean(user_ratings)
        all_review_categories.update(user_categories)
        all_review_cities.update(user_cities)
        above_average_ratings = sum([1 for r in user_ratings if r >= average_rating])
        userd = {
            'user_id': uid,
            'average_rating': average_rating,
            'above_average_ratings': above_average_ratings,
            'num_ratings': len(user_reviews_proc),
            'user_categories': sorted(user_categories.items(), key=lambda t: t[1], reverse=True),
            'user_cities': sorted(user_cities.items(), key=lambda t: t[1], reverse=True),
            'reviews': user_reviews_proc
        }
        out_file.write(json.dumps(userd)+'\n')
        total_users += 1
    total_items = len(itemid2item_proc)
    print('Total items: {:d}; Total users: {:d}'.format(total_items, total_users))
    sparsity = 1-float(total_reviews)/(total_users*total_items)
    print('Total reviews: {:d}; Sparsity: {:.4f}'.format(total_reviews, sparsity))
    print('Reviews per user:\n')
    print(pd.DataFrame(reviews_per_user).describe())
    print(f'Wrote: {out_file.name}')
    out_file.close()
    with codecs.open(os.path.join(in_path, 'itemid2items.json'), 'w', 'utf-8') as fp:
        json.dump(itemid2item_proc, fp)
        print(f'Wrote: {fp.name}')
    with codecs.open(os.path.join(in_path, 'review_cities.txt'), 'w', 'utf-8') as fp:
        for city in sorted(all_review_cities, key=all_review_cities.get, reverse=True):
            fp.write('{:s}, {:d}\n'.format(city, all_review_cities[city]))
    with codecs.open(os.path.join(in_path, 'review_categories.txt'), 'w', 'utf-8') as fp:
        for category in sorted(all_review_categories, key=all_review_categories.get, reverse=True):
            fp.write('{:s}, {:d}\n'.format(category, all_review_categories[category]))
            

def select_users_dataaug(in_path):
    """
    Select the candidate users for who we will generate narrative queries with GPT-3.
    """
    # Read in the items.
    with codecs.open(os.path.join(in_path, 'itemid2items.json'), 'r', 'utf-8') as fp:
        itemid2item = json.load(fp)
        print(f'Read: {fp.name}')
    
    # Read the users.
    user_reviews = codecs.open(os.path.join(in_path, 'user_reviews.jsonl'), 'r', 'utf-8')
    user_reviews_dataaug = codecs.open(os.path.join(in_path, 'user_items_reviews-dataaug.jsonl'), 'w', 'utf-8')
    total_items = set()
    total_users = 0
    total_reviews = 0
    all_review_categories = collections.Counter()
    all_review_cities = collections.Counter()
    reviews_per_user = []
    for line in user_reviews:
        userd = json.loads(line.strip())
        # Retain users who have rated things because they like them
        # and make sure they have atleast 10 reviews they generally liked.
        # Skip users with more than 30 positive reviews because we prob cant
        # represent their interests with the narrative query.
        if userd['average_rating'] > 3.0 and (10 <= userd['above_average_ratings'] <= 30):
            retained_reviews = []
            user_cities = set()
            user_categories = set()
            for review in userd['reviews']:
                if review['rating'] >= userd['average_rating']:
                    itemid = review['business_id']
                    total_items.add(itemid)
                    review['name'] = itemid2item[itemid]['name']
                    review['city'] = itemid2item[itemid]['city']
                    review['state'] = itemid2item[itemid]['state']
                    review['country'] = itemid2item[itemid]['country']
                    review['category'] = itemid2item[itemid]['category']
                    retained_reviews.append(review)
                    user_cities.add(itemid2item[itemid]['city'])
                    user_categories.update(itemid2item[itemid]['category'])
                    total_reviews += 1
            reviews_per_user.append(len(retained_reviews))
            all_review_categories.update(user_categories)
            all_review_cities.update(user_cities)
            total_users += 1
            out_userd = {
                'user_id': userd['user_id'],
                'user_categories': list(user_categories),
                'user_cities': list(user_cities),
                'reviews': retained_reviews
            }
            user_reviews_dataaug.write(json.dumps(out_userd)+'\n')
    print('Total items: {:d}; Total users: {:d}'.format(len(total_items), total_users))
    print('Total reviews: {:d}'.format(total_reviews))
    print('Reviews per user:')
    print(pd.DataFrame(reviews_per_user).describe())
    print(f'Wrote: {user_reviews_dataaug.name}')
    user_reviews_dataaug.close()
    with codecs.open(os.path.join(in_path, 'user_items_reviews-dataaug_cities.txt'), 'w', 'utf-8') as fp:
        for city in sorted(all_review_cities, key=all_review_cities.get, reverse=True):
            fp.write('{:s}, {:d}\n'.format(city, all_review_cities[city]))
    with codecs.open(os.path.join(in_path, 'user_items_reviews-dataaug_categories.txt'), 'w', 'utf-8') as fp:
        for category in sorted(all_review_categories, key=all_review_categories.get, reverse=True):
            fp.write('{:s}, {:d}\n'.format(category, all_review_categories[category]))


def sample_review_sent(review_sents, to_sample, exclude=None):
    """
    :param review_sents: list(string); sentences in a review
    :param to_sample: int; how many to sample
    :param exclude: string; sentence to exclude from review_sents in sampling.
    """
    # Truncate very long sentences and retain the sentences which
    # are longer than 5 tokens.
    truncated_sents = []
    for si, s in enumerate(review_sents):
        if exclude and s == exclude:
            continue
        # A token needs to be at most 30 chars long.
        toks = [t for t in s.split() if len(t) < 30]
        if len(toks) > 20 or len(toks) < 5:
            continue
        else:
            truncated_sents.append(' '.join(toks))
        # Include at most the first 5 sentences.
        if len(truncated_sents) == 5:
            break
    if len(truncated_sents) >= 1:
        # Select a single random sentence from the review to include.
        review_sent = random.sample(truncated_sents, min(to_sample, len(truncated_sents)))
    else:
        # Include a random sentence up to 90 chars long if you fail to get a sentence.
        review_sent = random.sample(review_sents, min(to_sample, len(review_sents)))
        review_sent = review_sent[:90]
    return review_sent


def write_dataaug_examples(in_path, item_examples=False):
    """
    Given the users for who we should generate GPT3 data augmentations,
    generate the data to be used in prompts for GPT3 to complete.
    - Create the items string for 10 randomly choosen items.
    -- Filter out the items which are too far out from what the pointrec dataset looks for.
    - Sample the 2-3 sentences from their review and include it in the prompt.
    """
    if item_examples:
        out_fname = os.path.join(in_path, 'user_items_reviews-item_prompt_content.jsonl')
        outf_readable = os.path.join(in_path, 'user_items_reviews-item_prompt_content.txt')
        items_to_sample = 1
        review_sents_to_sample = 3
    else:
        out_fname = os.path.join(in_path, 'user_items_reviews-short_prompt_content.jsonl')
        outf_readable = os.path.join(in_path, 'user_items_reviews-short_prompt_content.txt')
        items_to_sample = 10
        review_sents_to_sample = 1
    # Blocklist is created manually by looking at 'user_items_reviews-dataaug_categories.txt'
    # Expand it if necessary.
    with codecs.open(os.path.join(in_path, 'user_items_reviews-dataaug_blocklist.txt'), 'r', 'utf-8') as fp:
        block_categories = set()
        for line in fp:
            block_categories.add(line.strip())
    user_item_reviews = codecs.open(os.path.join(in_path, 'user_items_reviews-dataaug.jsonl'), 'r', 'utf-8')
    print(f'Read: {user_item_reviews.name}')
    
    user_item_prompts = codecs.open(out_fname, 'w', 'utf-8')
    user_item_prompts_read = codecs.open(outf_readable, 'w', 'utf-8')
    prompt_length = []
    reviews_per_user = []
    total_examples = 0
    total_users = 0
    total_items = set()
    for line in user_item_reviews:
        userd = json.loads(line.strip())
        # Gather the reviews which are for items in valid categories.
        valid_itemidxs = []
        for i, review in enumerate(userd['reviews']):
            if len(set.intersection(set(review['category']), block_categories)) == 0:
                valid_itemidxs.append(i)
        if len(valid_itemidxs) < 5:
            continue
        reviews_per_user.append(len(valid_itemidxs))
        # Sample 10 if there are too many, else keep all the items.
        sampled_idxs = random.sample(valid_itemidxs, min(items_to_sample, len(valid_itemidxs)))
        sampled_itemids = []
        sampled_reviewids = []
        sampled_sents = []
        sampled_names = []
        for i in sampled_idxs:
            reviewd = userd['reviews'][i]
            sampled_itemids.append(reviewd['business_id'])
            sampled_reviewids.append(reviewd['review_id'])
            review_sent = sample_review_sent(review_sents=reviewd['review'], to_sample=review_sents_to_sample)
            sampled_sents.extend(review_sent)
            sampled_names.append('{:s} in {:s}'.format(reviewd['name'], reviewd['city']))
        if item_examples:
            assert (len(sampled_itemids) == len(sampled_reviewids) == len(sampled_names))
        else:
            assert(len(sampled_itemids) == len(sampled_reviewids) == len(sampled_sents) == len(sampled_names))
        total_toks = len(' '.join(sampled_sents).split())
        if total_toks > 200:
            if item_examples:  # skip the user if item is too long.
                continue
            else:  # If the number of review sents is too long then trim the user.
                num_items = len(sampled_itemids)
                if num_items > 5:
                    sampled_itemids = sampled_itemids[:num_items//2]
                    sampled_reviewids = sampled_reviewids[:num_items//2]
                    sampled_sents = sampled_sents[:num_items//2]
                    sampled_names = sampled_names[:num_items//2]
                else:
                    continue
        total_examples += len(sampled_itemids)
        total_users += 1
        prompt_length.append(len(' '.join(sampled_sents).split()))
        userd_out = {
            'review_sents': sampled_sents,
            'item_names': sampled_names,
            'business_ids': sampled_itemids,
            'review_ids': sampled_reviewids,
            'user_id': userd['user_id']
        }
        total_items.update(sampled_itemids)
        user_item_prompts.write(json.dumps(userd_out)+'\n')
        user_item_prompts_read.write('A user likes the following recommendations: {:s}\n\n'
                                     .format(', '.join(sampled_names)))
        user_item_prompts_read.write('The user wrote the following reviews: {:s}\n\n'
                                     .format(' '.join(sampled_sents)))
        user_item_prompts_read.write('==========\n')
    print(f'Read: {user_item_prompts.name}')
    user_item_prompts.close()
    user_item_prompts_read.close()
    print('Items per user post category filter:')
    print(pd.DataFrame(reviews_per_user).describe())
    print('Review sentence lengths:')
    print(pd.DataFrame(prompt_length).describe())
    print(f'Total examples: {total_examples}; Total users: {total_users}; Uniq items: {len(total_items)}')


def get_gpt3_narrativeqs(in_path, out_path, item_examples=False):
    """
    - Read the prompt content.
    - Make an API call to GPT3 and get the generated narrative query.
    - Write out the GPT3 generation.
    """
    if item_examples:
        start_count = 0
        to_write = 43782
        completion_length = 130
        gpt3_model_name = "text-curie-001"
        in_fname = os.path.join(in_path, 'user_items_reviews-item_prompt_content.jsonl')
    else:
        start_count = 0
        to_write = 10441
        completion_length = 130
        gpt3_model_name = "text-curie-001"
        in_fname = os.path.join(in_path, 'user_items_reviews-short_prompt_content.jsonl')
        
    # Read the prompt content.
    user_prompt_content = codecs.open(in_fname, 'r', 'utf-8')
    print(f'Read: {user_prompt_content.name}')
    
    # Go over the users and write out the outputs.
    out_file = codecs.open(os.path.join(out_path, f'user_items_reviews-{gpt3_model_name}.jsonl'), 'w', 'utf-8')
    out_file_read = codecs.open(os.path.join(out_path, f'user_items_reviews-{gpt3_model_name}.txt'), 'w', 'utf-8')
    if item_examples:
        labeled_prompts = [yelppoi_prompts.itemlabeled1_poi, yelppoi_prompts.itemlabeled2_poi, yelppoi_prompts.itemlabeled3_poi]
    else:
        labeled_prompts = [yelppoi_prompts.labeled1_poi, yelppoi_prompts.labeled2_poi, yelppoi_prompts.labeled3_poi]
    labeled_prompt = '\n\n'.join(labeled_prompts)
    out_file_read.write(labeled_prompt)
    out_file_read.write('\n++++++++++\n\n')
    processed_users = 0
    print(f'Calling API starting at line number: {start_count}')
    for line in user_prompt_content:
        # If a previous run failed allow starting from a arbitrary new point.
        # so examined examples are not re-rexamined.
        if processed_users < start_count:
            processed_users += 1
            continue
        userd = json.loads(line.strip())
        liked_names = userd['item_names']
        liked_review_sents = userd['review_sents']
        test_case = "A user likes the following recommendations: {:s}\n\n" \
                    "The user wrote the following reviews: {:s}\n\n" \
                    "In response to the request on Reddit:".\
            format(', '.join(liked_names), ' '.join(liked_review_sents))
        shuf_idxs = list(range(3))
        random.shuffle(shuf_idxs)
        shuf_prompts = '\n\n'.join(labeled_prompts[i] for i in shuf_idxs)
        prompt = shuf_prompts + test_case
        try:
            response = openai.Completion.create(
                model=gpt3_model_name,
                prompt=prompt,
                temperature=0.7,
                max_tokens=completion_length,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
        except Exception:
            print('Sleeping: 10s')
            time.sleep(10)
            response = openai.Completion.create(
                model=gpt3_model_name,
                prompt=prompt,
                temperature=0.7,
                max_tokens=completion_length,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
        userd['llm_response'] = response
        userd['prompt_order'] = shuf_idxs
        out_file.write(json.dumps(userd)+'\n')
        generated_text = response['choices'][0]['text'].strip()
        out_file_read.write(test_case + generated_text + '\n')
        out_file_read.write('===========\n\n')
        processed_users += 1
        if processed_users % 10 == 0:
            print('Users processed: {:d}'.format(processed_users))
        if processed_users >= to_write:
            break
    print(f'Wrote users: {processed_users}')
    print(f'Wrote: {out_file.name}')
    out_file.close()
    out_file_read.close()


def _read_gpt3_runs_user_data(data_path, llm_model_name, gpt3_run_names):
    """
    - Read the data generated by GPT3 from disk.
    - Read user item reviews.
    """
    uid2narrative_queries = collections.defaultdict(list)
    example_count = 0
    for run_name in gpt3_run_names:
        print('Read: {:s}'.format(run_name))
        q_file = codecs.open(os.path.join(data_path, 'get_gpt3_outs',
                                          run_name, f'user_items_reviews-text-{llm_model_name}.jsonl'), 'r', 'utf-8')
        for qline in q_file:
            qd = json.loads(qline.strip())
            uid = qd['user_id']
            narrativeq = qd['llm_response']['choices'][0]['text'].strip()
            for sent, itemid in zip(qd['review_sents'], qd['business_ids']):
                uid2narrative_queries[uid].append({
                    'query': narrativeq,
                    'review_sent': sent,
                    'business_id': itemid,
                    'gpt3_run_name': run_name
                })
                example_count += 1
    print(f'Users: {len(uid2narrative_queries)}')
    print(f'Examples: {example_count}')
    
    user_item_reviews = {}
    bid2reviews = collections.defaultdict(list)
    with codecs.open(os.path.join(data_path, 'user_items_reviews-dataaug.jsonl'), 'r', 'utf-8') as fp:
        for uline in fp:
            userd = json.loads(uline.strip())
            user_id = userd['user_id']
            user_item_reviews[user_id] = userd
            for r in userd['reviews']:
                bid2reviews[r['business_id']].append(r)
        print(f'Read: {fp.name}')
    bid2reviews = dict(bid2reviews)
    
    # uid2narrative_queries = dict(list(uid2narrative_queries.items())[:20])
    
    return uid2narrative_queries, example_count, user_item_reviews, bid2reviews


def _get_user_cats_cities(userd):
    """
    Get category and city for the user.
    """
    # Get the main category as the more frequent category.
    user_city2count = collections.Counter([r['city'] for r in userd['reviews']]).items()
    user_city2count = sorted(user_city2count, key=lambda t: t[1], reverse=True)
    user_cities = [t[0] for t in user_city2count]
    user_cat2count = collections.Counter()
    for r in userd['reviews']:
        user_cat2count.update(r['category'])
    user_cat2count = user_cat2count.items()
    user_cat2count = sorted(user_cat2count, key=lambda t: t[1], reverse=True)
    user_cats = [t[0] for t in user_cat2count]
    user_cat_counts = [t[1] for t in user_cat2count]
    return user_cities, user_cats, user_cat_counts


def _drop_poi_text(ititle, snippets, rdrop):
    """
    With a passed probability drop the description and return a result.
    """
    if rdrop > 0:
        choice = random.choice(range(int(1 / rdrop)))
        if choice == 0:  # choice will be zero with chance rdrop.
            poi_text = ititle
        else:
            poi_text = ititle + ' ' + ' '.join(snippets)
    else:
        poi_text = ititle + ' ' + ' '.join(snippets)
    return poi_text


def _build_poi_text(dataaug_dict, bid2userreviews, user_cats, user_cat_counts):
    """
    Build the poi text to be used for training (and filtering)
    """
    itemid = dataaug_dict['business_id']
    itemreviewd = bid2userreviews[itemid]
    item_cats = [cat for cat, count in zip(user_cats, user_cat_counts)
                 if cat in set(itemreviewd['category'])]
    item_cat_counts = [count for cat, count in zip(user_cats, user_cat_counts)
                       if cat in set(itemreviewd['category'])]
    icity = itemreviewd['city']
    iname = itemreviewd['name']
    try:
        imaincat = random.choices(item_cats, weights=item_cat_counts, k=1)[0]
    except IndexError:
        imaincat = ''
    try:
        isubcat = random.choices(item_cats, weights=[1 / w for w in item_cat_counts], k=1)[0]
    except IndexError:
        isubcat = ''
    if imaincat and isubcat:
        ititle = f'{iname} in {icity}, US is known for {imaincat}. Specifically: {isubcat}.'
    elif imaincat:
        ititle = f'{iname} in {icity}, US is known for {imaincat}.'
    elif isubcat:
        ititle = f'{iname} in {icity}, US is known for {isubcat}.'
    else:
        ititle = f'{iname} in {icity}, US.'
    # Sample some review sentences and include what was used in the LLM prompt.
    snippets = sample_review_sent(review_sents=itemreviewd['review'],
                                  to_sample=2, exclude=dataaug_dict['review_sent'])
    snippets.append(dataaug_dict['review_sent'])
    return ititle, snippets


def filter_create_pairdoc_examples(data_path, out_path, filter_model, filter_frac, rdrop,
                                   llm_model_name):
    """
    Given the queries authored by an LLM exclude some of the item-query
    pairs.
    - filter_methods can be: random or query-likelihood (with various pretrained LMs)
    """
    assert (0 <= rdrop <= 1)
    assert (0 < filter_frac < 1)
    if llm_model_name == 'curie-001':
        fname_prefix = 'curie-10k'
        gpt3_run_names = ['2023_04_23-13_49_53-10kcurie']
    elif llm_model_name == 'davinci-003':
        fname_prefix = 'davinci-10k'
        gpt3_run_names = ['2023_01_19-07_06_47-19kfirst', '2023_01_19-15_14_32-19krestart100',
                          '2023_01_19-20_46_35-2k_shufshortp']  # '2023_03_30-20_15_12-full8k']
    else:
        raise ValueError(f'Unknown llm_model_name: {llm_model_name}')
    name2model = {
        'qlft5l': "google/flan-t5-large",
        'qlft5xl': "google/flan-t5-xl",
        'qlft5xxl': "google/flan-t5-xxl"
    }
    # Read the examples generated with GPT3 and user-item-review data.
    uid2narrative_queries, example_count, user_item_reviews, bid2reviews = \
        _read_gpt3_runs_user_data(data_path, llm_model_name=llm_model_name,
                                  gpt3_run_names=gpt3_run_names)
    
    # Load model and tokenizer
    if filter_model in name2model:
        tokenizer = T5Tokenizer.from_pretrained(name2model[filter_model])
        model = T5ForConditionalGeneration.from_pretrained(name2model[filter_model], device_map="auto",
                                                           cache_dir='/gypsum/work1/mccallum/smysore/tmp')
    print('Filtering with: {:}; filter frac: {:}'.format(filter_model, filter_frac))
    
    # Go over the generated data and add a score to them.
    for user_id, narrative_qs in uid2narrative_queries.items():
        userd = user_item_reviews[user_id]
        user_cities, user_cats, user_cat_counts = _get_user_cats_cities(userd)
        bid2userreviews = dict([(r['business_id'], r) for r in userd['reviews']])
        # Build the query.
        qmaincat = user_cats[0]
        qcity = user_cities[0]
        qtitle = f'Looking for {qmaincat} in {qcity}, US.'
        query_text = qtitle + ' ' + narrative_qs[0]['query']
        poi_texts = []
        for dataaug_dict in narrative_qs:
            ititle, snippets = _build_poi_text(dataaug_dict, bid2userreviews, user_cats, user_cat_counts)
            poi_texts.append(ititle + ' ' + ' '.join(snippets))
            dataaug_dict['query_text'] = query_text
            dataaug_dict['poi_text'] = (ititle, snippets)
        if filter_model in name2model:
            poi_neg_scores = pre_proc_buildreps.ql_score_with_t5(model, tokenizer, query_text, poi_texts)
        elif filter_model == 'rand':
            poi_neg_scores = np.random.randint(0, 1000, len(poi_texts)).tolist()
        else:
            raise ValueError('Unknown filter_model: {:}'.format(filter_model))
        assert(len(poi_neg_scores) == len(narrative_qs))
        for dataaug_dict, score in zip(narrative_qs, poi_neg_scores):
            dataaug_dict['filt_neg_score'] = score
    
    # Create examples.
    # Create dev and train splits.
    user_ids = list(uid2narrative_queries.keys())
    user_ids.sort()
    random.shuffle(user_ids)
    random.shuffle(user_ids)
    train_count = int(len(user_ids) * 0.80)
    train_uids, dev_uids = set(user_ids[:train_count]), set(user_ids[train_count:])
    print(f'Dev users: {len(dev_uids)}; Train users: {len(train_uids)}')
    
    if rdrop > 0:
        out_train_file = codecs.open(os.path.join(out_path, 'train-{:}-rdrop{:}-{:}{:}.jsonl'.
                                                  format(fname_prefix, rdrop, filter_model, filter_frac)), 'w', 'utf-8')
        out_dev_file = codecs.open(os.path.join(out_path, 'dev-{:}-rdrop{:}-{:}{:}.jsonl'.
                                                format(fname_prefix, rdrop, filter_model, filter_frac)), 'w', 'utf-8')
    else:
        out_train_file = codecs.open(os.path.join(out_path, 'train-{:}-{:}{:}.jsonl'.
                                                  format(fname_prefix, filter_model, filter_frac)), 'w', 'utf-8')
        out_dev_file = codecs.open(os.path.join(out_path, 'dev-{:}-{:}{:}.jsonl'.
                                                format(fname_prefix, filter_model, filter_frac)), 'w', 'utf-8')
    
    examples_per_user = []
    for user_id in user_item_reviews:
        if (user_id not in train_uids) and (user_id not in dev_uids):
            continue
        is_dev = True if user_id in dev_uids else False
        # Both train and dev need a query-positive.
        narrative_qs = uid2narrative_queries[user_id]
        ex_to_keep = int(len(narrative_qs)*(1-filter_frac))
        ex_written = 0
        # Smallest neg_score at the top.
        for dataaug_dict in sorted(narrative_qs, key=lambda d: d['filt_neg_score'], reverse=False):
            if ex_written > ex_to_keep:
                break
            # Build the query.
            query_text = dataaug_dict['query_text']
            query = {'TITLE': query_text, 'ABSTRACT': []}  # place the title and request together to mimic test time.
            ititle, snippets = dataaug_dict['poi_text']
            poi_text = _drop_poi_text(ititle, snippets, rdrop)
            pos_poi = {'TITLE': poi_text, 'ABSTRACT': []}  # place the title and request together to mimic test time.
            out_ex = {
                'user_id': user_id,
                'cited_pids': [user_id, dataaug_dict['business_id']],
                'gpt3_run_name': dataaug_dict['gpt3_run_name'],
                'query': query,
                'pos_context': pos_poi
            }
            if is_dev:
                neg_bid = random.sample(list(bid2reviews.keys()), 1)[0]
                neg_reviewd = random.sample(bid2reviews[neg_bid], 1)[0]
                try:
                    neg_cat = neg_reviewd['category'][0]
                except IndexError:
                    neg_cat = 'Unknown'
                try:
                    neg_subcat = random.sample(neg_reviewd['category'], k=1)[0]
                except ValueError:
                    neg_subcat = 'Unknown'
                neg_city = neg_reviewd['city']
                neg_name = neg_reviewd['name']
                neg_title = f'{neg_name} in {neg_city}, US is known for {neg_cat}. Specifically: {neg_subcat}.'
                # Sample some review sentences and include what was used in the LLM prompt.
                neg_snippets = sample_review_sent(review_sents=neg_reviewd['review'],
                                                  to_sample=3)
                neg_poi_text = _drop_poi_text(neg_title, neg_snippets, rdrop)
                neg_poi = {'TITLE': neg_poi_text,
                           'ABSTRACT': []}  # place the title and request together to mimic test time.
                out_ex['neg_context'] = neg_poi
                out_dev_file.write(json.dumps(out_ex) + '\n')
            else:
                out_train_file.write(json.dumps(out_ex) + '\n')
            ex_written += 1
        examples_per_user.append(ex_written)
    
    print(f'Wrote: {out_train_file.name}')
    out_train_file.close()
    all_summ = pd.DataFrame(examples_per_user).describe()
    print('Examples per user: {:}'.format(all_summ))
    total_examples = sum(examples_per_user)
    print(f'Number of examples: {total_examples}')


def create_pairdoc_examples(data_path, out_path, dataset, model_name, rdrop, llm_model_name):
    """
    Given the query generations from GPT3 generate paired query-POI pairs
    for training a biencoder.
    - gpt3_run_names basenames for the gpt3 output directories.
    :param data_path: string; the path to load the user-items-reviews data from.
    :param out_path: string; where output jsonls will get written.
    :param rdrop: float; probability of dropping the review for an item.
    """
    assert(0 <= rdrop <= 1)
    if llm_model_name == 'curie-001':
        fname_prefix = 'curie-43k'
        gpt3_run_names = ['2023_04_15-16_09_18-itemex43k']
    elif llm_model_name == 'davinci-003':
        fname_prefix = 'davinci-10k'
        gpt3_run_names = ['2023_01_19-07_06_47-19kfirst', '2023_01_19-15_14_32-19krestart100',
                          '2023_01_19-20_46_35-2k_shufshortp']  # '2023_03_30-20_15_12-full8k']
    else:
        raise ValueError(f'Unknown llm_model_name: {llm_model_name}')
    # Read the examples generated with GPT3 and user-item-review data.
    uid2narrative_queries, example_count, user_item_reviews, bid2reviews = \
        _read_gpt3_runs_user_data(data_path, llm_model_name=llm_model_name,
                                  gpt3_run_names=gpt3_run_names)
    
    # Create dev and train splits.
    user_ids = list(uid2narrative_queries.keys())
    user_ids.sort()
    random.shuffle(user_ids)
    random.shuffle(user_ids)
    train_count = int(len(user_ids)*0.92)
    train_uids, dev_uids = set(user_ids[:train_count]), set(user_ids[train_count:])
    print(f'Dev users: {len(dev_uids)}; Train users: {len(train_uids)}')

    if rdrop > 0:
        out_train_file = codecs.open(os.path.join(out_path, 'train-{:s}rdrop{:}.jsonl'.
                                                  format(fname_prefix, rdrop)), 'w', 'utf-8')
        out_dev_file = codecs.open(os.path.join(out_path, 'dev-{:s}rdrop{:}.jsonl'.
                                                format(fname_prefix, rdrop)), 'w', 'utf-8')
    else:
        out_train_file = codecs.open(os.path.join(out_path, f'train-{fname_prefix}.jsonl'), 'w', 'utf-8')
        out_dev_file = codecs.open(os.path.join(out_path, f'dev-{fname_prefix}.jsonl'), 'w', 'utf-8')
    
    examples_per_user = []
    for user_id in user_item_reviews:
        userd = user_item_reviews[user_id]
        if (user_id not in train_uids) and (user_id not in dev_uids):
            continue
        is_dev = True if user_id in dev_uids else False
        # Both train and dev need a query-positive.
        # Gather user city and categories for making the query title.
        user_cities, user_cats, user_cat_counts = _get_user_cats_cities(userd)
        # Go over the reviews used for query construction and build a example.
        bid2userreviews = dict([(r['business_id'], r) for r in userd['reviews']])
        narrative_qs = uid2narrative_queries[user_id]
        for dataaug_dict in narrative_qs:
            # Build the query.
            qmaincat = user_cats[0]
            qcity = user_cities[0]
            qtitle = f'Looking for {qmaincat} in {qcity}, US.'
            query_text = qtitle + ' ' + dataaug_dict['query']
            query = {'TITLE': query_text, 'ABSTRACT': []}  # place the title and request together to mimic test time.
            # Build the POI.
            ititle, snippets = _build_poi_text(dataaug_dict, bid2userreviews, user_cats, user_cat_counts)
            # Drop some of the review snippets
            poi_text = _drop_poi_text(ititle, snippets, rdrop)
            pos_poi = {'TITLE': poi_text, 'ABSTRACT': []}  # place the title and request together to mimic test time.
            out_ex = {
                'user_id': user_id,
                'cited_pids': [user_id, dataaug_dict['business_id']],
                'gpt3_run_name': dataaug_dict['gpt3_run_name'],
                'query': query,
                'pos_context': pos_poi
            }
            if is_dev:
                neg_bid = random.sample(list(bid2reviews.keys()), 1)[0]
                neg_reviewd = random.sample(bid2reviews[neg_bid], 1)[0]
                try:
                    neg_cat = neg_reviewd['category'][0]
                except IndexError:
                    neg_cat = 'Unknown'
                try:
                    neg_subcat = random.sample(neg_reviewd['category'], k=1)[0]
                except ValueError:
                    neg_subcat = 'Unknown'
                neg_city = neg_reviewd['city']
                neg_name = neg_reviewd['name']
                neg_title = f'{neg_name} in {neg_city}, US is known for {neg_cat}. Specifically: {neg_subcat}.'
                # Sample some review sentences and include what was used in the LLM prompt.
                neg_snippets = sample_review_sent(review_sents=neg_reviewd['review'],
                                                  to_sample=3)
                neg_poi_text = _drop_poi_text(neg_title, neg_snippets, rdrop)
                neg_poi = {'TITLE': neg_poi_text, 'ABSTRACT': []}  # place the title and request together to mimic test time.
                out_ex['neg_context'] = neg_poi
                out_dev_file.write(json.dumps(out_ex) + '\n')
            else:
                out_train_file.write(json.dumps(out_ex) + '\n')
        examples_per_user.append(len(narrative_qs))
            
    print(f'Wrote: {out_train_file.name}')
    out_train_file.close()
    all_summ = pd.DataFrame(examples_per_user).describe()
    print('Examples per user: {:}'.format(all_summ))
    total_examples = sum(examples_per_user)
    print(f'Number of examples: {total_examples}')


def create_pairdoc_examples_crossenc(in_path, out_path, filter_model, filter_frac,
                                     llm_model_name, neg_model, trained_neg_runpath):
    """
    - Given a set of positive examples generated from LLM outputs for biencoder training
        construct training examples for a cross-encoder.
    - Negatives for the cross-encoder are generated using the biencoder trained on the data.
    """
    train_neg_per_pos = 4
    dev_negs = 200
    if llm_model_name == 'curie-001':
        fname_prefix = 'curie-10k'
    elif llm_model_name == 'curie-001-itemqs':
        fname_prefix = 'curie-43k'
        assert(filter_frac == None and filter_model == None)
    elif llm_model_name == 'davinci-003':
        fname_prefix = 'davinci-10k'
    else:
        raise ValueError(f'Unknown llm_model_name: {llm_model_name}')
    
    # Read the dev and test splits and embed the data for generating negatives.
    if filter_model and filter_frac:
        in_train_file = codecs.open(os.path.join(in_path, 'train-{:}-{:}{:}.jsonl'.
                                                 format(fname_prefix, filter_model, filter_frac)), 'r', 'utf-8')
        in_dev_file = codecs.open(os.path.join(in_path, 'dev-{:}-{:}{:}.jsonl'.
                                               format(fname_prefix, filter_model, filter_frac)), 'r', 'utf-8')
        out_train_file = codecs.open(os.path.join(
            out_path, 'ce-train-{:}-{:}{:}-{:}.jsonl'.format(fname_prefix, filter_model, filter_frac, neg_model)),
            'w', 'utf-8')
        out_dev_file = codecs.open(os.path.join(
            out_path, 'ce-dev-{:}-{:}{:}-{:}.jsonl'.format(fname_prefix, filter_model, filter_frac, neg_model)),
            'w', 'utf-8')
    else:
        in_train_file = codecs.open(os.path.join(in_path, 'train-{:}.jsonl'.format(fname_prefix)), 'r', 'utf-8')
        in_dev_file = codecs.open(os.path.join(in_path, 'dev-{:}.jsonl'.format(fname_prefix)), 'r', 'utf-8')
        out_train_file = codecs.open(os.path.join(out_path, 'ce-train-{:}-{:}.jsonl'.format(fname_prefix, neg_model)),
                                     'w', 'utf-8')
        out_dev_file = codecs.open(os.path.join(out_path, 'ce-dev-{:}-{:}.jsonl'.format(fname_prefix, neg_model)),
                                   'w', 'utf-8')
    print(f'Read: {in_train_file.name}')
    print(f'Read: {in_dev_file.name}')
    all_uid2query = collections.OrderedDict()
    all_bid2doc = collections.OrderedDict()
    train_uid2posbids = collections.defaultdict(list)
    dev_uid2posbids = collections.defaultdict(list)
    for split, file in [('train', in_train_file), ('dev', in_dev_file)]:
        for line in file:
            exdict = json.loads(line.strip())
            uid = exdict['user_id']
            pos_bid = exdict['cited_pids'][1]
            all_uid2query[uid] = exdict['query']['TITLE']
            all_bid2doc[pos_bid] = exdict['pos_context']['TITLE']
            if split == 'dev':
                dev_uid2posbids[uid].append(pos_bid)
                # The dev file doesnt have the bid for the negs so create one. facepalm.
                neg_id = hashlib.md5()
                neg_id.update(exdict['neg_context']['TITLE'].encode('utf-8'))
                neg_bid = neg_id.hexdigest()
                all_bid2doc[neg_bid] = exdict['neg_context']['TITLE']
            else:
                train_uid2posbids[uid].append(pos_bid)
    print(f'Users: {len(all_uid2query)}; Docs: {len(all_bid2doc)}')
    
    # Go over documents and form sb reps for documents.
    print(f'Negatives model: {neg_model}')
    n_neighbors = 300
    if neg_model in {'recinpars', 'sbcontriver'}:
        sb_model, sim_fun = pre_proc_buildreps.init_sbmodels(model_name=neg_model,
                                                             trained_model_path=trained_neg_runpath)
        query_vectors = sb_model.encode(list(all_uid2query.values()), show_progress_bar=True, batch_size=64)
        print('Encoded queries: {:}'.format(query_vectors.shape))
        document_vectors = sb_model.encode(list(all_bid2doc.values()), show_progress_bar=True, batch_size=64)
        print('Encoded documents: {:}'.format(document_vectors.shape))
        nn_index = neighbors.NearestNeighbors(n_neighbors=n_neighbors)
        nn_index.fit(document_vectors)
        query2neighbor_idxs = nn_index.kneighbors(X=query_vectors, return_distance=False)
    elif neg_model == 'okapibm25':
        cleaned_docs = [re.sub('[^a-zA-Z0-9 \n\.]', '', s) for s in list(all_bid2doc.values())]
        tokenized_corpus = [doc.lower().split() for doc in cleaned_docs]
        cleaned_queries = [re.sub('[^a-zA-Z0-9 \n\.]', '', s) for s in list(all_uid2query.values())]
        tokenized_queries = [doc.lower().split() for doc in cleaned_queries]
        bm25 = BM25Okapi(tokenized_corpus)
        query2neighbor_idxs = []
        for tq in tokenized_queries:
            dscores = bm25.get_scores(tq)
            ndoc_idxs = sorted(zip(range(len(all_bid2doc)), dscores), key=lambda tu: tu[1], reverse=True)
            query2neighbor_idxs.append([tu[0] for tu in ndoc_idxs[:n_neighbors]])
        query2neighbor_idxs = np.vstack(query2neighbor_idxs)
    # query2neighbor_idxs = np.random.randint(0, len(all_bid2doc), (len(all_uid2query), n_neighbors))
    
    # Write the dev and test splits.
    all_bids = list(all_bid2doc.keys())
    pos_ex_per_user = []
    neg_ex_per_user = []
    for idx, uid in enumerate(all_uid2query):
        nidxs = query2neighbor_idxs[idx].tolist()
        # Get things below rank 50 since they're often relevant at the top.
        query_neighbors = [all_bids[ni] for ni in nidxs][100:]
        if uid in train_uid2posbids:
            pos_bids = train_uid2posbids[uid]
            true_neg_bids = set.difference(set(query_neighbors), set(pos_bids))
            for pbid in pos_bids:
                exdict = {
                    'user_id': uid, 'cited_pids': [uid, pbid],
                    'query_text': all_uid2query[uid], 'doc_text': all_bid2doc[pbid],
                    'label': 1
                }
                out_train_file.write(json.dumps(exdict)+'\n')
            pos_ex_per_user.append(len(pos_bids))
            neg_bids = random.sample(list(true_neg_bids), k=min(len(true_neg_bids), len(pos_bids)*train_neg_per_pos))
            for nbid in neg_bids:
                exdict = {
                    'user_id': uid, 'cited_pids': [uid, nbid],
                    'query_text': all_uid2query[uid], 'doc_text': all_bid2doc[nbid],
                    'label': 0
                }
                out_train_file.write(json.dumps(exdict)+'\n')
            neg_ex_per_user.append(len(neg_bids))
        else:
            pos_bids = dev_uid2posbids[uid]
            true_neg_bids = set.difference(set(query_neighbors), set(pos_bids))
            pos_exdicts = []
            dev_ex = {
                'user_id': uid,
                'query_text': all_uid2query[uid]
            }
            for pbid in pos_bids:
                exdict = {'cited_pids': [uid, pbid], 'doc_text': all_bid2doc[pbid], 'label': 1}
                pos_exdicts.append(exdict)
            dev_ex['positive_docs'] = pos_exdicts
            neg_bids = random.sample(list(true_neg_bids), k=min(len(true_neg_bids), dev_negs))
            neg_exdicts = []
            for nbid in neg_bids:
                exdict = {'cited_pids': [uid, nbid], 'doc_text': all_bid2doc[nbid], 'label': 0}
                neg_exdicts.append(exdict)
            dev_ex['negative_docs'] = neg_exdicts
            out_dev_file.write(json.dumps(dev_ex)+'\n')
    print(f'Wrote: {out_train_file.name}')
    in_train_file.close()
    print(f'Wrote: {out_dev_file.name}')
    out_dev_file.close()
    print('Positive per user: mean: {:.2f}; median: {:.2f}; max: {:.2f}; min: {:.2f}; total: {:.2f}'.format(
        statistics.mean(pos_ex_per_user), statistics.median(pos_ex_per_user), max(pos_ex_per_user),
        min(pos_ex_per_user), sum(pos_ex_per_user)))
    print('Negative per user: mean: {:.2f}; median: {:.2f}; max: {:.2f}; min: {:.2f}; total: {:.2f}'.format(
        statistics.mean(neg_ex_per_user), statistics.median(neg_ex_per_user), max(neg_ex_per_user),
        min(neg_ex_per_user), sum(neg_ex_per_user)))
    print(f'Total train examples: {sum(pos_ex_per_user)+sum(neg_ex_per_user)}')
    

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand',
                                       help='The action to perform.')
    
    print_args = subparsers.add_parser('print_dataset')
    print_args.add_argument('--in_path', required=True,
                            help='Path to the raw dataset')
    write_args = subparsers.add_parser('write_json')
    write_args.add_argument('--in_path', required=True,
                            help='Path to the raw dataset')
    select_args = subparsers.add_parser('select_users_dataaug')
    select_args.add_argument('--in_path', required=True,
                             help='Path to the json dataset')
    write_dataaug_args = subparsers.add_parser('write_dataaug_examples')
    write_dataaug_args.add_argument('--in_path', required=True,
                                    help='Path to the json dataset')
    write_dataaug_args.add_argument('--item_examples', default=False, action="store_true",
                                    help='If the data aug examples should be made per item.')
    get_dataaug_args = subparsers.add_parser('get_gpt3_outs')
    get_dataaug_args.add_argument('--in_path', required=True,
                                  help='Path to the json dataset')
    get_dataaug_args.add_argument('--out_path', required=True,
                                  help='Path PER run for the output file '
                                       'since runs are expensive.')
    get_dataaug_args.add_argument('--item_examples', default=False, action="store_true",
                                  help='If the data aug examples should be made per item.')
    write_ex_args = subparsers.add_parser('write_pairdoc_examples')
    write_ex_args.add_argument('--in_path', required=True,
                               help='Path to the json dataset')
    write_ex_args.add_argument('--out_path', required=True,
                               help='Path where output examples should be written.')
    write_ex_args.add_argument('--llm_model_name', required=False,
                               choices=['curie-001', 'davinci-003'],
                               help='The llm used for generating synthetic data.')
    write_ex_args.add_argument('--rdrop', required=False, default=0, choices=[0, 0.2, 0.3, 0.5],
                               help='Probability of dropping the description for the item', type=float)
    write_ex_args.add_argument('--dataset', required=True,
                               choices=['yelppoi'],
                               help='Source dataset; prediction code reads this for the apt target')
    write_ex_args.add_argument('--model_name', required=True,
                               choices=['recinpars'],
                               help='InPars but for recommendations.')
    fwrite_ex_args = subparsers.add_parser('filter_write_pairdoc_examples')
    fwrite_ex_args.add_argument('--in_path', required=True,
                                help='Path to the json dataset')
    fwrite_ex_args.add_argument('--out_path', required=True,
                                help='Path where output examples should be written.')
    fwrite_ex_args.add_argument('--rdrop', required=False, default=0, choices=[0, 0.2, 0.3, 0.5],
                                help='Probability of dropping the description for the item', type=float)
    fwrite_ex_args.add_argument('--filter_frac', required=False, default=0.5, choices=[0.2, 0.3, 0.5],
                                help='Fraction of items in a user profile to drop', type=float)
    fwrite_ex_args.add_argument('--filter_model', required=False, default='qlft5xl',
                                choices=['qlft5l', 'qlft5xl', 'qlft5xxl', 'rand'],
                                help='Model to use for filtering')
    fwrite_ex_args.add_argument('--llm_model_name', required=False,
                                choices=['curie-001', 'davinci-003'],
                                help='The llm used for generating synthetic data.')
    fwrite_ex_args.add_argument('--dataset', required=True,
                                choices=['yelppoi'],
                                help='Source dataset; prediction code reads this for the apt target')
    fwrite_ex_args.add_argument('--model_name', required=True,
                                choices=['recinpars'],
                                help='InPars but for recommendations.')
    cewrite_ex_args = subparsers.add_parser('write_crossenc_examples')
    cewrite_ex_args.add_argument('--in_path', required=True,
                                 help='Path to the json dataset')
    cewrite_ex_args.add_argument('--out_path', required=True,
                                 help='Path where output examples should be written.')
    cewrite_ex_args.add_argument('--neg_model', required=True,
                                 choices=['recinpars', 'sbcontriver', 'okapibm25'],
                                 help='Model to use for generating negatives for the cross-encoder.')
    cewrite_ex_args.add_argument('--neg_model_runpath', required=False, default=None,
                                 help='Run path for the model used for generating negatives.')
    cewrite_ex_args.add_argument('--rdrop', required=False, default=0, choices=[0, 0.2, 0.3, 0.5],
                                help='Probability of dropping the description for the item', type=float)
    cewrite_ex_args.add_argument('--filter_frac', required=False, default=None, choices=[0.2, 0.3, 0.5],
                                 help='Fraction of items in a user profile to drop', type=float)
    cewrite_ex_args.add_argument('--filter_model', required=False, default=None,
                                 choices=['qlft5l', 'qlft5xl', 'qlft5xxl', 'rand'],
                                 help='Model to use for filtering')
    cewrite_ex_args.add_argument('--llm_model_name', required=False,
                                 choices=['curie-001', 'curie-001-itemqs', 'davinci-003'],
                                 help='The llm used for generating synthetic data.')
    cewrite_ex_args.add_argument('--dataset', required=True,
                                 choices=['yelppoi'],
                                 help='Source dataset; prediction code reads this for the apt target')
    cewrite_ex_args.add_argument('--model_name', required=True,
                                 choices=['recinparsce'],
                                 help='InPars but for recommendations.')
    
    cl_args = parser.parse_args()
    
    if cl_args.subcommand == 'print_dataset':
        print_dataset(in_path=cl_args.in_path)
    elif cl_args.subcommand == 'write_json':
        write_jsons(in_path=cl_args.in_path)
    elif cl_args.subcommand == 'select_users_dataaug':
        select_users_dataaug(in_path=cl_args.in_path)
    elif cl_args.subcommand == 'write_dataaug_examples':
        if cl_args.item_examples:
            write_dataaug_examples(in_path=cl_args.in_path, item_examples=True)
        else:
            write_dataaug_examples(in_path=cl_args.in_path, item_examples=False)
    elif cl_args.subcommand == 'get_gpt3_outs':
        if cl_args.item_examples:
            get_gpt3_narrativeqs(in_path=cl_args.in_path, out_path=cl_args.out_path,
                                 item_examples=True)
        else:
            get_gpt3_narrativeqs(in_path=cl_args.in_path, out_path=cl_args.out_path)
    elif cl_args.subcommand == 'write_pairdoc_examples':
        create_pairdoc_examples(data_path=cl_args.in_path, out_path=cl_args.out_path,
                                dataset=cl_args.dataset, model_name=cl_args.model_name,
                                rdrop=cl_args.rdrop, llm_model_name=cl_args.llm_model_name)
    elif cl_args.subcommand == 'filter_write_pairdoc_examples':
        filter_create_pairdoc_examples(data_path=cl_args.in_path, out_path=cl_args.out_path,
                                       rdrop=cl_args.rdrop, filter_frac=cl_args.filter_frac,
                                       filter_model=cl_args.filter_model,
                                       llm_model_name=cl_args.llm_model_name)
    elif cl_args.subcommand == 'write_crossenc_examples':
        create_pairdoc_examples_crossenc(in_path=cl_args.in_path, out_path=cl_args.out_path,
                                         filter_frac=cl_args.filter_frac, filter_model=cl_args.filter_model,
                                         llm_model_name=cl_args.llm_model_name,
                                         neg_model=cl_args.neg_model,
                                         trained_neg_runpath=cl_args.neg_model_runpath)


if __name__ == '__main__':
    main()
