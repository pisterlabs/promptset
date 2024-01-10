#%%
from pathlib import Path
import json
import pandas as pd
'''
pandas
'''

review_file = Path(__file__).parent.resolve() / 'yelp_academic_dataset_review.json'
business_file = Path(__file__).parent.resolve() / 'yelp_academic_dataset_business.json'

business_df = pd.read_json(business_file, lines=True)
business_df = business_df[business_df['categories'].notna()]


# %%
rest_df = business_df[business_df['categories'].str.contains('Restaurants')]
rest_df.set_index('business_id', inplace=True)
#%%

for line in open(review_file):
    review = json.loads(line)
    if review['business_id'] == 'k0hlBqXX-Bt0vf1op7Jr1w':
        print(review['text'])
        break


# %%

review

#%%

from dotenv import load_dotenv
import openai
import os
import json
from pathlib import Path
from dataclasses import asdict

from dataset.ACOS.utils import (
    ACOSQuad, ACOSReview, ACOSPrediction,
    load_acos_reviews, openai_pred_acos, evaluate_acos_preds,
    laptop_acos_train_file, laptop_acos_dev_file,
    restaurant_acos_train_file, restaurant_acos_dev_file,
)

load_dotenv()
openai.organization = os.getenv('OPENAI_ORG_ID')
openai.api_key = os.getenv('OPENAI_KEY')

openai_model = "text-davinci-003"

# %%
acos_reviews: list[ACOSReview] = load_acos_reviews(restaurant_acos_train_file, 10)
print(f"Loaded {len(acos_reviews)} ACOS reviews")

# %%
# reviews_to_predict = [ACOSReview(review['text'], [])]

# acos_preds: list[ACOSPrediction] = openai_pred_acos(reviews_to_predict, acos_reviews, openai_model)



# %%
rest_df.sort_values('review_count', ascending=False).head(10)
# rest_df[(rest_df['state'] == 'CA')].sort_values('review_count', ascending=False).head(10)


#%%
'''
Global list of 200 reviews for 1 restaurant

Groups of reivews based on aspect
Groups of reivews based on category

For genericity, 10 review for 100 restaurants

'''
num_reviews = 200
bid = '_ab50qdWOk0DdB6XOrBitw'

reviews_to_predict: list[ACOSReview] = []
for line in open(review_file):
    review = json.loads(line)
    if review['business_id'] == bid:
        reviews_to_predict.append(ACOSReview(review['text'], []))
        num_reviews -= 1
        if num_reviews == 0:
            break

print(f"Loaded {len(reviews_to_predict)} reviews to predict")
# %%
acos_preds: list[ACOSPrediction] = openai_pred_acos(reviews_to_predict, acos_reviews, openai_model)

# acos_preds2: list[ACOSPrediction] = openai_pred_acos(reviews_to_predict[100:], acos_reviews, openai_model)
# acos_preds += acos_preds2
len(acos_preds)
#%%
output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)

output_file = output_dir / f'yelp-acos-{bid}-{num_reviews}.jsonl'
with open(output_file, 'w') as f:
    for i, acos_pred in enumerate(acos_preds):
        f.write(json.dumps(asdict(acos_pred)) + '\n')


# %%
def print_acos_pred(acos_pred: ACOSPrediction):
    print(f"Review: {acos_pred.review}")
    print(f"Predicted:")
    for i, quad in enumerate(acos_pred.acos_preds):
        print(f"\tAspect: {quad.aspect}")
        print(f"\tCategory: {quad.category}")
        print(f"\tOpinion: {quad.opinion}")
        print(f"\tSentiment: {quad.sentiment}")
        print()


print_acos_pred(acos_preds[3])


# %%
# from collections import Counter
# pos_aspect_word_counts = Counter()
# pos_aspect_counts = Counter()
# neg_aspect_word_counts = Counter()
# neg_aspect_counts = Counter()
# for acos_pred in acos_preds:
#     for quad in acos_pred.acos_preds:
#         if quad.aspect is None:
#             continue
#         if quad.sentiment == 'positive':
#             pos_aspect_word_counts.update(quad.aspect)
#             pos_aspect_counts.update([tuple(quad.aspect)])
#         elif quad.sentiment == 'negative':
#             neg_aspect_word_counts.update(quad.aspect)
#             neg_aspect_counts.update([tuple(quad.aspect)])


# pos_aspect_counts.most_common(10)
# pos_aspect_word_counts.most_common(10)

# neg_aspect_counts.most_common(10)
# neg_aspect_word_counts.most_common(10)

# print(f"Positive aspect counts: {len(pos_aspect_counts)}")
# print(f"Negative aspect counts: {len(neg_aspect_counts)}")
# print(f"Positive aspect word counts: {len(pos_aspect_word_counts)}")
# print(f"Negative aspect word counts: {len(neg_aspect_word_counts)}")
# print(f"Top 10 positive aspect words: \n\t{pos_aspect_word_counts.most_common(10)}")
# print(f"Top 10 negative aspect words: \n\t{neg_aspect_word_counts.most_common(10)}")
# print(f"Top 10 positive aspects: \n\t{pos_aspect_counts.most_common(10)}")
# print(f"Top 10 negative aspects: \n\t{neg_aspect_counts.most_common(10)}")
# %%
from collections import Counter
aspect_counts = Counter()
category_counts = Counter()
for acos_pred in acos_preds:
    for quad in acos_pred.acos_preds:
        if quad.aspect is not None:
            aspect_counts.update([' '.join(quad.aspect)])
        category_counts.update([quad.category])

print(f"Aspect counts: {len(aspect_counts)}")
print(f"Top 10 aspects: \n\t{aspect_counts.most_common(10)}")
print(f"Category counts: {len(category_counts)}")
print(f"Top 10 categories: \n\t{category_counts.most_common(10)}")


# %%
aspect_groups: dict[str, list[ACOSPrediction]] = {}
for word, count in aspect_counts.most_common(7):
    aspect_groups[word] = []
    for acos_pred in acos_preds:
        for quad in acos_pred.acos_preds:
            if quad.aspect is None:
                continue
            if word == ' '.join(quad.aspect):
                aspect_groups[word].append(acos_pred)

print(f"Aspect group size: \n\t{[(word, len(group)) for word, group in aspect_groups.items()]}")

category_groups: dict[str, list[ACOSPrediction]] = {}
for word, count in category_counts.most_common(5):
    category_groups[word] = []
    for acos_pred in acos_preds:
        for quad in acos_pred.acos_preds:
            if word == quad.category:
                category_groups[word].append(acos_pred)

print(f"Category group size: \n\t{[(word, len(group)) for word, group in category_groups.items()]}")
#%%


'''
Global list of 200 reviews for 1 restaurant

Groups of reivews based on aspect
Groups of reivews based on category

For genericity, 10 review for 100 restaurants

'''

aspect_group_output_file = output_dir / f'yelp-acos-{bid}-{num_reviews}-aspect-groups.json'

json.dump([
    {
        'group_name': aspect,
        'group_size': len(group),
        'group': [asdict(acos_pred) for acos_pred in group]
    }
    for aspect, group in aspect_groups.items()
], open(aspect_group_output_file, 'w'), indent=2)


# %%

category_group_output_file = output_dir / f'yelp-acos-{bid}-{num_reviews}-category-groups.json'

json.dump([
    {
        'group_name': category,
        'group_size': len(group),
        'group': [asdict(acos_pred) for acos_pred in group]
    }
    for category, group in category_groups.items()
], open(category_group_output_file, 'w'), indent=2)

# %%

from collections import defaultdict
sampled_bids = rest_df[rest_df['review_count'] > 50].sample(100).index

sampled_reviews: dict[str, list[str]] = defaultdict(list)
for line in open(review_file):
    review = json.loads(line)
    if review['business_id'] in sampled_bids and len(sampled_reviews[review['business_id']]) < 10:
        sampled_reviews[review['business_id']].append(review['text'])


print(f"Sampled restaurant counts: {len(sampled_reviews)}")
print(f"Sampled review counts: {sum(len(reviews) for reviews in sampled_reviews.values())}")

# %%

sampled_rest_review_output_file = output_dir / f'yelp-acos-100-sampled-rest-reviews.json'

json.dump([
    {
        'restaurant_name': rest_df.loc[bid]['name'],
        'restaurant_id': bid,
        'reviews': reviews
    }
    for bid, reviews in sampled_reviews.items()
], open(sampled_rest_review_output_file, 'w'), indent=2)

# %%
