import os
import boto3
import openai
from collections import defaultdict
import random
from dynamodb_json import json_util as dynamodb_json

openai.api_key = os.environ['OPENAI_API_KEY']

_dynamo_resource = boto3.resource('dynamodb')
_words_by_word_type = None


def subvert(event, context):

    load_words()

    for record in event['Records']:

        if record['eventName'] != 'INSERT' and record['eventName'] != 'MODIFY':
            print(f"Skipped record {record['eventID']} because it's not an INSERT or MODIFY event.")
            continue

        story = dynamodb_json.loads(record['dynamodb']['NewImage'])
        print(f"story: {story}")
        if 'SubvertedTitles' in story and story['SubvertedTitles'] is not None:
            print(f"Skipped {story['Title']} because it has already been subverted.")
            continue

        story['SubvertedTitles'] = compute_subverted_titles(story['Title'])
        update_story(story)


def load_words():

    global _words_by_word_type
    _words_by_word_type = defaultdict(set)

    words_table = _dynamo_resource.Table('Words')
    response = words_table.scan()
    words = response['Items']

    for word in words:
        _words_by_word_type[word['WordType']].add(word['Word'])


def compute_subverted_titles(title):
    subverted_titles = []

    prompts = ["Rewrite this headline so that it rhymes",
               "Rewrite this headline so that it's a pun",
               "Rewrite this headline with either assonance or alliteration",
               "Rewrite this headline as a haiku and don't include a period at the end",
               "Rewrite this headline so that it's angry"]

    for prompt in prompts:
        prompt = complete_prompt(prompt)
        tweaked_title = replace_one_word(title)
        subverted_title = fetch_chat_gpt_rewrite(tweaked_title, prompt)
        subverted_titles.append(subverted_title)

    return subverted_titles


def replace_one_word(title):
    candidates = get_candidate_words_to_alter(title)
    random.Random().shuffle(candidates)

    for candidate in candidates:
        replacement_word = get_replacement_word(candidate)
        if replacement_word is not None:
            title = title.replace(candidate, replacement_word)
            print(f"Replaced {candidate} with {replacement_word}. New title: {title}")
            break

    return title


def get_candidate_words_to_alter(title):

    candidates = title.split()

    # Also check if each combination of two consecutive words together are considered a word. This is useful
    # for full names.
    for i in range(len(candidates) - 1):
        candidates.append(f"{candidates[i]} {candidates[i + 1]}")

    candidates = [c for c in candidates if len(c) > 3]
    return candidates


def get_replacement_word(word):

    for word_type in _words_by_word_type.keys():
        if word in _words_by_word_type[word_type]:
            return random.choice(list(_words_by_word_type[word_type]))

    return None


def complete_prompt(prompt):

    random_reference_word_type = random.choice(['place', 'person', 'noun'])
    random_reference = random.choice(list(_words_by_word_type[random_reference_word_type]))

    reference_phrases = ["and include a reference to",
                         "and include an homage to",
                         "and include a"]

    return f"{prompt}, {random.choice(reference_phrases)} {random_reference}:"


def fetch_chat_gpt_rewrite(title, prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a copywriter who writes short headlines in a pithy, succinct and funny style like the New York Post."},
                  {"role": "user", "content": f"{prompt} {title}"}],
        temperature=1.1,
        max_tokens=50,
        frequency_penalty=0.5,
        presence_penalty=-0.5,
    )
    subverted_title = response['choices'][0]['message']['content']
    subverted_title = subverted_title.strip()
    if subverted_title.startswith("\"") and subverted_title.endswith("\""):
        subverted_title = subverted_title[1:-1]

    print(f"Subverted title: {subverted_title} (used {response['usage']['total_tokens']} tokens)")

    return {
        'SubvertedTitle': subverted_title,
        'Prompt': f"{prompt} {title}",
        'TotalTokens': response['usage']['total_tokens']
    }


def update_story(story):
    stories_table = _dynamo_resource.Table('Stories')
    stories_table.update_item(
        Key={
            'YearMonthDay': story['YearMonthDay'],
            'Title': story['Title']
        },
        UpdateExpression="set SubvertedTitles = :s",
        ExpressionAttributeValues={
            ':s': story['SubvertedTitles']
        }
    )
