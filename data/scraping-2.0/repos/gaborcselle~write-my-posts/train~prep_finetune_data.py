import openai
import config
import common
import json
import tweetnlp
import pprint
from tqdm import tqdm

openai.api_key = config.OPENAI_API_KEY

SYSTEM_MESSAGE = "You write X/Twitter posts about topics provided by the user."

ner_model = tweetnlp.load_model('ner') 

def prepare_example_conversation(topic, result):
    messages = []
    messages.append({"role": "system", "content": SYSTEM_MESSAGE})
    messages.append({"role": "user", "content": topic})
    messages.append({"role": "assistant", "content": result})
    return {"messages": messages}


def write_jsonl(data_list: list, filename: str) -> None:
    """Write a list of dictionaries to a jsonl file, which is the format required by OpenAI."""
    with open(filename, "w") as out:
        for ddict in data_list:
            jout = json.dumps(ddict) + "\n"
            out.write(jout)

def tag_all_tweets(tweet_text_list):
    """
    Take all the tweet text, run ner_model() on them, 
    return tagged data as a list of
    {'tweet_text': 'tweet text', 'entities': ['entity1', 'entity2', ...]}
    """
    tagged_tweets = []
    for tweet_text in tqdm(tweet_text_list):
        ner_output = ner_model.ner(tweet_text)

        # if no entities are found, skip this tweet
        if len(ner_output) == 0:
            continue

        # remove the prefix space from each 'entity' if it has one
        elist = [inner_list['entity'][1:] if inner_list['entity'][0] == ' ' else inner_list['entity'] for inner_list in ner_output]

        tagged_tweets.append({'tweet_text' : tweet_text, 'entities' : elist})
    
    return tagged_tweets

def main():
    """Read tweets, tag topics, format into example conversations, write to jsonl files."""

    # open tweets.js and remove the prefix of "window.YTD.tweets.part0 = "
    # then parse the contents
    with open('data/tweets.js', 'r') as f:
        tweets = f.read()
        tweets = tweets.replace('window.YTD.tweets.part0 = ', '')

    # parse the remaining string as JSON
    tweets = json.loads(tweets)

    # extract the full_text from all the tweets
    # and put them into a list
    tweet_text_list = [tweet['tweet']['full_text'] for tweet in tweets]

    # TODO(gabor): Open question: drop tweets starting with @?
    #   They are replies and might not be useful for the finetune.

    # tag all the tweets
    tagged_tweets = tag_all_tweets(tweet_text_list)

    # make example conversations for each tweet
    examples_convos = []
    for tagged_tweet in tagged_tweets:
        tweet_text = tagged_tweet['tweet_text']
        entities = tagged_tweet['entities']
        entities_text = ', '.join(entities)
        examples_convos.append(prepare_example_conversation(entities_text, tweet_text))

    # train 80 / test 20 split
    examples_count = len(examples_convos)
    train_count = int(examples_count * 0.8)

    # split the examples into train and test
    train_examples = examples_convos[:train_count]
    validation_examples = examples_convos[train_count:]

    # write the train and test examples to jsonl files
    write_jsonl(train_examples, common.TRAINING_FILE_NAME)
    write_jsonl(validation_examples, common.VALIDATION_FILE_NAME)

if __name__ == "__main__":
    main()