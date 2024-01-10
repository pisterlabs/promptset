import openai
import tweepy as twitter
import re
from Resources import keys
from Twitter import Twitter
import Brain
import Settings

auth = twitter.OAuthHandler(keys.consumer_key, keys.consumer_secret)
auth.set_access_token(keys.oa_key, keys.oa_secret)
api = twitter.API(auth)

def send_reply(order,curr_id, user):
    result = ""
    # create a reply
    try:
        # create a response
        result = Brain.Brain().create_response(order, [f"{user}:", "You:", "Human:"])
        # process the response
        result=process_str(result)
        print(f"My Reply: {result}")
    except openai.OpenAIError as e:
        print(f"[send_reply] openAI Error: {e}")

    # tweet the reply
    if Settings.production:
        try:
            Twitter().tweet_reply(result, curr_id)
        except twitter.errors.TweepyException as e:
            print(f"[send_reply] Twitter Error: {e}\n")
    else:
        print("reply tweeted - Development mode\n")

def construct_conv_order(tw_id):
    chats = []
    rd = api.get_status(id=tw_id)
    while True:
        try:
            data = rd[0]._json
        except:
            data = rd._json

        text = data['text']
        user = data['user']['screen_name']

        if user == 'Justin_prg':
            user = 'You'

        chats.append(f"{user}:{text}")
        parent_id = data['in_reply_to_status_id']
        if parent_id is None:
            break

        rd = api.get_status(id=parent_id)
    # reverse chats
    chats.reverse()
    order = ""
    for i, chat in enumerate(chats):
        chat = re.sub('\n', '', chat)
        chat += '\n'
        order += chat

    order = f'{Settings.prompt_reply}' + order
    order+='You:'
    print("\n-------start of the order-------")
    print(order)
    print("-------end of the order-------\n")
    return order

def process_str(result):
    result = re.sub('@[a-zA-Z_0-9]*', '', result)
    result = re.sub('\n', '', result)

    last = result.find('Friend:')
    if last==-1:
        return result
    else:
        return result[:last]