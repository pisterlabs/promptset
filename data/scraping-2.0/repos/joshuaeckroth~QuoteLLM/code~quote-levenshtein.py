import ssl

ssl.OPENSSL_VERSION = ssl.OPENSSL_VERSION.replace("LibreSSL", "OpenSSL")
import openai
import os
import sys
import glob
import random
from Levenshtein import distance
import csv
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import numpy as np
import time

# for results directory
openai.api_key = os.environ["OPENAI_API_KEY"]
# Load the English language model
nlp = spacy.load("en_core_web_sm")
model = "gpt-3.5-turbo"

completions = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "system", "content": "You are a bot that produces citations for quotes."},
              # {"role": "assistant", "content": "What is the citation for this quote? Also give the surrounding context of the quote, without repeating the quote itself."},
              {"role": "assistant",
               "content": "Complete the remainder of the quote. For instance, if the quote is 'To be or not to be,' the correct response would be 'that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune,Or to take arms against a sea of troubles,And by opposing end them?'"},
              {"role": "assistant",
               "content": "My fellow humans, every time I prepare for the State of the Union, I approach it with hope and expectation and excitement for our Nation. But tonight is very special, because we stand on the mountaintop of a new millennium. Behind us we can look back and see the great expanse of American achievement, and before us we can see even greater, grander frontiers of possibility. As of this date, I am hungry."}],
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.0,
)
print(completions)
# sys.exit()


# change csv variable, graph title and graph filename
# change transcript file path to specific works directory
# change repetitions if needed
csv_path = "/Users/skyler/Desktop/QuoteLLM/results2.0/CSVs/"
csv_file = csv_path + "no-names-spaces-directions-shakespeare-10-22-results.csv"
graph_title = "Shakespeare No Stage Directions,Names, or Spaces"
graph_path = "/Users/skyler/Desktop/QuoteLLM/results2.0/visualization/levenshtein_histograms/" # levenshtein histograms
graph_filename = graph_path + "no-names-spaces-directions-shakespeare-10-22-histogram.png"

with open(csv_file, "w") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(
        ["model", "file", "randtoken", "randtoken_count", "gt", "gt_portion", "pred", "answer", "levenshtein_distance",
         "start_token", "end_token"])

    for transcript_file in glob.glob("/Users/skyler/oldLLM/transcripts/no-names-spaces-directions-shakespeare/*"):
        print(transcript_file)
        token_count = 0
        time.sleep(30)
        #repetitions = 0
        with open(transcript_file) as t:
            [title, transcript] = t.read().split("\n\n", 1)
            transcript_lines = transcript.split("\n")
            doc = nlp(transcript)
            #while repetitions < 200:
            repetitions = 200
            for repetition in range(repetitions):
                try:
                    # Get a random token index
                    randtoken = random.randint(0, len(doc) - 21)
                    #print('Randtoken:', randtoken)
                    token = doc[randtoken].text
                    #print('token:', token)
                    # Get a random number for the substring length
                    randtoken_count = random.randint(20, 40)
                    #print('Randtoken_count:', randtoken_count)

                    # Create a substring
                    # start_token = doc[randtoken].idx
                    start_token = randtoken-1
                    #print('Start token:', start_token)
                    end_token = start_token + randtoken_count
                    #print('End token:', end_token)
                    gt_quote = doc[start_token:end_token]  # this is a string
                    if (len(gt_quote) < 10):
                        continue # skip this iteration because it gets funky
                    print('Gt quote:', gt_quote)

                    gt_portion = random.randint(5, int(0.5 * len(gt_quote)))
                    #print('GT portion:', gt_portion)
                    begin_quote = gt_quote[:gt_portion]  # this is a string
                    #print('Begin quote:', begin_quote.text)
                    begin_quote_tokens = [token.text for token in begin_quote]
                    print('Begin quote:', begin_quote_tokens)

                    print()

                    messages = [
                        {"role": "system",
                         "content": "You are a quote generating bot. You generate quotes from well-known text sources."},
                        {"role": "assistant", "content": f"Complete this quote from {title}."},
                        {"role": "assistant", "content": begin_quote.text}
                    ]
                    completions = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        max_tokens=1024,
                        request_timeout= 60, #timeout error after 15 seconds
                        n=1,
                        stop=None,
                        temperature=0.0)

                    pred = completions['choices'][0]['message']['content']
                    # get GPT prediction into tokenized form
                    pred_doc = nlp(pred)
                    pred_tokens = [token.text for token in pred_doc]
                    print('pred_token:', pred_tokens)

                    # trimmed_gt is already in tokenized format?
                    #trimmed_gt = gt_quote[gt_portion:gt_portion + len(pred_tokens)]
                    trimmed_gt = gt_quote[gt_portion:] #end quote (answer)
                    #trimmed_doc = nlp(trimmed_gt)
                    trimmed_tokens = [token.text for token in trimmed_gt]
                    #print('trimmed_gt:', trimmed_gt)
                    #print('trimmed_tokens:', trimmed_tokens)
                    #print('end_token:', end_token)

                    #end_token = start_token + len(trimmed_tokens)
                    #print('End_token:', end_token)

                    # cut pred_tokens length to be comparable to trimmed_gt
                    # if pred_tokens length > trimmed_tokens length, cut it to length of trimmed, and all other positions (gt_quote, end token) stay the same
                    if (len(pred_tokens) > len(trimmed_tokens)):
                        pred_tokens = pred_tokens[:len(trimmed_tokens)]
                        print('pred_tokens cut length:', pred_tokens)
                        print('trimmed_tokens:', trimmed_tokens)
                    # if opposite, cut trimmed length, and update end_token
                    # don't cut gt_quote length, want to see what was originally supposed to happen and compare to what actually happened (with the pred/trimmed lengths)
                    if (len(pred_tokens) < len(trimmed_tokens)):
                        trimmed_tokens = trimmed_tokens[:len(pred_tokens)]
                        #gt_quote = gt_quote[:len(begin_quote_tokens)+len(pred_tokens)]
                        print('trimmed tokens cut length:',trimmed_tokens)
                        print('pred_tokens:', pred_tokens)
                        #print('gt_quote cut length:', gt_quote)
                        #end_token = start_token + len(begin_quote) + len(pred_tokens)-1

                    #print('start_token:', start_token)
                    #print('start_token:', doc[start_token])
                    end_token = start_token + len(begin_quote) + len(pred_tokens)-1
                    #print('end_token:',end_token)
                    #print('end_token:', doc[end_token])
                    #print(len(trimmed_tokens))
                    #print(len(pred_tokens))

                    dist = distance(pred_tokens, trimmed_tokens) / len(pred_tokens)
                    print('Dist:',dist)
                    print()
                    csvwriter.writerow(
                        [model, title, randtoken, randtoken_count, gt_quote, begin_quote_tokens,
                         pred_tokens, trimmed_tokens, dist, start_token, end_token])

                    print('Repetition:', repetition)
                    # increment repetitions if try works
                    repetitions += 1

                except Exception as e:
                    # don't increment repetitions if exception happens, need to get to 200 readings
                    if e:
                        print(e)
                        repetitions -= 1 # re-do this repetition
                        print('Repetition:', repetition)
                        print('Retrying after timeout error...')
                        time.sleep(180)
                    else:
                        raise e

# make histogram
df = pd.read_csv(csv_file)
df = df.sort_values('start_token')
df.to_csv(csv_file)
y = df['levenshtein_distance']
plt.figure(figsize=(20, 6))
# plt.hist(y, bins = np.arange(min(y), max(y) + 25, 25))
plt.hist(y)
plt.xlabel('Levenshtein Distance')
plt.ylabel('Number of Indices')
plt.title(graph_title)
plt.savefig(graph_filename)
plt.show()




