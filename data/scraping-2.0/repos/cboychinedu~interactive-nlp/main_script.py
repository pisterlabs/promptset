#!/usr/bin/env python3 

# Importing the necessary packages. 
import nltk 
import tflearn 
import requests 
import re 
import socket 
import random 
import pickle 
import json 
import numpy as np 
import tensorflow as tf
import logging
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import torch
import torch.nn.functional as F 
from pytorch_pretrained_bert import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments
from utils import get_dataset_personalities, download_pretrained_model
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords 
from bs4 import BeautifulSoup 
from subprocess import call 
from time import sleep 
from gtts import gTTS 


# Setting the variable for the stemmer function, Python text to speech
# And loading in the json dataset. 
stemmer = LancasterStemmer()
ps = PorterStemmer()
with open('model/intents.json') as file:
    data = json.load(file)


# Setting a variable to hold the question, and empty dictionary.
# and creating an empty list to hold the filtered sentence.
running = True
remote_server = 'www.google.com'
stop_words = stopwords.words('english')
information = {}
filtered_sentence = []
filtered_questions = []
model1_words = ['what', 'do', 'you', 'know', 'about ', 'is', 'is ', 'define', 'find', 'tell', 'tell ', 'me', 'about', 'define ']


# Creating a function to seperate words into their different Named 
# Entity Function and Part of speech. 
def extract_noun(nouns):
    words = ps.stem(nouns)
    words = word_tokenize(words)
    for w in words:
        if (w not in model1_words) and (w not in stop_words):
            filtered_questions.append(w)
    return filtered_questions   



# Loading in the pickle file that contains the labels, training, word
# and output data into memory. 
try:
    with open('model/data.pickle', 'rb') as f:
        words, labels, training, output = pickle.load(f)

# Creating an empty list to store some values. 
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    
    # Creating a loop that would stem the words in the json dataset, 
    # and append them into the list created above. 
    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])
            
        # Creating an if statement to append the word that are not present in,
        # the labels list into the label list.
        if intent['tag'] not in labels:
            labels.append(intent['tag'])
        
    # Stemming the words and converting them into lowercase alphabets,
    # then setting an if statement to remove the ? character.
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    
    # Sorting the value for the words in labels and saving them into a 
    # new variable called labels.
    labels = sorted(labels)
    
    # Creating a list that would hold the training and output values.
    training = []
    output = []
    
    # Creating a new variable that would hold a range of values from 0 to the length of the 
    # labels 
    out_empty = [0 for _ in range(len(labels))]
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
                
        # Creating a variable that would hold out the empty list values.
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1 
        
        # Appending the numbers in the bag of word list into the training list.
        # and appending the values of the output row into output list.
        training.append(bag)
        output.append(output_row)
        
    # Setting the training and output variable as a numpy array.
    training = np.array(training)
    output = np.array(output)
    
    # loading the pickle file into memory and dumping the labels, training data, 
    # and output values.
    with open('model/data.pickle', 'wb') as f:
        pickle.dump((words, labels, training, output), f)


# Testing for internet connection by using the socket module to 
# Access google website to see if there is a connection, and return
# a booleon value True and assign it to a variable called value.
def internet_test():
    global value
    try:
        host = socket.gethostbyname(remote_server)
        s = socket.create_connection((host, 80), 2)
        s.close() 
        value = True 
    
    # Printing out that there is not internet connection, and 
    # Setting the value to be False.
    except:
        print('There is no internet connection.')
        value = False



# Setting the Number of layers to be 3 and 8 neurons for each layers, with
# a sotfmax activation.
tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# Setting the Type of model used and setting the neural network
model = tflearn.DNN(net)


# Creating a function for Speaking the results, and replies 
# Using Google text to speech module.
def google_voice(result):
    tts = gTTS(result)
    tts.save('voice.mp3')
    call('mpg321 voice.mp3 2>/dev/null', shell=True)
    call('rm voice.mp3', shell=True)


# Loading in the Trained model. 
model.load("model/model.tflearn")

# Defining a function to take in two words, and stem the words before 
# Saving them into a variable
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    
    # using nltk word tokenizer to convert the sentences into a list of words.
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    
    # Creating a loop to loop through the list of words and append the 
    # value 1 into the list if the word is present and return it as a numpy array.
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)



# Creating a function that takes in the users's input, 
# and sends the result to wikipedia page for the required results.
# Also test for internet connection before acessing wikipedia page.
def asked_question(quest): 
    question = quest
    
    # Calling in the internet test to check for internet connection, 
    # And return a boolean value based on the results gotten.
    # if connection exists, then perform the actions below.
    global value
    if value is True:
        question = question.lower()
        match = re.search('define', question)
        if match is match:
            question = re.sub('define ', '', question)
        else:
            pass
        # calling in the extract_noun funtion to remove all the words and leave
        # only nouns and pronouns and assign it to a varible called question.
        # Then loop through the varible created and append the non stop words 
        # into a list called filtered sentence.
        question = extract_noun(question)
        # Creating a variable called new value that would contain the 
        # string that would be parsed into wikipedia page.
        # joining the value of the string by an underscore if there were 2 or more words present.
        value = question 
        if len(value) is 1:
            new_value = ''.join(value)

        else:
            length = len(value)
            if length == 2:
                new_value = '{}_{}'.format(value[length-length], value[length-1])

            elif length == 3:
                new_value = '{}_{}_{}'.format(value[length-length], value[length-2], value[length-1])

            elif length == 4:
                new_value = '{}_{}_{}_{}'.format(value[length-length], value[length-3], value[length-2], value[length-1])


        # Specifying the url for the website to get the Required information. 
        # passing the search result through a parset and removing the html tags, 
        # then coverting the result into text for processing. 
        # Specifying the count value to be zero and break at 11 for data retrival.
        url = 'https://en.wikipedia.org/wiki/{}'.format(new_value)
        page = requests.get(url).text
        soup = BeautifulSoup(page, 'lxml')
        cnt = 0 
        for p in soup.find(class_='mw-parser-output').find_all('p'):
            text = p.text
            text = text.lower() 
            text = text.strip().rstrip()
            text = sent_tokenize(text)
            for items in text:
                if cnt >= 11:
                    break
                information['value{}'.format(cnt)] = items
                cnt = cnt + 1 
        
        # Specify the count to zero, and 9 for maximum number of text that our, 
        # speech function reads out to us.
        # Reading the first key value in the dictionary for the value of counts noted.
        # created an if/else statement denoting if the value of count exceeds 4, ask the 
        # user if he/she wants extra information. 
        count = 0 
        for i in range(0, 9):
            text = information['value{}'.format(i)]
            # Creating an if else statement to specify the maximum count to be 4,
            # and if its greater than 4, ask us if we want more information.
            if count >= 4:
                result = 'Do you still want more information on {}'.format(new_value)
                google_voice(result)
                count = 0
                msg = input(': ')
                if msg.lower() == 'yes':
                    continue
                if msg.lower() == 'no':
                    break
            else:
                pass
            
            # Calling in google voice to speak the text and increment the count by 1.
            google_voice(text)
            sleep(0.2)
            count = count + 1 
    
    # for there not to be an internet connection, tell us.
    if value is False:
        call('mpg321 no_internet.mp3 2>/dev/null', shell=True)


# Defining a function to start the program and start chatting with the Bot. 
def main(message):
    internet_test()
    msg = message
    msg = msg.lower()
    
    # Running the Actual Prediction on the input sentences against the Trained model to give
    # floating point values, then converting them into the indexes of the labels by using the 
    # numpy argmax function. 
    # Then placing the value into the labels to get the Actual Predicted tag for the input question 
    results = model.predict([bag_of_words(msg, words)])
    results_index = np.argmax(results)
    tag = labels[results_index]
    
    # Creating a loop to loop through the intents dictionary and perform a check 
    # to see if the predicted tag equals the tag in the dictionary. 
    # if it equals the tag, then print a random choice in the tag response list. 
    for tg in data['intents']:
        if tg['tag'] == tag:
            responses = tg['responses']
            
    reply = random.choice(responses)
    
    # Creating a try and except method to perform the following functions below 
    # Creating an if statement to check if the tag equals the definition tag 
    # and starting the wikipedia function to pass the words asked into it. 
    try:
        if tag == 'definition':
            print('MyData: {}'.format(reply))
            google_voice(reply)
            asked_question(msg)
            
        else:
            sleep(1)
            print('MyData: {}'.format(reply))
            google_voice(reply)
            
    except:
        print('MyData: i did not get that..')
        pass
    

# Creating a function to filter a distribution of logits using top-k,
# top-p (nucleus) and/or threshold filtering
def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


# Defining a function to take in the personality, history , model and argument. 
def sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance, sequence = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)

        if "gpt2" == args.model:
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


# Defining a function to Start the A.I Bot and take in some Arguments.
def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="gpt", help="Model type (gpt or gpt2)")
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        args.model_checkpoint = download_pretrained_model()

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class = GPT2Tokenizer if "gpt2" == args.model else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model_class = GPT2LMHeadModel if "gpt2" == args.model else OpenAIGPTLMHeadModel
    model = model_class.from_pretrained(args.model_checkpoint)

    model.to(args.device)
    model.eval()

    logger.info("Sample a personality")
    personalities = get_dataset_personalities(tokenizer, args.dataset_path, args.dataset_cache)
    personality = random.choice(personalities)
    logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))

    history = []
    while True:
        print("MyData: Type ==> 'Convo' for an interactive chat OR  Type ==> 'ask' for Asking Questions... ")
        raw_text = input("You: ")
        raw_text = raw_text.lower()
        if raw_text.replace(' ', '') == 'ask':
            print('MyData: Please Ask your Question..')
            message = input('You: ')
            main(message)
        
        elif raw_text.replace(' ', '') == 'convo':
            print('MyData: Hello, whats up?')
            continue 
        while not raw_text:
            print('Please i did not get that')
            raw_text = input("You: ")
        history.append(tokenizer.encode(raw_text))
        with torch.no_grad():
            out_ids = sample_sequence(personality, history, tokenizer, model, args)
        history.append(out_ids)
        history = history[-(2*args.max_history+1):]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        print("MyData: {}".format(out_text))


# Starting the Robot.
run() 

