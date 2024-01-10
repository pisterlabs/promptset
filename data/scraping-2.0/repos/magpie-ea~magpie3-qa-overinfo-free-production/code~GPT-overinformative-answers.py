import openai
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import argparse 

# set openAI key in separate .env file w/ content
# OPENAIKEY = yourkey
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# items = pd.read_csv('trials_LLMs_e1.csv')
# items = pd.read_csv('../experiments/contextSensitive_free_production/trials/trials_e2_fctPrompt_fixedOrder.csv')#[15:].reset_index()
items = pd.read_csv('trials_LLMs_e2.csv')

# preface for GPT3 as a one-shot learner in E1
oneShotExample = '''EXAMPLE: 

You are hosting a barbecue party. You are standing behind the barbecue. You have the following goods to offer: pork sausages, vegan burgers, grilled potatoes and beef burgers. 
Someone asks: Do you have grilled zucchini?

You reason about what that person most likely wanted to have. That they asked for grilled zucchini suggests that they might want vegetarian food. From the items you have pork sausages and beef burgers are least likely to satisfy the persons desires. Vegan burgers and grilled potatoes come much closer. Grilled potatoes are most similar to grilled zucchini.
You reply: I'm sorry, I don't have any grilled zucchini. But I do have some grilled potatoes.

YOUR TURN:
'''
# preface for GPT3 as a one-shot learner in E2
oneShotExampleE2 = '''EXAMPLE:

You give a dinner party at your apartment. More people showed up than you expected. 
Your neighbor, who just arrived, approaches you and asks: Do you have a spare chair I could borrow?

You do not, in fact, have a spare chair, but you do have the following items: a broom, a TV armchair, a drum throne, a ladder and a kitchen table. You deliberate your response as follows. The practical goal of the questioner is to sit down at the dinner table. For this purpose, the most useful object from the list of available items is the stool.
So you say: No, I don't have a spare chair, but you can have the stool.

YOUR TURN:
'''
oneShotExampleE2_polina = '''EXAMPLE:

You are out in the forest for a camping trip with your friends. You are about to camp for the first night in a new location.
Your friend starts to stake the tent and asks: Do you have a hammer?

You do not, in fact, have a hammer, but you do have the following available options: a handsaw, a rock, an oil lamp and a pocket knife.
You deliberate your response as follows. The practical goal of the questioner is to hammer the tent stake into the ground. You reason about the most useful alternative from the list of available options.

YOUR TURN:
'''


def getLogProbContinuation(initialSequence, continuation, preface = ''):
    """
    Helper for retrieving log probability of different response types from GPT-3.
    """
    initialSequence = preface + initialSequence
    response = openai.Completion.create(
            engine      = "text-davinci-003", 
            prompt      = initialSequence + " " + continuation,
            max_tokens  = 0,
            temperature = 1, 
            logprobs    = 0,
            echo        = True,
        ) 
    text_offsets = response.choices[0]['logprobs']['text_offset']
    cutIndex = text_offsets.index(max(i for i in text_offsets if i < len(initialSequence))) + 1
    endIndex = response.usage.total_tokens
    answerTokens = response.choices[0]["logprobs"]["tokens"][cutIndex:endIndex]
    answerTokenLogProbs = response.choices[0]["logprobs"]["token_logprobs"][cutIndex:endIndex] 
    meanAnswerLogProb = np.mean(answerTokenLogProbs)
    sentenceLogProb = np.sum(answerTokenLogProbs)

    return meanAnswerLogProb, sentenceLogProb, (endIndex - cutIndex)

def sampleContinuation(initialSequence, topk = 1, max_tokens = 32, preface = ''):
    """
    Helper for sampling predicted responses given prompts.
    """
    initialSequence = preface + initialSequence
    response = openai.Completion.create(
            engine      = "text-davinci-003", 
            prompt      = initialSequence,
            max_tokens  = max_tokens,
            temperature = 1, 
            logprobs    = 0,
            echo        = True,
            n           = topk,
        ) 
    print("response ", response)
    answers = []
    probs = []
    for i in range(topk):
        text_offsets = response.choices[i]['logprobs']['text_offset']
        cutIndex = text_offsets.index(max(i for i in text_offsets if i < len(initialSequence))) + 1
        try:
            endIndex = response.choices[i]["logprobs"]["tokens"].index("<|endoftext|>")
        # cases when eos token was not predicted
        except ValueError:
            print("No eos token ")
            endIndex = text_offsets.index(text_offsets[-1]) + 1
        answerTokens = response.choices[i]["logprobs"]["tokens"][cutIndex:endIndex]
        answerTokenLogProbs = response.choices[i]["logprobs"]["token_logprobs"][cutIndex:endIndex] 
        meanAnswerLogProb = np.mean(answerTokenLogProbs)
        probs.append(meanAnswerLogProb)
        answerSentence = "".join(answerTokens).replace("\n", "").strip()
        answers.append(answerSentence)
        print("AnswerSentence ", answerSentence)
    return answers, probs


def processItem(item, wait = 0, preface = ''):
    # colnames_e1 = ["taciturn", "competitor", "sameCategory1", "sameCategory2", "sameCategory3", "otherCategory", "fullList_0", "fullList_1", "fullList_2", "fullList_3", "fullList_4", "fullList_5"]
    colnames = ["taciturn", "competitor", 'mostSimilar', "otherCategory", 'sameCategory_0', 'sameCategory_1', 'sameCategory_2', 'sameCategory_3',
       'sameCategory_4', 'sameCategory_5', 'sameCategory_6', 'sameCategory_7',
       'sameCategory_8', 'sameCategory_9', 'sameCategory_10',
       'sameCategory_11', 'sameCategory_12', 'fullList_0', 'fullList_1',
       'fullList_2', 'fullList_3', 'fullList_4', 'fullList_5', 'fullList_6',
       'fullList_7', 'fullList_8', 'fullList_9', 'fullList_10', 'fullList_11',
       'fullList_12', 'fullList_13', 'fullList_14', 'fullList_15',
       'fullList_16', 'fullList_17', 'fullList_18', 'fullList_19',
       'fullList_20', 'fullList_21', 'fullList_22', 'fullList_23']
    probs = {k: 0 for k in colnames}
    ppls = {k: 0 for k in colnames}
    exception_counter = {k: 0 for k in colnames}
    
    for a in colnames:
        # caqtch potential request timeouts
        for _ in range(2):
            try:
                prob, sent_ll, seq_length = getLogProbContinuation(item.context, item[a], preface)
                probs[a] = np.exp(prob)
                ppls[a] = np.exp(-sent_ll/seq_length)
                time.sleep(5)
                break
            except openai.error.ServiceUnavailableError:
                print("OpenAI connection error")
                exception_counter[a] += 1
                time.sleep(20)
                continue

        

    if any([v > 1 for v in list(exception_counter.values())]):
        raise ValueError("some probs were not computed!")

    # include averaging over different permutations
    avg_probs = [probs["taciturn"],
        probs["competitor"],
        probs["mostSimilar"],
        np.mean([probs[k] for k in [c for c in colnames if c.startswith("sameCategory")]]),
        probs["otherCategory"],
        np.mean([probs[k] for k in [c for c in colnames if c.startswith("fullList")]])
    ]
    # also compute perplexities
    avg_ppls = [ppls["taciturn"],
        ppls["competitor"],
        ppls["mostSimilar"],
        np.mean([ppls[k] for k in [c for c in colnames if c.startswith("sameCategory")]]),
        ppls["otherCategory"],
        np.mean([ppls[k] for k in [c for c in colnames if c.startswith("fullList")]])
    ]
    probs = avg_probs/np.sum(avg_probs)
    
    results = pd.DataFrame({
        "itemName": [item.itemName] * 6,
        "answer_type": ["taciturn", "competitor", "mostSimilar", "sameCategory", "otherCategory", "fullList"],
        "answer_type_prob_avg": probs,
        "answer_type_ppl": avg_ppls,
    }, index = [item.itemName]*6)
    
    # write out by item results, in case API stops responding
    results.to_csv(f"GPT3-davinci-003-predictions-e2-{item.itemName}.csv")
    time.sleep(wait) # to prevent overburdening free tier of OpenAI
    return(results)
    
def sampleAnswersForItem(item, wait = 0, preface = '', topk = 1, max_tokens = 32):
    """
    Helper for sampling formatting the responses.
    """
    answers = []
    probs = []
    answer, prob = sampleContinuation(item.context_fct_prompt, preface=preface, topk=topk, max_tokens=max_tokens)
    answers.append(answer)
    probs.append(prob)
    results = pd.DataFrame(
        {
        "itemName"     : item.itemName,
        "predictions"  : answers,
        "probs"        : probs,  
        }, index = [item.itemName]
    )
    # flatten df in case more than one answer was sampled
    results = results.explode(["predictions", "probs"])
    # also save each item for the case of time outs
    results.to_csv(f"GPT3-davinci-003-samples-e2-{item.itemName}.csv")
    time.sleep(wait) # to prevent overburdening free tier of OpenAI
    return results    

if __name__ == "__main__":
    # parse cmd args
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--task", help = "which response should be retrieved from GPT3?", choices = ["probs", "samples"])
    parser.add_argument("-os", "--one_shot", help = "one shot or zero shot context for GPT3?", action="store_true")
    parser.add_argument("-n", "--num_samples", help = "number of samples to draw (WARNING: high credit cost)", nargs="?", default=1, type = int)
    parser.add_argument("-m", "--max_tokens", help = "maximal number of tokens to sample for each response (WARNING: high credit cost)", nargs="?", default=32, type = int)
    
    args = parser.parse_args()
    # run scoring
    if args.task == "probs": 
        # one-shot vs zero shot
        if args.one_shot:
            # don't forget to use the appropriate prompt
            results_oneShotLearner = pd.concat([processItem(items.loc[i], wait = 40, preface = oneShotExampleE2) for i in range(len(items))])
            results_oneShotLearner.to_csv('GPT3-davinci-003-predictions-oneShotLearner-e2.csv', index = False)
        else:
            results_zeroShotLearner = pd.concat([processItem(items.loc[i], wait = 40, preface = "") for i in range(len(items))])
            results_zeroShotLearner.to_csv('GPT3-davinci-003-predictions-overinfo-zeroShotLearner-e2.csv', index = False)
    # run sampling
    elif args.task == "samples":
        # one shot
        if args.one_shot:
            # don't forget to use the appropriate prompt
            samples_oneShotLearner = pd.concat([sampleAnswersForItem(items.loc[i], wait = 45, preface = oneShotExampleE2, topk=args.num_samples, max_tokens=args.max_tokens) for i in range(len(items))])
            samples_oneShotLearner.to_csv(f'GPT3-davinci-003-samples-oneShotLearner-e2.csv', index = False)
        # vs zero shot
        else:
            
            samples_zeroShotLearner = pd.concat([sampleAnswersForItem(items.loc[i], wait = 45, preface = "", topk=args.num_samples, max_tokens=args.max_tokens) for i in range(len(items))])
            samples_zeroShotLearner.to_csv('GPT3-davinci-002-samples-zeroShotLearner-e2.csv', index = False)

    else:
        raise ValueError("Unknown task type")