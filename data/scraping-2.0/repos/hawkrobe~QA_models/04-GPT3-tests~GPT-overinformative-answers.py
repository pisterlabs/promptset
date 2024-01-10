import openai
import time
import numpy as np
import pandas as pd
from dotenv import dotenv_values

# set openAI key in separate .env file w/ content
# OPENAIKEY = yourkey
openai.api_key = dotenv_values('.env')['OPENAIKEY']

items = pd.read_csv('Q&A-stimuli-overinfo - answerTypes.csv')

# preface for GPT3 as a one-shot learner
oneShotExample = '''EXAMPLE:

You are hosting a barbecue party. You are standing behind the barbecue. You have the following goods to offer: pork sausages, vegan burgers, grilled potatoes and beef burgers.

Someone asks: "Do you have grilled zucchini?"

You reason about what that person most likely wanted to have. That they asked for grilled zucchini suggests that they might want vegetarian food. From the items you have pork sausages and beef burgers are least likely to satisfy the persons desires. Vegan burgers and grilled potatoes come much closer. Grilled potatoes are most similar to grilled zucchini.

You reply: "I'm sorry, I don't have any grilled zucchini. But I do have some grilled potatoes."

YOUR TURN:

'''

def getLogProbContinuation(initialSequence, continuation, preface = ''):
    initialSequence = preface + initialSequence
    response = openai.Completion.create(
            engine      = "text-davinci-002", 
            prompt      = initialSequence + " " + continuation,
            max_tokens  = 0,
            temperature = 1, 
            logprobs    = 0,
            # stop        = ".",
            echo        = True
        ) 
    text_offsets = response.choices[0]['logprobs']['text_offset']
    cutIndex = text_offsets.index(max(i for i in text_offsets if i < len(initialSequence))) + 1
    # endIndex = response.choices[0]["logprobs"]["tokens"].index("<|endoftext|>")
    endIndex = response.usage.total_tokens
    answerTokens = response.choices[0]["logprobs"]["tokens"][cutIndex:endIndex]
    answerTokenLogProbs = response.choices[0]["logprobs"]["token_logprobs"][cutIndex:endIndex] 
    meanAnswerLogProb = np.mean(answerTokenLogProbs)
    return(meanAnswerLogProb)

def processItem(item, wait = 0, preface = ''):
    print("processing item: " + item.itemName)
    probs = np.exp(np.array(
        [getLogProbContinuation(item.vignette, item.taciturn, preface),
        getLogProbContinuation(item.vignette, item.competitor, preface),
        getLogProbContinuation(item.vignette, item.sameCategory, preface),
        getLogProbContinuation(item.vignette, item.otherCategory, preface),
        getLogProbContinuation(item.vignette, item.fullList, preface)]
    ))
    probs = probs / np.sum(probs)
    results = pd.DataFrame(
        {
        "itemName"     : item.itemName,
        "taciturn"     : probs[0],
        "competitor"   : probs[1],
        "sameCategory" : probs[2],
        "otherCategory": probs[3],
        "fullList"     : probs[4] 
        }, index = [item.itemName]
    )
    time.sleep(wait) # to prevent overburdening free tier of OpenAI
    return(results)
    

results = pd.concat([processItem(items.loc[i], wait = 30) for i in range(len(items))])
results.to_csv('GPT3-predictions-overinfo.csv', index = False)

results_oneShotLearner = pd.concat([processItem(items.loc[i], wait = 30, preface = oneShotExample) for i in range(len(items))])
results_oneShotLearner.to_csv('GPT3-predictions-overinfo-oneShotLearner.csv', index = False)

# processItem(items.loc[0])
# processItem(items.loc[0], preface = oneShotExample)
# processItem(items.loc[1])
# processItem(items.loc[2])
# processItem(items.loc[3])
# processItem(items.loc[4])
# processItem(items.loc[5])



