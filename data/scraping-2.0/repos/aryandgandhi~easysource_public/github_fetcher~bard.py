from bardapi import Bard
import os
import openai


#edit bard cookie value accordingly
#list index out of range is a bard issue replace the __Secure-1PSID 


def get_bard_summary(url):

    token = 'your_token_goes_here'
    bard = Bard(token=token)
    # openai.api_key = "yourkeyhere"
    if url is not None:
        bard_sum = bard.get_answer("I want to get started working on the following repository. In 5 lines please summarize this repository for me. Give me a brief but extremely informative summary: " + url)['content']
#         openai.Completion.create(
#   model="text-babbage-001",
#   prompt= "Please summarize what the following repository does in 3 sentences to give me the best understanding of it from this general summary: " + bard_sum,
#   max_tokens=30,
#   temperature=0
# )
    else:
        bard_sum = "temporarily not working"
    print(bard_sum)
    return bard_sum




def get_bard_summary_issue(url, issue):
    token = 'your_token_goes_here'
    bard = Bard(token=token)
    if url is not None and issue is not None:
   
        bard_sum_issue = bard.get_answer("Write down the first few steps to solve this: " + issue)['content']

    else:
        bard_sum_issue = "temporarily not working"
    return bard_sum_issue

def contains_word(s, w):
    return (' ' + w + ' ') in (' ' + s + ' ')
    
def get_bard_difficulty(issue):
    token = 'your_token_goes_here'
    ret = ""
    bard = Bard(token=token)
    if issue is not None:
   
        bard_word = bard.get_answer("Based on a beginner programmers knowledge level, if you had to assign " + issue + "one word, medium, easy or hard, output that word. Think very carefully and do your best to give an honest opinion based on the easy, medium and hard rankings. ")['content']
        bard_word = bard_word.lower()


        if contains_word(bard_word, "hard"):
            ret += "hard"

        elif contains_word(bard_word, "medium"):
            ret += "medium"
        else:
            ret += "easy"


    else:
        ret = "temporarily not working"
    return ret