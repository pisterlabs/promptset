import tensorflow_hub as hub
import openai
import copy
import time
from Researcher.Relevance import embed_sentences
from Researcher.Relevance import compute_similarity
from Researcher.LinkSearch import LinkSearch
from Researcher.crossRef_Search import CrossRef_Search
from saveHistory import initializeHistory, loadHistory
#from Researcher.kw import step_processor



openai.api_key = "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"


def OpenaiHandshake(chain):
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=chain,
            temperature=0.8,
            presence_penalty=1
        )
    response_m = response.choices[0].message
    return (response_m, response_m["content"])

def articles_search(step, num=10, budget=3000):
    result = LinkSearch(step, num, budget)
    num_try = 0
    while len(result)==0 and num_try<=6:
        print("fails to find articles, try again")
        result = LinkSearch(step, num, budget)
        num_try+=1
    return result  #expected[{"source":xx, "content":["abstract", "1th paragraph", "2th paragraph", "3th paragraph"...]}] of length num#
                   #LinkSearch written by Jason outputs [{"source":, "content":}]#

def collect_paragraphs(articles):
    total_paragraphs_list = []
    for article in articles:
        total_paragraphs_list += article["content"]
    return total_paragraphs_list

def similarity_score(total_embeddings):
    scores = []
    for index in range(0,len(total_embeddings)-1): #take embeddings of paragraph instead of step#
        scores.append(compute_similarity(total_embeddings[index],total_embeddings[-1]))
    return scores


def semantic_finder(step): #step is in str#
    #step = step_processor(step)
    articles = articles_search(step)
    num_paragraph_per_article = [len(article["content"]) for article in articles] #save number of paragraphs for article for later information retrieve#
    paragraphs_list_from_articles = collect_paragraphs(articles)
    paragraphs_plus_query_list = paragraphs_list_from_articles + [step]
    total_embeddings = embed_sentences(paragraphs_plus_query_list) #[embed1, embed2, ..., embedstep]#
    scores = similarity_score(total_embeddings)
    scores_mean = sum(scores)/len(scores)
    return  (scores_mean, scores, articles, num_paragraph_per_article) #float, list, list, list#

def chain_initializer(state, key_word):
    if key_word == "empty":
        chain = []
        chain.append({"role":"system", "content":system_prompt.format(topic=state["topic"], question=state["question"])})
    else:
        num_steps = len(list(state.keys()))-2
        chain = []
        chain.append({"role":"system", "content":system_prompt.format(topic=state["topic"], question=state["question"])})
        for index in range(1, num_steps+1):
            chain.append({"role":"user", "content":user_prompt_step.format(step_order="{}th step".format(index), step_content=state["{}th step".format(index)])})
    return chain


def step_candidates_generator(state, num_candidates,next_step_prompt): #states:{"topic":, "question":, "1th step":, "2th step":, ...} every elememt is in string# 
    if len(list(state.keys())) > 2:
        step_candidates = []
        while len(step_candidates) != num_candidates:
            chain = chain_initializer(state, "")
            chain.append({"role":"user", "content":next_step_prompt})
            assistant, step_candidate = OpenaiHandshake(chain)
            step_candidates.append(step_candidate)
        return step_candidates #list#
    else:
        step_candidates = []
        while len(step_candidates) < num_candidates:
            chain = chain_initializer(state, "empty")
            chain.append({"role":"user", "content":kick_start_prompt_beginning})
            assistant, step_candidate = OpenaiHandshake(chain)
            step_candidates.append(step_candidate)
        return step_candidates

def semantic_filter(step_candidates, num_selections): #steps = [step candidate 1, step candidate 2, ...] num_selections = number of candidates you want to select #
    semantic_finder_output_list = []
    for step_candidate in step_candidates:
        dict = {}
        scores_mean, scores, articles, num_paragraph_per_article = semantic_finder(step_candidate)
        dict["step candidate"] = step_candidate
        dict["candidate score"] = scores_mean
        dict["paragraph scores"] = scores
        dict["articles"] = articles
        dict["num_paragraph_per_article"] = num_paragraph_per_article
        semantic_finder_output_list.append(dict)
    
    if num_selections < len(step_candidates):
        selected_candidates_indexes = [i for i in range(len(step_candidates))]
    else:  
        score_of_candidates = [step_candidate["candidate score"] for step_candidate in semantic_finder_output_list]
        selected_candidates_indexes = sorted(range(len(score_of_candidates)), key=lambda i: score_of_candidates[i])[-num_selections:]
    selected_candidates = [semantic_finder_output_list[selected_candidate_index] for selected_candidate_index in selected_candidates_indexes]
    return selected_candidates

def paragraph_retrieval(paragraph_scores, articles, num_paragraph_per_article, num_paragraphs_selected=10):
    selected_paragraphs = []
    selected_paragraphs_indexes = sorted(range(len(paragraph_scores)), key=lambda i: paragraph_scores[i])[-num_paragraphs_selected:]
    for index in selected_paragraphs_indexes:
        article_index = 0
        article_paragraph_combined_length = num_paragraph_per_article[article_index] 
        combined_length_before = 0
        while article_paragraph_combined_length < index+1:
            article_index += 1
            combined_length_before = article_paragraph_combined_length
            article_paragraph_combined_length += num_paragraph_per_article[article_index]
        selected_paragraphs.append({"source":articles[article_index]["source"], "information":articles[article_index]["content"][index-combined_length_before]})
    return selected_paragraphs

def gpt_evaluation(potential_new_steps, state): #list#
    if len(list(state.keys())) > 2:
        chain = chain_initializer(state, "")
    else:
        chain = chain_initializer(state, "empty")
    chain.append({"role":"user", "content":user_prompt_gpt_evaluation.format(potential_next_steps=str(potential_new_steps))})
    chain.append({"role":"user", "content":kick_start_prompt_vote})
    assistant, vote_ = OpenaiHandshake(chain)
    vote = eval(vote_)
    return vote

def state_append(state,append_state):
    num = len(list(state.keys()))
    new_state = state.copy()
    new_state["{}th step".format(num-1)] = append_state
    return new_state

def state_update(state, selected_candidates, num_next_step=1):
    potential_new_steps = []
    num_selected_candidates = len(selected_candidates)
    for selected_candidate in selected_candidates:
        selected_candidate_step = selected_candidate["step candidate"]
        paragraph_scores = selected_candidate["paragraph scores"]
        articles = selected_candidate["articles"]
        num_paragraph_per_article = selected_candidate["num_paragraph_per_article"]
        selected_paragraphs = paragraph_retrieval(paragraph_scores, articles, num_paragraph_per_article)
        if len(list(state.keys())) > 2:
            chain = chain_initializer(state, "")
        else: 
            chain =chain_initializer(state, "empty")
        chain.append({"role":"user", "content":user_prompt_state_update.format(step=selected_candidate_step, information=str(selected_paragraphs))})
        chain.append({"role":"user", "content":kick_start_prompt_state_update})
        assistant, edited_selected_candidate_step = OpenaiHandshake(chain)
        potential_new_steps.append(edited_selected_candidate_step)
    returned_state = []
    if num_next_step >= num_selected_candidates:
        for next_step in potential_new_steps:
            new_state = state_append(state, next_step)
            returned_state.append(new_state)
        return returned_state 
    else:
        vote = gpt_evaluation(potential_new_steps, state)
        selection_index = sorted(range(len(vote)), key=lambda i: vote[i])[-num_next_step:]
        for index in selection_index:
            new_state = state_append(state, potential_new_steps[index])
            returned_state.append(new_state)
        return returned_state

def queryBlock_sub(initial_state, breadth_of_search, num_first_round, num_second_round, num_search, num_max_retry):
    global retry_count
    #Initialization#
    count = 1
    states = initial_state
    while count <= num_search+1:
        indicator = False
        start_time = time.time()
        new_states = []
        if count <= num_search:
            next_step_prompt = kick_start_prompt
        else:
            next_step_prompt = end_plan_prompt
        try:
            for state in states:
                step_candidates = step_candidates_generator(state,num_first_round,next_step_prompt)
                time_used_step_candidate = time.time()-start_time

                start_time = time.time()
                selected_candidates = semantic_filter(step_candidates, num_second_round)
                time_used_selected_candidates = time.time()-start_time

                start_time = time.time()
                new_state = state_update(state, selected_candidates, breadth_of_search)
                time_used_state_update = time.time()-start_time

                new_states += new_state
            time_used = time.time()-start_time
        except:
            print("-------------sctore break point-----------------------------------")
            cache = states
            indicator = True
            break 
        print("""----------------------------------{}th round complete with time used ["time_used_step_candidate": {}s, "time_used_selected_candidates": {}s, "time_used_state_update": {}s]------------------------""".format(count, time_used_step_candidate, time_used_selected_candidates, time_used_state_update))
        print(states)
        states = new_states
        count += 1    

    if indicator and retry_count<num_max_retry:
        initial_state = cache
        print("--------------successfully goes back to the last break point!------------------------")
        start_count = count
        retry_count+=1
        queryBlock_sub(initial_state, breadth_of_search, num_first_round, num_second_round, start_count, num_max_retry)
    elif indicator and retry_count==num_max_retry:
        print("Reach max num of retry!")
        return cache
    else: #Output remaining steps#
        return states

def queryBlock(user_id, initial_state, breadth_of_search=1, num_first_round=4, num_second_round=2, num_search=6, num_max_retry=50, history_initialization=False):
    global retry_count
    if history_initialization:
        initializeHistory()
    retry_count = 0
    states = queryBlock_sub(initial_state, breadth_of_search, num_first_round, num_second_round, num_search, num_max_retry)
    print("----------------Output States----------------------")
    print(states)
    plan_time = time.time()
    user_id = user_id
    loadHistory(user_id, plan_time, states)
#history [{"id":, "time":, "state":state}]#







#Prompting-----------------------------------------------------------------------------------------------------------------------------------------#
system_prompt = """
You are an experienced researcher. A user conducts a research under topic {topic}. During research, he finds some problem 
and ask you for help. Here is his question {question}. What you are supposed to do is help him design a detailed plan step by step 
to solve his problem. You are not allowed to say anything unless be required, and please strictly follow later instructions.
"""

user_prompt_step = """
{step_order}: {step_content}
"""

user_prompt_state_update = """
After setting up above steps, the user comes up with the next step: {step}. He finds some information which maybe helpful for providing 
more details for this new step. The information takes the format [{{"source": , "information": }},{{"source": , "information": }}, ...]. Your just is 
to edit this new step by using provided information by user. Each editing parts should end with {{"source": , "information": }} that you used. 
Here is the information user sent to you {information}.
"""

user_prompt_gpt_evaluation = """
After setting up above steps, the user now has several potential next step displayed in the list he will later send to you.
Your job is to evaulaute each possible next step one by one based on feasibility, amount of detail and logic coherence with steps already setting up. 
Follow below reasoning step by step:
Firstly, the user sends you his topic, question, and previous setting up steps and a list containing k number of possible next steps. Because it contains k possible next steps,
you need to output a list containing a number of k integers. Each integer in the output list represents how many votes you give to corresponding potential next step based on feasibility, amount of detail and logic coherence with steps already setting up. .
Finally, you must onlys generate the list "[vote numbers]", and nothing else.

Here is the potential next step list {potential_next_steps}.  
"""

end_plan_prompt = """
Please Generate all pf the remaining steps.
"""

kick_start_prompt_beginning = """
Please generate the first step.
"""

kick_start_prompt = """
Above is the user's now available plan. Please generate the next step. 
"""

kick_start_prompt_state_update = """
Please start generates your update to the user's lates step
"""

kick_start_prompt_vote = """
Please start generate your votes.
"""
#Demo------------------------------------------------------------------------------------------------#
#initial_state = [{"topic":"Mathematic Modeling of Diabete within Human Body", "question":"How to model the influence of sensitivity to insulin on glucose concentration?"}]#
#states = demo(initial_state)#
#print(states)#