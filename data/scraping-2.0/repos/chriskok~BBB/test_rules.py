# import nltk
# import re
# import pandas as pd
# import time

# # from nltk.tokenize import word_tokenize
# # from nltk.wsd import lesk
# # from nltk.corpus import wordnet as wn
# # from nltk.corpus import stopwords
# # stemmer = nltk.stem.PorterStemmer()
# # from itertools import product

# # from pywsd.lesk import simple_lesk
# # from pywsd import disambiguate
# # from pywsd.similarity import max_similarity as maxsim

# from blocks.models import *
# import building_blocks as bb 
# import openai
# from my_secrets import my_secrets
# openai_key = my_secrets.get('openai_key')
# openai.api_key = openai_key


# ================================== #
#          KEYWORD MATCHING          #
# ================================== #

# sent1 = "Because then we can them in different areas, rather than hardcoding them to one area."
# sent2 = "They can help create designs that will be consistent across different pages or projects."
# sent3 = "it creates a consistent interface that can be easily changed instead of changing each individual instance manually."

# sent4 = "We want to execute some instructions without blocking other lines of code."
# sent5 = "The bible represents christianity's moral code."
# sent6 = "Here is my secret code!"
# sent7 = "Did you write this Python code?"

# print(lesk(sent1, 'reuse'))
# print(lesk(sent2, 'consistent'))
# print(lesk(sent3, 'manual'))
# print(lesk(sent3, 'manually'))

# print(lesk(sent4, 'code').definition())
# print(lesk(sent5, 'code').definition())
# print(lesk(sent6, 'code').definition())
# print(lesk(sent7, 'code').definition())

# print(simple_lesk(sent4, 'code').definition())
# print(simple_lesk(sent5, 'code').definition())
# print(simple_lesk(sent6, 'code').definition())
# print(simple_lesk(sent7, 'code').definition())

# print(disambiguate(sent4, algorithm=maxsim, similarity_option='wup', keepLemmas=True))
# # print(disambiguate(sent5, algorithm=maxsim, similarity_option='wup', keepLemmas=True))
# print(disambiguate(sent6, algorithm=maxsim, similarity_option='wup', keepLemmas=True))
# print(disambiguate(sent7, algorithm=maxsim, similarity_option='wup', keepLemmas=True))

# # for ss in wn.synsets('consistent'):
# #     print(ss, ss.definition())


# allsyns1 = set(ss for word in sent1.split(' ') for ss in wn.synsets(word))
# allsyns2 = set(ss for word in sent2.split(' ') for ss in wn.synsets(word))
# # allsyns1 = set([lesk(sent1, 'reuse')])
# # allsyns2 = set([lesk(sent2, 'consistent')])
# # best = max((wn.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in product(allsyns1, allsyns2))
# # print(f"BEST: {best}")
# # print(list((wn.wup_similarity(s1, s2) or 0, s1, s2, s1.definition(), s2.definition()) for s1, s2 in product(allsyns1, allsyns2) if wn.wup_similarity(s1, s2) > 0.7))
# print(list((wn.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in product(allsyns1, allsyns2) if wn.wup_similarity(s1, s2) > 0.7))

# ================================== #
#          SENTENCE MATCHING         #
# ================================== #

# ori_sentence_set = ['We can change a Google page without reloading it and send, request, and receive data from an AWS server without blocking the rest of your interface.', 
#                 'We want to be able to make changes to a Google page without reloading it every time. We also want to send, request, and receive data from an AWS server without blocking the interface.', 
#                 'It changes the Google page without reloading it and can send, request and receive data without blocking the rest of the interface']

# negative_sentence_set = ['This is to allow other components when showing the webpage does not get blocked by operations that require long time.',]
# # negative_sentence_set = ['This allows for multiple tasks to run at the same time.',
# #                          'This is to allow other components when showing the webpage does not get blocked by operations that require long time.',
# #                          'This way you dont have to wait for things to happen to update other separate parts of a webpage/app.']

# def process_sentences(ori_sentence_set, verbose=False):
#     # clean sentences (lowercase, remove punctuation)
#     sentence_set = [re.sub(r'[^\w\s]', '', sentence.lower()) for sentence in ori_sentence_set]

#     # remove stopwords with nltk
#     stop_words = set(stopwords.words('english'))
#     sentence_set = [' '.join([word for word in sentence.split() if word not in stop_words]) for sentence in sentence_set]

#     # identify all common words in the sentences
#     common_words = set.intersection(*map(set, map(str.split, sentence_set)))
#     if(verbose): print(f"common_words: {common_words}")

#     # get stems of words in common_words
#     stemmed_common_words = [stemmer.stem(word) for word in common_words]
#     if(verbose): print(f"stemmed_common_words: {stemmed_common_words}")

#     # get synonyms of words in common_words
#     synonyms = []
#     for word in common_words:
#         for syn in wn.synsets(word):
#             for l in syn.lemmas():
#                 synonyms.append(l.name())
#     synonyms = set(synonyms)
#     if(verbose): print(f"synonyms: {synonyms}")

#     # get named entities in the sentences
#     named_entities = []
#     for sentence in ori_sentence_set:
#         for chunk in nltk.ne_chunk(nltk.pos_tag(word_tokenize(sentence))):
#             if hasattr(chunk, 'label'):
#                 named_entities.append((' '.join(c[0] for c in chunk), chunk.label()))
#     named_entities = set(named_entities)
#     named_entities = dict((x, y) for x, y in named_entities)
#     if(verbose): print(f"named_entities: {named_entities}")

#     return sentence_set, common_words, stemmed_common_words, synonyms, named_entities

# # recursively process the next string in the list
# patterns = {}
# pattern_limit = 3

# def add_pattern(pattern, depth, negative=False):
#     if pattern in patterns:
#         patterns[pattern] += 1 * (depth * 0.5) if not negative else -2 * (depth * 0.5)
#     else:
#         patterns[pattern] = 1 * (depth * 0.5) if not negative else -2 * (depth * 0.5)

# def process_next_string(word, tag, pos_set, ori_i, i, current_pattern):
    
#     # stop if we've reached the end of the list
#     if i == len(pos_set) - 1:
#         return current_pattern
    
#     # stop if we've reached the pattern limit
#     curr_depth = i - ori_i + 1
#     if curr_depth > pattern_limit:
#         return current_pattern
    
#     word_l = word.lower()
    
#     # check if word in common words (or a stem of that)
#     if stemmer.stem(word_l) in stemmed_common_words:
#         new_pattern_and = f"[{word_l}]" if current_pattern == "" else current_pattern + "+" + f"[{word_l}]"
#         new_pattern_or = f"[{word_l}]" if current_pattern == "" else current_pattern + "|" + f"[{word_l}]"
#         for pattern in [new_pattern_and, new_pattern_or]:
#             add_pattern(pattern, curr_depth)
#             process_next_string(pos_set[i + 1][0], pos_set[i + 1][1], pos_set, ori_i, i + 1, pattern)
    
#     if stemmer.stem(word_l) in n_stemmed_common_words:
#         new_pattern_and = f"[{word_l}]" if current_pattern == "" else current_pattern + "+" + f"[{word_l}]"
#         new_pattern_or = f"[{word_l}]" if current_pattern == "" else current_pattern + "|" + f"[{word_l}]"
#         for pattern in [new_pattern_and, new_pattern_or]:
#             add_pattern(pattern, curr_depth, negative=True)
#             process_next_string(pos_set[i + 1][0], pos_set[i + 1][1], pos_set, ori_i, i + 1, pattern)

#     # check if word is a synonym of a common word
#     if word_l in synonyms:
#         new_pattern_and = f"({word_l})" if current_pattern == "" else current_pattern + "+" + f"({word_l})"
#         new_pattern_or = f"({word_l})" if current_pattern == "" else current_pattern + "|" + f"({word_l})"
#         for pattern in [new_pattern_and, new_pattern_or]:
#             add_pattern(pattern, curr_depth)
#             process_next_string(pos_set[i + 1][0], pos_set[i + 1][1], pos_set, ori_i, i + 1, pattern)

#     if word_l in n_synonyms:
#         new_pattern_and = f"({word_l})" if current_pattern == "" else current_pattern + "+" + f"({word_l})"
#         new_pattern_or = f"({word_l})" if current_pattern == "" else current_pattern + "|" + f"({word_l})"
#         for pattern in [new_pattern_and, new_pattern_or]:
#             add_pattern(pattern, curr_depth, negative=True)
#             process_next_string(pos_set[i + 1][0], pos_set[i + 1][1], pos_set, ori_i, i + 1, pattern)

#     # check if word is a named entity
#     if word in named_entities:
#         new_pattern_and = f"({word})" if current_pattern == "" else current_pattern + "+" + f"({word})"
#         new_pattern_or = f"({word})" if current_pattern == "" else current_pattern + "|" + f"({word})"
#         new_pattern_label_and = f"${named_entities[word]}" if current_pattern == "" else current_pattern + "+" + f"${named_entities[word]}"
#         new_pattern_label_or = f"${named_entities[word]}" if current_pattern == "" else current_pattern + "|" + f"${named_entities[word]}"
#         for pattern in [new_pattern_and, new_pattern_or, new_pattern_label_and, new_pattern_label_or]:
#             add_pattern(pattern, curr_depth)
#             process_next_string(pos_set[i + 1][0], pos_set[i + 1][1], pos_set, ori_i, i + 1, pattern)

#     if word in n_named_entities:
#         new_pattern_and = f"({word})" if current_pattern == "" else current_pattern + "+" + f"({word})"
#         new_pattern_or = f"({word})" if current_pattern == "" else current_pattern + "|" + f"({word})"
#         new_pattern_label_and = f"${named_entities[word]}" if current_pattern == "" else current_pattern + "+" + f"${named_entities[word]}"
#         new_pattern_label_or = f"${named_entities[word]}" if current_pattern == "" else current_pattern + "|" + f"${named_entities[word]}"
#         for pattern in [new_pattern_and, new_pattern_or, new_pattern_label_and, new_pattern_label_or]:
#             add_pattern(pattern, curr_depth, negative=True)
#             process_next_string(pos_set[i + 1][0], pos_set[i + 1][1], pos_set, ori_i, i + 1, pattern)

#     # check if word
#     # if tag in patterns:
#     #     patterns[tag].append(word)
#     # else:
#     #     patterns[tag] = [word]

# # identify all parts of speech in the sentences
# pos_set = [nltk.pos_tag(word_tokenize(sentence)) for sentence in ori_sentence_set]
# sentence_set, common_words, stemmed_common_words, synonyms, named_entities = process_sentences(ori_sentence_set)
# n_sentence_set, n_common_words, n_stemmed_common_words, n_synonyms, n_named_entities = process_sentences(negative_sentence_set)

# # iterate through the pos_set to build patterns for matching
# for pos in pos_set:
#     for i, (word, tag) in enumerate(pos):
#         process_next_string(word, tag, pos, i, i, "")

# # sort patterns dictionary by value
# patterns = {k: v for k, v in sorted(patterns.items(), key=lambda item: item[1], reverse=True)}
# print(patterns)

# ================================== #
#         SENTENCE COMPARISONS       #
# ================================== #

# def prompt_chatgpt(prompt):
#     model="gpt-3.5-turbo"
#     try:
#         response = openai.ChatCompletion.create(
#             model=model,
#             messages=prompt,
#             temperature=0,
#         )
#         if "error" in response:
#             print("OPENAI ERROR: {}".format(response))
#             return "ERROR"
#         else:
#             return response["choices"][0]["message"]["content"]
#     except Exception as e: 
#         print(e)
#         return "ERROR"

# def classify_with_rubric(question, answers, rubric, all_rubrics):
#     answers_string = '\n'.join([f"#{i+1}: {e}" for i, e in enumerate(answers)])
#     all_rubrics_string = '\n'.join(all_rubrics)
#     prompt = [
#             {"role": "system", "content": f"You are an expert teacher in a class, you have the following question in your final exam: {question.question_text}. You are currently evaluating which of the answers match the following rubric: {rubric}. This is the full list of rubrics: \n\n{all_rubrics_string}"},
#             # {"role": "user", "content": f"For each of the following students' answers (formatted as such: #<number>: <answer>) please state if the rubric applies to it. Please be conservative by only saying Yes if this is the MOST relevant rubric amongst the list of rubrics. For each answer, strictly follow the output format: \'#<number>: <Y/N>\' where Y is Yes and N is No.\n\n{answers_string}"},
#             {"role": "user", "content": f"For each of the following students' answers (formatted as such: #<number>: <answer>) please score it from a scale of 0.0 to 1.0 based on how applicable the rubric is to it. Please be conservative and take all the other rubrcs into account from the list of rubrics. For each answer, strictly follow the output format: \'#<number>: <score>\'\n\n{answers_string}"},
#         ]
#     chatgpt_response = prompt_chatgpt(prompt)
#     return chatgpt_response

# def convert_chatgpt_response(chatgpt_response):
#     # convert numbered list in string to list of strings
#     rubric_classifications = [x.strip().split(': ')[1] for x in chatgpt_response.split('\n') if x.strip() != '']
#     return rubric_classifications

# def get_rubric_classifications(question, answers, rubric, all_rubrics):
#     answers_list = answers.values_list('answer_text', flat=True)
#     answers_id_list = answers.values_list('id', flat=True)
#     all_classifications = []
#     # iterate through answers and add to list if they are under 8000 characters total
#     curr_length = 0
#     curr_answers = []
#     for index, ans in enumerate(answers_list):
#         if (curr_length + len(ans) < 10000):
#             curr_answers.append(ans)
#             curr_length += len(ans)
#         else:
#             all_classifications.extend(convert_chatgpt_response(classify_with_rubric(question, curr_answers, rubric, all_rubrics)))
#             # TODO: Asynchronous execution of OpenAI API calls
#             print(f"Index: {index} - Added {len(curr_answers)} answers")
#             curr_answers = [ans]
#             curr_length = len(ans)
#             time.sleep(15) # wait avoid OpenAI API rate limit
    
#     all_classifications.extend(convert_chatgpt_response(classify_with_rubric(question, curr_answers, rubric, all_rubrics)))  # final time, with remaining answers
#     if (len(all_classifications) == len(answers_list)):
#         return all_classifications
#     else:
#         print("ERROR: Number of concept maps does not match number of answers")
#         print(all_classifications)
#         print(len(all_classifications))
#         return None

# def produce_comparisons(chosen_answers, chosen_question, sentence, index, all_rubrics):
#     # filters answers 
#     df = pd.DataFrame(list(chosen_answers.values()))
#     # apply existing sentence similarity methods
#     methods = ['sbert', 'spacy', 'tfidf']
#     sim_scores = [0.5, 0.6, 0.7, 0.8]
#     for method in methods:
#         for sim_score in sim_scores:
#             new_df = bb.similar_sentence(df, sentence, sim_score_threshold=sim_score, method=method)
#             # print(f"evaluating: {method} @ {sim_score} -> {new_df.shape}")
#             # add column to df: True if row in new_df, False otherwise
#             df[f"{method}_{sim_score}"] = df["answer_text"].isin(new_df["answer_text"])
#             # print(df[f"{method}_{sim_score}"].value_counts())  # checking if the column is added correctly
#     # apply chatgpt rubric checking
#     chatgpt_rubrics = get_rubric_classifications(chosen_question, chosen_answers, sentence, all_rubrics)
#     if (chatgpt_rubrics): df['chatgpt_classified_rubric'] = chatgpt_rubrics
#     df.to_csv(f"results/sentencesim_comparisons/Q{chosen_question.id}_R{index+1}.csv")
#     print(f"Completed: Question - {chosen_question.id}, Rubric #{index+1} - {sentence}")

# q7_rubrics = ["Allow user interaction at any time", "Allow query data from server/API without disrupting user flow", "Render content on the webpage in real-time", "Javascript is single-threaded"]
# q27_rubrics = ["Clearly states the reason of consistency and reusability/efficiency", "Clearly states the reason of consistency only", "Clearly states the reason of reusability/efficiency only", "Does not explicitly state either reason but is somewhat correct."]
# # get all answers for a question
# chosen_answers = Answer.objects.filter(question_id=37).order_by('outlier_score')
# chosen_question = Question.objects.get(id=37)
# for index, rubric in enumerate(q27_rubrics):
#     produce_comparisons(chosen_answers, chosen_question, rubric, index, q27_rubrics)


# ================================== #
#         LANGCHAIN + RUBRICS        #
# ================================== #

# from langchain.llms import OpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# import os

# import openai
# from my_secrets import my_secrets
# openai_key = my_secrets.get('openai_key')
# openai.api_key = openai_key
# os.environ["OPENAI_API_KEY"] = openai_key

# # This is an LLMChain to write a synopsis given a title of a play.
# llm = OpenAI(temperature=.7)
# template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.

# Title: {title}
# Playwright: This is a synopsis for the above play:"""
# prompt_template = PromptTemplate(input_variables=["title"], template=template)
# synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)
# # This is an LLMChain to write a review of a play given a synopsis.
# llm = OpenAI(temperature=.7)
# template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

# Play Synopsis:
# {synopsis}
# Review from a New York Times play critic of the above play:"""
# prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
# review_chain = LLMChain(llm=llm, prompt=prompt_template)

# # This is the overall chain where we run these two chains in sequence.
# from langchain.chains import SimpleSequentialChain
# overall_chain = SimpleSequentialChain(chains=[synopsis_chain, review_chain], verbose=True)

# review = overall_chain.run("Tragedy at sunset on the beach")


# ================================== #
#          CHATGPT FUNCTIONS         #
# ================================== #

import openai
from my_secrets import my_secrets
openai_key = my_secrets.get('openai_key')
openai.api_key = openai_key
import json

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)

# Step 1, send model the user query and what functions it has access to
def run_conversation():
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
        functions=[
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        ],
        function_call="auto",
    )

    message = response["choices"][0]["message"]

    # Step 2, check if the model wants to call a function
    if message.get("function_call"):
        function_name = message["function_call"]["name"]

        # Step 3, call the function
        # Note: the JSON response from the model may not be valid JSON
        function_response = get_current_weather(
            location=message.get("location"),
            unit=message.get("unit"),
        )

        # Step 4, send model the info on the function call and function response
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[
                {"role": "user", "content": "What is the weather like in boston?"},
                message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                },
            ],
        )
        return second_response

print(run_conversation())