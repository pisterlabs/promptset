from flask import Flask
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, jsonify, Response, current_app
)

import openai
# import textwrap as tw
import re

# from pprint import pprint

bp = Blueprint('home', __name__)


@bp.route('/')
def index():
    return render_template('home.html')


@bp.route('/topic/<string:user_topic>')
def topic(user_topic=None):
    # get explanation
    openai.api_key = current_app.config["API_KEY"]
    prompt = f'For a 5 years old kid, explain what {user_topic} is'
    completion = openai.Completion.create(engine="text-davinci-002", max_tokens=256, prompt=prompt)
    topic_explanation = completion.choices[0].text.strip()

    prompt_metaphor = f'For a 5 years old kid, explain what {user_topic} is with a metaphor'
    metaphor_completion = openai.Completion.create(engine="text-davinci-002", max_tokens=256, prompt=prompt_metaphor)

    metaphor_topic = metaphor_completion.choices[0].text.strip()

    topic_details = {'prompt': prompt, 'topic_explanation': topic_explanation,
                     'prompt_metaphor': prompt_metaphor, 'metaphor_topic': metaphor_topic}

    url_img = None
    if user_topic == 'universe':
        url_img = "https://cdn.mos.cms.futurecdn.net/jfkAU4tM8XMUAPZDm4h5Nh.jpeg"
    elif user_topic == 'light':
        url_img = "https://cdn.mos.cms.futurecdn.net/b32yP4hScsApPYBAEU33NC.jpg"
    elif user_topic == 'history':
        url_img = "https://dualcreditathome.com/wp-content/uploads/2014/02/history.jpg"
    else:
        url_img = "https://miro.medium.com/max/560/1*37K-onRVb3qECgc0h_GOTQ.png"

    return render_template("topic.html", user_topic=user_topic, topic_details=topic_details,
                           url_img=url_img)


@bp.route('/select/<string:user_topic>')
def select(user_topic=None):
    return redirect(url_for('home.topic', user_topic=user_topic))


@bp.route("/subtopic/select", methods=['POST'])
def render_sub_subtopic():
    data_json = request.get_json()
    sub_topic = data_json["subtopic"]
    user_topic = data_json["topic"]

    prompt = f'For a 5 years old kid, explain what {sub_topic} in {user_topic} is with an metaphor.'
    completion = openai.Completion.create(engine="text-davinci-002", max_tokens=256, temperature=0.7, prompt=prompt)

    subtopic_metaphor = completion.choices[0].text.strip()

    prompt_list_subsub = f'''List 8 subtopics related to {sub_topic} in {user_topic} with brief explanation. Each in the form:
        subsubtopic: explanation
        1.
    '''
    completion = openai.Completion.create(engine="text-davinci-002", max_tokens=256, temperature=0.7, prompt=prompt_list_subsub)
    result = completion.choices[0].text.strip()

    list_subsub = re.split(r'\d\. ', result)
    while len(list_subsub) < 8 or ("?" in list_subsub[1]) or (":" not in list_subsub[1]):
        completion = openai.Completion.create(engine="text-davinci-002", max_tokens=256, temperature=0.7, prompt=prompt_list_subsub)
        result = completion.choices[0].text.strip()
        list_subsub = re.split(r'\d\. ', result)
    if list_subsub[0] == '':
        list_subsub = list_subsub[1:]

    # extract pure subsubtopics
    pure_subsub = []
    for subsub_i, subsub_topic in enumerate(list_subsub):
        pure = re.findall(r'(.*):', subsub_topic)[0]
        pure_subsub.append(pure)

    to_return = {'prompt': prompt, 'subtopic_metaphor': subtopic_metaphor,
                 'subsub_prompt': prompt_list_subsub,
                 'subsub_topic': list_subsub,
                 'subsub_pure': pure_subsub}
    return jsonify(to_return)


@bp.route("/subsubtopic/select", methods=["POST"])
def select_subsub_topics():
    data_json = request.get_json()

    subsub_topic = data_json["subsubtopic"]
    sub_topic = data_json["subtopic"]
    user_topic = data_json["user_topic"]

    prompt = f"For {sub_topic} in {user_topic}, explain what {subsub_topic} is with an metaphor for a five years old kid."
    completion = openai.Completion.create(engine="text-davinci-002", max_tokens=256, temperature=0.7, prompt=prompt)

    subsubtopic_explanation = completion.choices[0].text.strip()

    to_return = {'prompt': prompt, 'subsubtopicExplanation': subsubtopic_explanation,
                 'subsubtopic': subsub_topic}
    return jsonify(to_return)


@bp.route("/followupquestion", methods=["POST"])
def follow_up_question():
    data_json = request.get_json()

    subsub_topic = data_json["subsubtopic"]
    sub_topic = data_json["subtopic"]
    user_topic = data_json["user_topic"]
    question = data_json["question"]

    prompt = f"For {subsub_topic} in {sub_topic} in {user_topic}, explain with metaphor for a five years old kid:  \"{question}\""
    completion = openai.Completion.create(engine="text-davinci-002", max_tokens=256, temperature=0.7, prompt=prompt)

    result = completion.choices[0].text.strip()
    to_return = {'question': prompt, 'answer': result}
    return jsonify(to_return)


# # explain topic.html route
# @bp.route('/explain_topic', methods=['GET', 'POST'])
# def explain_topic():
#     openai.api_key = current_app.config["API_KEY"]
#
#     # json_data = request.get_json()
#     global topic
#     topic = request.get_json()['topic.html']
#     prompt = f'For a 5 years old kid, explain what {topic} is'
#     completion = openai.Completion.create(engine="text-davinci-002", max_tokens=256, prompt=prompt)
#
#     topicExplanation = completion.choices[0].text.strip()
#     to_return = {'prompt': prompt, 'topicExplanation': topicExplanation}
#     return jsonify(to_return)


# list subtopics
@bp.route('/list_subtopics/<string:user_topic>', methods=['GET'])
def list_subtopics(user_topic=None):
    openai.api_key = current_app.config["API_KEY"]

    prompt = f'''List 8 subtopics related to {user_topic} with brief explanation. Each in the form:
    subtopic: explanation
    1.
    '''
    completion = openai.Completion.create(engine="text-davinci-002", max_tokens=256, temperature=0.7, prompt=prompt)
    result = completion.choices[0].text.strip()
    # global subtopicList
    subtopic_list = re.split(r'\d\. ', result)
    while len(subtopic_list) < 8:
        completion = openai.Completion.create(engine="text-davinci-002", max_tokens=256, temperature=0.7, prompt=prompt)
        result = completion.choices[0].text.strip()
        subtopic_list = re.split(r'\d\. ', result)
    if subtopic_list[0] == '':
        subtopic_list = subtopic_list[1:]

    # extract pure subtopics
    pure_subtopics = []
    for index_subtopic, each_subtopic in enumerate(subtopic_list):
        pure = re.findall(r'(.*):', each_subtopic)[0]
        pure_subtopics.append(pure)

    to_return = {'prompt': prompt, 'subtopic_list': subtopic_list, 'pure_subtopics': pure_subtopics}
    return jsonify(to_return)
#
#
# # explain subtopic route
# @bp.route('/explain_subtopic', methods=['GET', 'POST'])
# def explain_subtopic():
#     openai.api_key = current_app.config["API_KEY"]
#
#     # json_data = request.get_json()
#     global subtopic
#     subtopicIndex = int(request.get_json()['subtopic'])
#     subtopic = re.split(r':', subtopicList[subtopicIndex - 1])[0]
#
#     prompt = f'For a 5 years old kid, explain what {subtopic} in {topic} is with an metaphor.'
#     completion = openai.Completion.create(engine="text-davinci-002", max_tokens=256, temperature=0.7, prompt=prompt)
#
#     subtopicExplanation = completion.choices[0].text.strip()
#
#     to_return = {'prompt': prompt, 'subtopicExplanation': subtopicExplanation}
#     return jsonify(to_return)
#
#
# # list subsubtopics
# @bp.route('/list_subsubtopics', methods=['GET', 'POST'])
# def list_sub_sub_topics():
#     openai.api_key = current_app.config["API_KEY"]
#
#     global subtopic
#
#
#     prompt = f'''List 8 subtopics related to {subtopic} in {topic} with brief explanation. Each in the form:
#     subsubtopic: explanation
#     1.
#     '''
#     completion = openai.Completion.create(engine="text-davinci-002", max_tokens=256, temperature=0.7, prompt=prompt)
#     result = completion.choices[0].text.strip()
#     global subsubtopicList
#     subsubtopicList = re.split(r'\d\. ', result)
#     while len(subsubtopicList) < 8 or ("?" in subsubtopicList[1]):
#         completion = openai.Completion.create(engine="text-davinci-002", max_tokens=256, temperature=0.7, prompt=prompt)
#         result = completion.choices[0].text.strip()
#         subsubtopicList = re.split(r'\d\. ', result)
#     if (subsubtopicList[0] == ''):
#         subsubtopicList = subsubtopicList[1:]
#
#     # extract pure subsubtopics
#     pureSubsubtopics = []
#     for index, subsubtopic in enumerate(subsubtopicList):
#         pureSubsubtopic = re.findall(r'(.*):', subsubtopic)[0]
#         pureSubsubtopics.append(pureSubsubtopic)
#
#     to_return = {'prompt': prompt, 'subsubtopicList': subsubtopicList, 'pureSubsubtopics': pureSubsubtopics}
#     return jsonify(to_return)
#
#
# # explain subsubtopic route
# @bp.route('/explain_subsubtopic', methods=['GET', 'POST'])
# def explainSubsubtopic():
#     openai.api_key = current_app.config["API_KEY"]
#
#     # json_data = request.get_json()
#     global subsubtopic
#     subsubtopicIndex = int(request.get_json()['subsubtopic'])
#     subsubtopic = re.split(r':', subsubtopicList[subsubtopicIndex - 1])[0]
#
#     prompt = f"For {subtopic} in {topic}, explain what {subsubtopic} is with an metaphor for a five years old kid."
#     completion = openai.Completion.create(engine="text-davinci-002", max_tokens=256, temperature=0.7, prompt=prompt)
#
#     subsubtopicExplanation = completion.choices[0].text.strip()
#
#     to_return = {'prompt': prompt, 'subsubtopicExplanation': subsubtopicExplanation}
#     return jsonify(to_return)
#
#
# # followup question
# @bp.route('/followup', methods=['GET', 'POST'])
# def followupQuestion():
#     openai.api_key = current_app.config["API_KEY"]
#
#     question = request.get_json()['question']
#     prompt = f"For {subsubtopic} in {subtopic} in {topic}, explain with metaphor for a five years old kid:  \"{question}\""
#     completion = openai.Completion.create(engine="text-davinci-002", max_tokens=256, temperature=0.7, prompt=prompt)
#
#     result = completion.choices[0].text.strip()
#     to_return = {'question': prompt, 'answer': result}
#     return jsonify(to_return)
