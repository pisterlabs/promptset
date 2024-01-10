from logic.ParserClass import SyllabusParser
from logic.CatClassifierModel import PhraseClassifier
import joblib
from flask import Flask, request, jsonify
import os
import requests
from openaicode import make_openai_request
import json

dict_of_topics = {"Class Name": ['Class Name', 'Instructor'],
                    "Course Number": ['Course Number'],
                    "Prerequisites": ['Prerequisites'],
                    "Course Schedule": ['Days', 'Time', 'Location'],
                    "Assignments": ['Assignment', 'Due Date'],
                    "Grading": ['Category', 'Percentage']}

dict_of_topics_extra = {"Prerequisites": ['Heavy Pre-reqs'],
                        "Course Schedule": ['Flexible'],
                        "Assignments": ['Midterms/Exams'],
                        "Grading": ['Curve', 'Extra Credit']}
def test():
    parser = SyllabusParser("samplepdfs/0._211_Syllabus_Fall23-1.pdf")
    bold, sections = parser.getImportantTopicSections()
    topics_observed = {}
    
    for i in range(len(bold)):
        if len(sections[i]) > 2000:
            sections[i] = sections[i][:2000]
        
        resp, ex_resp, topic = make_openai_request(bold[i], sections[i])
        data = resp.strip()
        
        if data is None or 'Nothing' in data:
            continue
        
        extra_data = ex_resp.strip() if ex_resp else None
        
        if extra_data is None or 'Nothing' in extra_data:
            topics_observed[topic] = [data]
        else:
            topics_observed[topic] = [data, extra_data]
    
    data = []
    
    for topic in topics_observed:
        data[topic] = [topics_observed[topic]]
        #if topic in dict_of_topics.keys():
        #    data[topic] = {}
        #    for i in range(len(dict_of_topics[topic])):
        #        if i < len(topics_observed[topic][0]):
        #            data[topic][dict_of_topics[topic][i]] = topics_observed[topic][0][i]
        #
        #if topic in dict_of_topics_extra.keys():
        #    data[topic] = {}
        #    if len(topics_observed[topic]) > 1:  # Check if there's extra data
        #        for i in range(len(dict_of_topics_extra[topic])):
        #            if i < len(topics_observed[topic][1]):
        #                data[topic][dict_of_topics_extra[topic][i]] = topics_observed[topic][1][i]
    print(data)  # Print the JSON data
    return data

test()