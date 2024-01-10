import json
import os
import streamlit as st
import re
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

from rag_query import (to_rag_document_list, query_by_skills, retrieve_unique_courses)
from rag_document import RagDocument
from utils import *
from prompts import Prompts

load_dotenv()

source_filepath = 'datasets/tesda_regulations_json_section1_summarized/'
tesda_regulation_pdf_list = []
for filename in os.listdir(source_filepath):
    tesda_regulation_pdf_list.append(load_tesda_regulation_pdf_from_json(os.path.join(source_filepath, filename)))

tesda_course_list = [tesda_regulation_pdf.name for tesda_regulation_pdf in tesda_regulation_pdf_list]


def agent(input_dict: dict, template: str, llm=None):
    if not llm:
        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            model=st.secrets['CHAT_MODEL'],
            streaming=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )

    prompt = PromptTemplate(
        input_variables=list(input_dict.keys()), template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    return chain.run(input_dict)


def get_tesda_course(course: str) -> TesdaRegulationPDF:
    return get_tesda_regulation_pdf(tesda_regulation_pdf_list, course)


def find_best_course(recommended_courses: List[str], disability: str, interests: str, llm=None):
    section1_pages = [get_tesda_regulation_pdf(tesda_regulation_pdf_list, course).summary['section1']
                      for course in recommended_courses]

    recommended_courses_str = ''
    count = 1
    for course, section1 in zip(recommended_courses, section1_pages):
        recommended_courses_str += f'{count}. COURSE: {course}\nDETAILS: {section1}\n\n'
        count += 1

    prompt = Prompts.FIND_BEST_COURSE_QUERY

    return agent({'disability': disability,
                  'interests': interests,
                  'courses_with_information': recommended_courses_str},
                 template=prompt, llm=llm).replace("\"", '')


def retrieve_top_unique_courses(documents: List[RagDocument], top_n: int = 4) -> List[str]:
    course_distances = {}
    for doc in documents:
        course_name = doc.metadata.get('name')
        if course_name not in course_distances or doc.distance > course_distances[course_name]:
            course_distances[course_name] = doc.distance

    print(course_distances)
    sorted_courses = sorted(course_distances, key=course_distances.get, reverse=True)
    return sorted_courses[:min(top_n, len(sorted_courses))]


def find_related_courses(course_name, course_list):
    match = re.match(r"(.+?)\s+(NC\s*\w+)", course_name)
    if not match:
        return []

    base_course, level = match.groups()
    related_courses = []
    for course in course_list:
        if base_course in course:
            related_courses.append(course)
    return related_courses


def retrieve_additional_courses(interests_query, avoidable_courses, n=10):
    rag_documents = to_rag_document_list(query_by_skills(interests_query, n=n, avoidable_courses=avoidable_courses))
    return sorted(rag_documents, key=lambda doc: doc.distance, reverse=True)


def find_similar_courses(current_courses, interests_query, avoidable_courses):
    if not current_courses or len(current_courses) < 3:
        additional_needed = 3 - len(current_courses) if current_courses else 3
        while len(current_courses) < 3:
            rag_documents = retrieve_additional_courses(interests_query, avoidable_courses)
            new_courses = retrieve_top_unique_courses(rag_documents)
            for course in new_courses:
                if course not in current_courses and course not in avoidable_courses:
                    current_courses.append(course)
                    if len(current_courses) == additional_needed:
                        break
    return current_courses

def recalibrate_top_courses(top_courses, avoidable_courses):
    related_courses_to_remove = []
    new_top_courses = []
    for course in avoidable_courses:
        related_courses_to_remove.extend(find_related_courses(course, tesda_course_list))

    related_courses_to_remove = list(set(related_courses_to_remove))
    print(f'Related courses to remove: {related_courses_to_remove}')
    for course in top_courses:
        if course not in related_courses_to_remove:
            new_top_courses.append(course)
    return new_top_courses


def find_similar_course(top_courses, avoidable_courses, disability, interests_query):
    # Next similar course 1
    if top_courses and len(top_courses) > 1:
        similar_course = find_best_course(top_courses, disability=disability, interests=interests_query)
    elif top_courses and len(top_courses) == 1:
        similar_course = top_courses[0]
    else:
        # empty
        rag_documents = retrieve_additional_courses(interests_query, avoidable_courses)
        top_courses = retrieve_top_unique_courses(rag_documents)
        similar_course = find_best_course(top_courses, disability=disability, interests=interests_query)
    top_courses, avoidable_courses = update_current_recommended_courses(similar_course, top_courses,
                                                                        avoidable_courses)

    return similar_course, top_courses, avoidable_courses


def update_current_recommended_courses(recommended_course, top_courses, avoidable_courses):
    related_best_courses = find_related_courses(recommended_course, tesda_course_list)
    avoidable_courses.extend(related_best_courses)
    for course in related_best_courses:
        if course in top_courses:
            top_courses.remove(course)
    avoidable_courses = list(set(avoidable_courses))
    top_courses = recalibrate_top_courses(top_courses, avoidable_courses)
    return top_courses, avoidable_courses


def get_similar_courses(top_courses, avoidable_courses, disability, interests_query, n=4):
    similar_courses = []
    for i in range(1, n):
        similar_course, top_courses, avoidable_courses = find_similar_course(top_courses, avoidable_courses,
                                                                             disability, interests_query)

        print(f'Similar Course {i}: {similar_course}')
        print(f'Top Courses {i}: {str(top_courses)}\n')
        print(f'Avoidable courses {i}: {str(avoidable_courses)}\n')
        similar_courses.append(similar_course)
    return similar_courses


def present_best_course(input_dict, llm):
    return agent(input_dict, template=Prompts.PRESENT_SUGGESTED_COURSE_QUERY,
                 llm=llm)


def present_similar_courses(similar_courses: List[str]):
    courses_info = [get_tesda_regulation_pdf(tesda_regulation_pdf_list, course)
                      for course in similar_courses]

    print('Course infos: ')
    print(courses_info)

    recommended_courses_str = ''
    count = 1
    for course in courses_info:
        summary = course.summary.get('section1')
        other_info = get_course_information_from_dataset(course.name)

        recommended_courses_str += (f'{count}. COURSE: {course.name}\nDETAILS: {summary}\n'
                                    f'OTHER INFO: {other_info}\n\n')
        count += 1

    return agent({'courses_information': recommended_courses_str},
                 template=Prompts.PRESENT_SIMILAR_COURSE_QUERY)


def generate_lesson_plan(course_name: str, disability: str, llm=None):
    directory = 'datasets/cc_short_csv/'
    course_cc_contents = f'Course: {course_name}\n'

    for filepath in os.listdir(directory):
        if filepath.startswith(course_name) and filepath.endswith('.csv'):
            full_path = os.path.join(directory, filepath)
            with open(full_path, 'r') as file:
                course_cc_contents += f'{file.read()}\n'

    return agent(input_dict={'course_information': course_cc_contents,
                             'disability': disability},
                 template=Prompts.PRESENT_LEARNING_MATERIAL, llm=llm)

def main():
    interests_query = ("I have a strong interest in assistive technology and want to learn how to create solutions to "
                       "help others with disabilities.")
    disability = 'Speech Impairment'
    avoidable_courses = []

    rag_documents = retrieve_additional_courses(interests_query, avoidable_courses)
    top_courses = retrieve_top_unique_courses(rag_documents)

    print('Top Courses')
    print(top_courses)

    best_course = find_best_course(top_courses, disability=disability, interests=interests_query)
    print(best_course)

    top_courses, avoidable_courses = update_current_recommended_courses(best_course, top_courses, avoidable_courses)

    print(f'Top Courses: {str(top_courses)}\n')
    print(f'Avoidable courses: {str(avoidable_courses)}\n')

    similar_courses = get_similar_courses(top_courses, avoidable_courses, disability, interests_query)
    print(similar_courses)

    response = present_similar_courses(similar_courses)

    print(response)

def lesson_plan_block():
    lesson_plan = generate_lesson_plan('Agroentrepreneurship NC IV',
                                       'Orthopedic')
    print(lesson_plan)


if __name__ == "__main__":
    lesson_plan_block()

