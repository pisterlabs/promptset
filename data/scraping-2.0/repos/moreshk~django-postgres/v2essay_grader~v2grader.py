 
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
# from spellchecker import SpellChecker
import requests
import re
import os
from dotenv import load_dotenv

load_dotenv()

# 0. Check for relevance of the input essay to the topic
def check_relevance(user_response, title, description, essay_type, grade):
    print("I am in check relevance")
    print(essay_type, grade, title, description)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    relevance_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc", "essay_type"],
        template="""You are an essay grader for Naplan. Your inputs are

        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        Essay Type: {essay_type}

        Your job is to check relevance of the essay with respect to the task title and task description and essay type.
        If the essay is completely irrelevant then mention "Provided input is not relevant to the title and description and cannot be graded further."
        If it is relevant (or has some degree of relevance) then mention "Provided input is relevant to the title and description.".
        """,
    )


    chain = LLMChain(llm=llm, prompt=relevance_prompt)

    inputs = {
        "essay": user_response,
        "task_title": title,
        "task_desc": description,
        "essay_type": essay_type,
    }

    # print(essay_type, title, description)
    feedback_from_api = chain.run(inputs)
    return feedback_from_api

def hello_world():
    return "Hello, World!"

# 1. Check for Audience criteria

def check_audience_persuasive(user_response, title, description, essay_type, grade):
    print("I am in check audience")
    print(essay_type, grade, title, description)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    relevance_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc", "grade", "essay_type"],
        template="""You are an essay grader for Naplan. Your inputs are

        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        Students Grade: {grade}

        Essay Type: {essay_type}

        Your task is to grade the provided essay on the criteria of Audience (Scored out of 6)

        Grade 3 and Grade 5 criteria: 
        1-2 Points: The student shows an awareness of the reader but may not consistently engage or persuade throughout the piece.
        3-4 Points: The student engages the reader with a clear intent to persuade. The tone is mostly consistent, and the reader's interest is maintained.
        5-6 Points: The student effectively engages, orients, and persuades the reader throughout, demonstrating a strong connection with the audience.

        Grade 7 and Grade 9 criteria:
        1-2 Points: The student demonstrates an understanding of the reader but may occasionally lack depth in engagement or persuasion.
        3-4 Points: The student consistently engages the reader, demonstrating a mature intent to persuade with a nuanced and consistent tone.
        5-6 Points: The student masterfully engages, orients, and persuades the reader, showcasing a sophisticated and insightful connection with the audience.

        Keep in mind the students grade and the essay type. Grade 3 and 5 have the same criteria, Grade 7 and Grade 9 have the same criteria. 
        Be more lenient to the lower grades. So the same essay would score higher if written by a grade 3 vs for grade 5 even if the criteria was same. Same for grade 7 vs grade 9.
        Provide feedback on the input essay in terms of what if anything was done well and what can be improved. Try to include examples.
        Keep your response limited to less than 5 sentences and format your response as Feedback: (your feedback) Grade: (your grade)/(Scored out of).
        Remember that your grade cannot exceed 6.
 """,
    )


    chain = LLMChain(llm=llm, prompt=relevance_prompt)

    inputs = {
        "essay": user_response,
        "task_title": title,
        "task_desc": description,
        "grade": grade,
        "essay_type": essay_type,
    }

    # print(essay_type, title, description)
    feedback_from_api = chain.run(inputs)
    print(feedback_from_api)
    return feedback_from_api


# 2. Check Text structure

def check_text_structure_persuasive(user_response, title, description, essay_type, grade):
    print("I am in check text structure")
    print(essay_type, grade, title, description)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    relevance_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc", "grade", "essay_type"],
        template="""You are an essay grader for Naplan. Your inputs are

        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        Students Grade: {grade}

        Essay Type: {essay_type}

        Your task is to grade the provided essay on the criteria of text structure (Scored out of 4)

        Grade 3 and Grade 5 criteria: 
        1 Point: The student provides a structure with recognizable components, though transitions might be inconsistent.
        2-3 Points: The student's writing has a clear introduction, body, and conclusion. Transitions between ideas are mostly smooth.
        4 Points: The writing is well-organized with effective transitions, guiding the reader seamlessly through a coherent argument.

        Grade 7 and Grade 9 criteria:
        1 Point: The student's writing has a structure, but it may occasionally lack depth or sophistication in transitions and organization.
        2-3 Points: The student's writing has a clear introduction, body, and conclusion. Transitions between ideas are smooth and enhance the flow, reflecting a deeper understanding of the topic.
        4 Points: The writing is expertly organized with seamless transitions, guiding the reader effortlessly through a well-structured, sophisticated, and nuanced argument.

        Keep in mind the students grade and the essay type. Grade 3 and 5 have the same criteria, Grade 7 and Grade 9 have the same criteria. 
        Be more lenient to the lower grades. So the same essay would score higher if written by a grade 3 vs for grade 5 even if the criteria was same. Same for grade 7 vs grade 9.
        Provide feedback on the input essay in terms of what if anything was done well and what can be improved. Try to include examples.
        Keep your response limited to less than 5 sentences and format your response as Feedback: (your feedback) Grade: (your grade)/(Scored out of).
        Remember that your grade cannot exceed 4.
 """,
    )


    chain = LLMChain(llm=llm, prompt=relevance_prompt)

    inputs = {
        "essay": user_response,
        "task_title": title,
        "task_desc": description,
        "grade": grade,
        "essay_type": essay_type,
    }

    # print(essay_type, title, description)
    feedback_from_api = chain.run(inputs)
    print(feedback_from_api)
    return feedback_from_api

# 3. Check Ideas

def check_ideas_persuasive(user_response, title, description, essay_type, grade):
    print("I am in check ideas")
    print(essay_type, grade, title, description)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    relevance_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc", "grade", "essay_type"],
        template="""You are an essay grader for Naplan. Your inputs are

        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        Students Grade: {grade}

        Essay Type: {essay_type}

        Your task is to grade the provided essay on the criteria of Ideas (Scored out of 5)

        Grade 3 and Grade 5 criteria: 
        1-2 Points: The student presents a simple argument or point of view with minimal supporting details.
        3-4 Points: The student's argument is clearer, with some relevant supporting details. The writing may occasionally lack depth or elaboration.
        5 Points: The student presents a well-thought-out argument, supported by relevant and detailed examples or reasons.

        Grade 7 and Grade 9 criteria:
        1-2 Points: The student presents a clear argument with supporting details, but these might occasionally lack originality or depth.
        3-4 Points: The student's argument is robust and demonstrates critical thinking. The writing showcases depth, relevance, and originality in its supporting evidence.
        5 Points: The student presents a comprehensive, insightful, and original argument, bolstered by highly relevant, detailed, and unique examples or reasons.

        Keep in mind the students grade and the essay type. Grade 3 and 5 have the same criteria, Grade 7 and Grade 9 have the same criteria. 
        Be more lenient to the lower grades. So the same essay would score higher if written by a grade 3 vs for grade 5 even if the criteria was same. Same for grade 7 vs grade 9.
        Provide feedback on the input essay in terms of what if anything was done well and what can be improved. Try to include examples.
        Keep your response limited to less than 5 sentences and format your response as Feedback: (your feedback) Grade: (your grade)/(Scored out of).
        Remember that your grade cannot exceed 5.
 """,
    )


    chain = LLMChain(llm=llm, prompt=relevance_prompt)

    inputs = {
        "essay": user_response,
        "task_title": title,
        "task_desc": description,
        "grade": grade,
        "essay_type": essay_type,
    }

    # print(essay_type, title, description)
    feedback_from_api = chain.run(inputs)
    print(feedback_from_api)
    return feedback_from_api

# 4. Persuasive Devices (Scored out of 4)

def check_persuasive_devices_persuasive(user_response, title, description, essay_type, grade):
    print("I am in check ideas")
    print(essay_type, grade, title, description)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    relevance_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc", "grade", "essay_type"],
        template="""You are an essay grader for Naplan. Your inputs are

        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        Students Grade: {grade}

        Essay Type: {essay_type}

        Your task is to grade the provided essay on the criteria of Persuasive Devices (Scored out of 4).

        Grade 3 and Grade 5 criteria: 
        1 Point: Some use of persuasive devices, though they may be basic or not always effective.
        2-3 Points: The student uses persuasive devices, such as rhetorical questions, emotive language, or anecdotes, with varying effectiveness.
        4 Points: The student skillfully employs a range of persuasive devices to enhance and strengthen their argument.

        Grade 7 and Grade 9 criteria:
        1 Point: The student employs persuasive devices, but they may lack variety or sophistication.
        2-3 Points: The student uses a diverse range of persuasive devices with consistent effectiveness, demonstrating a deeper understanding of rhetorical techniques.
        4 Points: The student adeptly and creatively uses a diverse range of persuasive devices, masterfully enhancing their argument with sophistication.

        Keep in mind the students grade and the essay type. Grade 3 and 5 have the same criteria, Grade 7 and Grade 9 have the same criteria. 
        Be more lenient to the lower grades. So the same essay would score higher if written by a grade 3 vs for grade 5 even if the criteria was same. Same for grade 7 vs grade 9.
        Provide feedback on the input essay in terms of what if anything was done well and what can be improved. Try to include examples.
        Keep your response limited to less than 5 sentences and format your response as Feedback: (your feedback) Grade: (your grade)/(Scored out of).
        Remember that your grade cannot exceed 4.
 """,
    )


    chain = LLMChain(llm=llm, prompt=relevance_prompt)

    inputs = {
        "essay": user_response,
        "task_title": title,
        "task_desc": description,
        "grade": grade,
        "essay_type": essay_type,
    }

    # print(essay_type, title, description)
    feedback_from_api = chain.run(inputs)
    print(feedback_from_api)
    return feedback_from_api

# 5. Vocabulary (Scored out of 5)

def check_vocabulary_persuasive(user_response, title, description, essay_type, grade):
    print("I am in check vocabulary")
    print(essay_type, grade, title, description)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    relevance_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc", "grade", "essay_type"],
        template="""You are an essay grader for Naplan. Your inputs are

        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        Students Grade: {grade}

        Essay Type: {essay_type}

        Your task is to grade the provided essay on the criteria of Vocabulary (Scored out of 5).

        Grade 3 and Grade 5 criteria: 
        1-2 Points: The student uses appropriate vocabulary for their age, though word choice might occasionally be repetitive or imprecise.
        3-4 Points: The student's vocabulary is varied, with words often chosen for effect and clarity.
        5 Points: The student's vocabulary is rich and purposeful, significantly enhancing the persuasive quality of the writing.

        Grade 7 and Grade 9 criteria:
        1-2 Points: The student's vocabulary is appropriate but might occasionally lack precision or sophistication.
        3-4 Points: The student's vocabulary is varied, sophisticated, and often chosen for its effect, enhancing clarity and persuasion.
        5 Points: The student's vocabulary is rich, sophisticated, and purposefully chosen, significantly elevating the persuasive quality of the writing with nuance.

        Keep in mind the students grade and the essay type. Grade 3 and 5 have the same criteria, Grade 7 and Grade 9 have the same criteria. 
        Be more lenient to the lower grades. So the same essay would score higher if written by a grade 3 vs for grade 5 even if the criteria was same. Same for grade 7 vs grade 9.
        Provide feedback on the input essay in terms of what if anything was done well and what can be improved. Try to include examples.
        Keep your response limited to less than 5 sentences and format your response as Feedback: (your feedback) Grade: (your grade)/(Scored out of).
        Remember that your grade cannot exceed 5.
 """,
    )


    chain = LLMChain(llm=llm, prompt=relevance_prompt)

    inputs = {
        "essay": user_response,
        "task_title": title,
        "task_desc": description,
        "grade": grade,
        "essay_type": essay_type,
    }

    # print(essay_type, title, description)
    feedback_from_api = chain.run(inputs)
    print(feedback_from_api)
    return feedback_from_api

# 6. Cohesion (Scored out of 4)

def check_cohesion_persuasive(user_response, title, description, essay_type, grade):
    print("I am in check cohesion")
    print(essay_type, grade, title, description)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    relevance_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc", "grade", "essay_type"],
        template="""You are an essay grader for Naplan. Your inputs are

        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        Students Grade: {grade}

        Essay Type: {essay_type}

        Your task is to grade the provided essay on the criteria of Cohesion (Scored out of 4).

        Grade 3 and Grade 5 criteria: 
        1 Point: The student's writing shows some connections between ideas, though these might be basic or unclear at times.
        2-3 Points: Use of referring words, text connectives, and other cohesive devices to link ideas, with occasional lapses.
        4 Points: The student masterfully controls multiple threads and relationships across the text, ensuring a cohesive and unified argument.

        Grade 7 and Grade 9 criteria:
        1 Point: The student's writing shows connections between ideas, but these might lack sophistication.
        2-3 Points: Effective use of advanced cohesive devices to link ideas, demonstrating a deeper understanding of textual flow.
        4 Points: The student expertly controls multiple threads and relationships across the text, ensuring a cohesive, unified, and flowing argument with advanced techniques.

        Keep in mind the students grade and the essay type. Grade 3 and 5 have the same criteria, Grade 7 and Grade 9 have the same criteria. 
        Be more lenient to the lower grades. So the same essay would score higher if written by a grade 3 vs for grade 5 even if the criteria was same. Same for grade 7 vs grade 9.
        Provide feedback on the input essay in terms of what if anything was done well and what can be improved. Try to include examples.
        Keep your response limited to less than 5 sentences and format your response as Feedback: (your feedback) Grade: (your grade)/(Scored out of).
        Remember that your grade cannot exceed 4.
 """,
    )


    chain = LLMChain(llm=llm, prompt=relevance_prompt)

    inputs = {
        "essay": user_response,
        "task_title": title,
        "task_desc": description,
        "grade": grade,
        "essay_type": essay_type,
    }

    # print(essay_type, title, description)
    feedback_from_api = chain.run(inputs)
    print(feedback_from_api)
    return feedback_from_api

# 7. Paragraphing (Scored out of 2)

def check_paragraphing_persuasive(user_response, title, description, essay_type, grade):
    print("I am in check paragraphing")
    print(essay_type, grade, title, description)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    relevance_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc", "grade", "essay_type"],
        template="""You are an essay grader for Naplan. Your inputs are

        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        Students Grade: {grade}

        Essay Type: {essay_type}

        Your task is to grade the provided essay on the criteria of Paragraphing (Scored out of 2).

        Grade 3 and Grade 5 criteria: 
        0 - no use of paragraphing
        1 Point: The student groups related ideas into paragraphs, though there might be occasional lapses in coherence.
        2 Points: Ideas are effectively and logically grouped into clear paragraphs, enhancing the structure and flow of the argument.

        Grade 7 and Grade 9 criteria:
        0 - no use of paragraphing
        1 Point: The student logically groups related ideas into paragraphs, but transitions might occasionally lack depth.
        2 Points: Ideas are effectively and logically grouped into clear paragraphs, enhancing the structure and flow of the argument with sophistication.

        Keep in mind the students grade and the essay type. Grade 3 and 5 have the same criteria, Grade 7 and Grade 9 have the same criteria. 
        Be more lenient to the lower grades. So the same essay would score higher if written by a grade 3 vs for grade 5 even if the criteria was same. Same for grade 7 vs grade 9.
        Provide feedback on the input essay in terms of what if anything was done well and what can be improved. Try to include examples.
        Keep your response limited to less than 5 sentences and format your response as Feedback: (your feedback) Grade: (your grade)/(Scored out of).
        Remember that your grade cannot exceed 2.
 """,
    )


    chain = LLMChain(llm=llm, prompt=relevance_prompt)

    inputs = {
        "essay": user_response,
        "task_title": title,
        "task_desc": description,
        "grade": grade,
        "essay_type": essay_type,
    }

    # print(essay_type, title, description)
    feedback_from_api = chain.run(inputs)
    print(feedback_from_api)
    return feedback_from_api

# 8. Sentence Structure (Scored out of 6)

def check_sentence_structure_persuasive(user_response, title, description, essay_type, grade):
    print("I am in check Sentence Structure")
    print(essay_type, grade, title, description)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    relevance_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc", "grade", "essay_type"],
        template="""You are an essay grader for Naplan. Your inputs are

        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        Students Grade: {grade}

        Essay Type: {essay_type}

        Your task is to grade the provided essay on the criteria of Sentence Structure (Scored out of 6).

        Scoring Guide:

        0 -	no evidence of sentences
        1 - some correct formation of sentences, some meaning can be construed
        2 - use of some complex sentences, correct sentences are mainly simple and/or compound sentences, meaning is predominantly clear
        3 - most (approx. 80%) simple and compound sentences correct
        AND some complex sentences are correct meaning is predominantly clear
        4 - most (approx. 80%) simple, compound and complex sentences are correct
        OR all simple, compound and complex sentences are correct but do not demonstrate variety, meaning is clear
        5 - sentences are correct (allow for occasional error in more sophisticated structures). The student effectively employs a variety of sentence structures, enhancing the clarity, rhythm, and sophistication of the writing.
        6 - all sentences are correct (allow for occasional slip, e.g. a missing word) writing contains controlled and well developed sentences that express precise meaning and are
        consistently effective. The student masterfully employs a diverse range of sentence structures, adding depth, clarity, and sophistication to the writing with nuance.

        Keep in mind the students grade and the essay type. Be more lenient to the lower grades and stricter with higher grades in your scoring. 
        Even though all grades have the same criteria, the same essay would score higher if written by a grade 3 vs a grade 5. Same for grade 7 vs grade 9.
        Provide feedback on the input essay in terms of what if anything was done well and what can be improved. Try to include examples.
        Keep your response limited to less than 5 sentences and format your response as Feedback: (your feedback) Grade: (your grade)/(Scored out of).
        Remember that your grade cannot exceed 6.
 """,
    )


    chain = LLMChain(llm=llm, prompt=relevance_prompt)

    inputs = {
        "essay": user_response,
        "task_title": title,
        "task_desc": description,
        "grade": grade,
        "essay_type": essay_type,
    }

    # print(essay_type, title, description)
    feedback_from_api = chain.run(inputs)
    print(feedback_from_api)
    return feedback_from_api


# 9. Punctuation (Scored out of 6)
def check_punctuation_persuasive(user_response, title, description, essay_type, grade):
    print("I am in check punctuation")
    print(essay_type, grade, title, description)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    relevance_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc", "grade", "essay_type"],
        template="""You are an essay grader for Naplan. Your inputs are

        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        Students Grade: {grade}

        Essay Type: {essay_type}

        Your task is to grade the provided essay on the criteria of Punctuation (Scored out of 6 for persuasive essay type and scored out of 5 for narrative essay type).

        Grade 3 and Grade 5 criteria: 
        1-2 Points: The student uses basic and some advanced punctuation with occasional errors.
        3-4 Points: The student correctly uses a range of punctuation, including quotation marks and apostrophes, with few mistakes.
        5-6 Points: Punctuation is used skillfully and accurately throughout the writing, significantly aiding the reader's understanding.

        Grade 7 and Grade 9 criteria:
        1-2 Points: The student uses a mix of basic and advanced punctuation with some errors.
        3-4 Points: The student accurately uses a wide range of punctuation, including more advanced forms, with few mistakes and for stylistic effect.
        5-6 Points: Punctuation is used expertly and accurately throughout the writing, not just for clarity but also for stylistic and rhetorical effect.

        Keep in mind the students grade and the essay type. Grade 3 and 5 have the same criteria, Grade 7 and Grade 9 have the same criteria. 
        Be more lenient to the lower grades. So the same essay would score higher if written by a grade 3 vs for grade 5 even if the criteria was same. Same for grade 7 vs grade 9.
        Provide feedback on the input essay in terms of what if anything was done well and what can be improved. Try to include examples. 
        
        Keep your response limited to less than 5 sentences and format your response as Feedback: (your feedback) Grade: (your grade)/(Scored out of).

        Remember the max score for persuasive essay type is 6 and narrative essay type is 5. Hence scored out of for persuasive will be 5 and for narrative will be 5.
 """,
    )


    chain = LLMChain(llm=llm, prompt=relevance_prompt)

    inputs = {
        "essay": user_response,
        "task_title": title,
        "task_desc": description,
        "grade": grade,
        "essay_type": essay_type,
    }

    # print(essay_type, title, description)
    feedback_from_api = chain.run(inputs)
    print(feedback_from_api)
    return feedback_from_api

# Spell check using BING API

def spell_check(text):
    print("I am in Bing Spell check")
    subscription_key = os.environ.get('BING_SPELLCHECK_KEY')

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Ocp-Apim-Subscription-Key": subscription_key,
    }
    
    endpoint_url = "https://api.bing.microsoft.com/v7.0/spellcheck"
    text_to_check = text.replace('\n', ' ').replace('\r', ' ')

    data = {
        "text": text_to_check,
        "mode": "proof",  # Use 'proof' mode for comprehensive checks
    }

    response = requests.post(endpoint_url, headers=headers, data=data)
    
    output = ""  # Initialize the output string

    if response.status_code == 200:
        result = response.json()
        for flagged_token in result.get('flaggedTokens', []):
            token = flagged_token['token']
            for suggestion in flagged_token.get('suggestions', []):
                suggested_token = suggestion['suggestion']
                if suggested_token.replace(token, '').strip() in ["", ":", ";", ",", ".", "?", "!"]:
                    continue
                if " " not in suggested_token:
                    output += f"Misspelled word: {token}\n"
                    output += f"Suggestion: {suggested_token}\n"
    else:
        output += f"Error: {response.status_code}\n"
        output += response.text + "\n"

    # If no mistakes were found, update the output to indicate this.
    if not output:
        output = "No spelling mistakes found"

    print("Response from Bing Spell check:", output)
    return output

# 10. Spelling (Scored out of 6)
def check_spelling_persuasive(user_response, title, description, essay_type, grade):
    print("I am in check spelling")
    print(essay_type, grade, title, description)

    spell_check_response = spell_check(user_response);

    # Making a second run to generate the grading

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    verification_prompt = PromptTemplate(
        input_variables=["essay", "mistakes", "grade"],
        template="""You are an spelling grader verifier for Naplan. Your inputs are

        Essay: {essay}

        Spelling mistakes: {mistakes}

        Students Grade: {grade}

        Another grader has already done the work of finding the spelling mistakes in the essay.

        You will then grade the essay on spellings using the below criteria.

        Grade 3 and Grade 5 criteria: 
        1-2 Points: The student spells most common words correctly, with errors in more challenging or less common words.
        3-4 Points: A majority of words, including challenging ones, are spelled correctly.
        5-6 Points: The student demonstrates an excellent grasp of spelling across a range of word types, with errors being very rare.

        Grade 7 and Grade 9 criteria:
        1-2 Points: The student spells most words correctly but may have errors with complex or specialized words.
        3-4 Points: A vast majority of words, including complex and specialized ones, are spelled correctly.
        5-6 Points: The student demonstrates an impeccable grasp of spelling across a diverse range of word types, including advanced and specialized vocabulary.

        In feedback also mention your reasoning behind the grade you assign and be generous in your grading if no spelling mistakes were received as input.
        Format your response as Feedback: (your feedback) Grade: (your grade)/(Scored out of).
        """,
    )

    chain = LLMChain(llm=llm, prompt=verification_prompt)

    inputs = {
        "essay": user_response,
        "mistakes": spell_check_response,
        "grade": grade,
    }

    # print(essay_type, title, description)
    second_feedback = chain.run(inputs)
    print("second run", second_feedback)

    return second_feedback

# 11. Narrative Audience (Scored out of 6)

def check_audience_narrative(user_response, title, description, essay_type, grade):
    print("I am in audience narrative")
    print(essay_type, grade, title, description)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    relevance_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc", "grade", "essay_type"],
        template="""You are an narrative writing (story) grader for Naplan. Your inputs are

        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        Students Grade: {grade}

        Essay Type: {essay_type}

        Your task is to grade the provided story on the criteria of Audience (Scored out of 6).
        Scoring guide: 
        0 - symbols or drawings which have the intention of conveying meaning.
        1 - response to audience needs is limited • contains simple written content. may be a title only OR • meaning is difficult to access OR • copied stimulus material, including prompt topic
        2 - shows basic awareness of audience expectations through attempting to orient the reader • provides some information to support reader understanding. may include simple narrative markers, e.g. – simple title – formulaic story opening: Long, long ago …; Once a boy was walking when … • description of people or places • reader may need to fill gaps in information • text may be short but is easily read.
        3 - orients the reader with an internally consistent story that attempts to support the reader by developing a shared understanding of context • contains sufficient information for the reader to follow the story fairly easily
        4 - supports reader understanding AND • begins to engage the reader
        5 - supports and engages the reader through deliberate choice of language and use of narrative devices.
        6 - caters to the anticipated values and expectations of the reader • influences or affects the reader through precise and sustained choice of language and use of narrative devices

        Keep in mind the students grade and the essay type. Be more lenient to the lower grades and stricter with higher grades in your scoring. 
        Even though all grades have the same criteria, the same story would score higher if written by a grade 3 vs a grade 5. Same for grade 7 vs grade 9.
        Provide feedback on the input story in terms of what if anything was done well and what can be improved. Try to include examples.
        Keep your response limited to less than 5 sentences and format your response as Feedback: (your feedback) Grade: (your grade)/(Scored out of).
        Remember that your grade cannot exceed 6.
 """,
    )


    chain = LLMChain(llm=llm, prompt=relevance_prompt)

    inputs = {
        "essay": user_response,
        "task_title": title,
        "task_desc": description,
        "grade": grade,
        "essay_type": essay_type,
    }

    # print(essay_type, title, description)
    feedback_from_api = chain.run(inputs)
    print(feedback_from_api)
    return feedback_from_api


# 12. Narrative Text Structure (0 - 4)

def check_text_structure_narrative(user_response, title, description, essay_type, grade):
    print("I am in text structure narrative")
    print(essay_type, grade, title, description)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    relevance_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc", "grade", "essay_type"],
        template="""You are an narrative writing (story) grader for Naplan. Your inputs are

        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        Students Grade: {grade}

        Essay Type: {essay_type}

        Your task is to grade the provided story on the criteria of Text Structure (Scored out of 4).
        Scoring guide: 
        0 - no evidence of any structural components of a times equenced text • symbols or drawings • inappropriate genre, e.g. a recipe, argument • title only
        1 - minimal evidence of narrative structure, e.g. a story beginning only or a `middle` with no orientation • a recount of events with no complication • note that not all recounts are factual • may be description
        2 - contains a beginning and a complication • where a resolution is present it is weak, contrived or `tacked on` (e.g. I woke up, I died, They lived happily ever after) • a complication presents a problem to be solved, introduces tension, and requires a response. It drives the story forward and leads to a series of events or responses • complications should always be read in context • may also be a complete story where all parts of the story are weak or minimal (the story has a problem to be solved but it does not add to the tension or excitement).
        3 - contains orientation, complication and resolution • detailed longer text may resolve one complication and lead into a new complication or layer a new complication onto an existing one rather than conclude
        4 - coherent, controlled and complete narrative, employing effective plot devices in an appropriate structure, and including an effective ending. sophisticated structures or plot devices include: - foreshadowing/flashback - red herring/cliffhanger - coda/twist - evaluation/reflection - circular/parallel plots

        Keep in mind the students grade and the essay type. Be more lenient to the lower grades and stricter with higher grades in your scoring. 
        Even though all grades have the same criteria, the same essay would score higher if written by a grade 3 vs a grade 5. Same for grade 7 vs grade 9.
        Provide feedback on the input story in terms of what if anything was done well and what can be improved. Try to include examples.
        Keep your response limited to less than 5 sentences and format your response as Feedback: (your feedback) Grade: (your grade)/(Scored out of).
        Remember that your grade cannot exceed 4.
 """,
    )


    chain = LLMChain(llm=llm, prompt=relevance_prompt)

    inputs = {
        "essay": user_response,
        "task_title": title,
        "task_desc": description,
        "grade": grade,
        "essay_type": essay_type,
    }

    # print(essay_type, title, description)
    feedback_from_api = chain.run(inputs)
    print(feedback_from_api)
    return feedback_from_api

# 13. Narrative Ideas (0 - 5)

def check_ideas_narrative(user_response, title, description, essay_type, grade):
    print("I am in ideas narrative")
    print(essay_type, grade, title, description)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    relevance_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc", "grade", "essay_type"],
        template="""You are an narrative writing (story) grader for Naplan. Your inputs are

        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        Students Grade: {grade}

        Essay Type: {essay_type}

        Your task is to grade the provided story on the criteria of Ideas: The creation, selection and crafting of ideas for a narrative. (Scored out of 5).
        Scoring guide: 
        0 - no evidence or insufficient evidence • symbols or drawings • title only
        1 - one idea OR • ideas are very few and very simple OR • ideas appear unrelated to each other OR • ideas appear unrelated to prompt
        2 - one idea with simple elaboration OR • ideas are few and related but not elaborated OR • many simple ideas are related but not elaborated
        3 - ideas show some development or elaboration • all ideas relate coherently
        4 - ideas are substantial and elaborated AND contribute effectively to a central storyline • the story contains a suggestion of an underlying theme
        5 - ideas are generated, selected and crafted to explore a recognisable theme • ideas are skilfully used in the service of the storyline • ideas may include: - psychological subjects - unexpected topics - mature viewpoints - elements of popular culture - satirical perspectives - extended metaphor - traditional sub-genre subjects: heroic quest / whodunnit / good vs evil / overcoming the odds

        Keep in mind the students grade and the essay type. Be more lenient to the lower grades and stricter with higher grades in your scoring. 
        Even though all grades have the same criteria, the same essay would score higher if written by a grade 3 vs a grade 5. Same for grade 7 vs grade 9.
        Provide feedback on the input story in terms of what if anything was done well and what can be improved. Try to include examples.
        Keep your response limited to less than 5 sentences and format your response as Feedback: (your feedback) Grade: (your grade)/(Scored out of).
        Remember that your grade cannot exceed 5.
 """,
    )


    chain = LLMChain(llm=llm, prompt=relevance_prompt)

    inputs = {
        "essay": user_response,
        "task_title": title,
        "task_desc": description,
        "grade": grade,
        "essay_type": essay_type,
    }

    # print(essay_type, title, description)
    feedback_from_api = chain.run(inputs)
    print(feedback_from_api)
    return feedback_from_api


# 14. Narrative Character and Setting (0 - 4)

def check_setting_narrative(user_response, title, description, essay_type, grade):
    print("I am in setting narrative")
    print(essay_type, grade, title, description)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    relevance_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc", "grade", "essay_type"],
        template="""You are an narrative writing (story) grader for Naplan. Your inputs are

        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        Students Grade: {grade}

        Essay Type: {essay_type}

        Your task is to grade the provided story on the criteria of Character and Setting.
        Character: the portrayal and development of character. 
        Setting: the development of a sense of place, time and atmosphere. (Scored out of 4).
        Scoring guide: 
        0 - no evidence or insufficient evidence, symbols or drawings, writes in wrong genre, title only

        1 - only names characters or gives their roles (e.g. father, the teacher, my friend, dinosaur, we, Jim) AND/OR only names the setting (e.g.school, the place we were at) setting is vague or confused	
        2 - suggestion of characterisation through brief descriptions or speech or feelings, but lacks substance or continuity 
        AND/OR
        suggestion of setting through very brief and superficial descriptions of place and/or time	
        basic dialogue or a few adjectives to describe a character or a place

        3 - characterisation emerges through descriptions, actions, speech or the attribution of thoughts and feelings to a character
        AND/OR
        setting emerges through description of place, time and atmosphere	

        4 - effective characterisation: details are selected to create distinct characters
        AND/OR
        Maintains a sense of setting throughout. Details are selected to create a sense of place and atmosphere. convincing dialogue, introspection and reactions to other characters

        Keep in mind the students grade and the essay type. Be more lenient to the lower grades and stricter with higher grades in your scoring. 
        Even though all grades have the same criteria, the same story would score higher if written by a grade 3 vs a grade 5. Same for grade 7 vs grade 9.
        Provide feedback on the input story in terms of what if anything was done well and what can be improved. Try to include examples.
        Keep your response limited to less than 5 sentences and format your response as Feedback: (your feedback) Grade: (your grade)/(Scored out of).
        Remember that your grade cannot exceed 4.
 """,
    )


    chain = LLMChain(llm=llm, prompt=relevance_prompt)

    inputs = {
        "essay": user_response,
        "task_title": title,
        "task_desc": description,
        "grade": grade,
        "essay_type": essay_type,
    }

    # print(essay_type, title, description)
    feedback_from_api = chain.run(inputs)
    print(feedback_from_api)
    return feedback_from_api


# 15. Narrative Character and Setting (0 - 4)

def check_setting_narrative(user_response, title, description, essay_type, grade):
    print("I am in setting narrative")
    print(essay_type, grade, title, description)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    relevance_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc", "grade", "essay_type"],
        template="""You are an narrative writing (story) grader for Naplan. Your inputs are

        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        Students Grade: {grade}

        Essay Type: {essay_type}

        Your task is to grade the provided story on the criteria of Character and Setting.
        Character: the portrayal and development of character. 
        Setting: the development of a sense of place, time and atmosphere. (Scored out of 4).
        Scoring guide: 
        0 - no evidence or insufficient evidence, symbols or drawings, writes in wrong genre, title only

        1 - only names characters or gives their roles (e.g. father, the teacher, my friend, dinosaur, we, Jim) AND/OR only names the setting (e.g.school, the place we were at) setting is vague or confused	
        2 - suggestion of characterisation through brief descriptions or speech or feelings, but lacks substance or continuity 
        AND/OR
        suggestion of setting through very brief and superficial descriptions of place and/or time	
        basic dialogue or a few adjectives to describe a character or a place

        3 - characterisation emerges through descriptions, actions, speech or the attribution of thoughts and feelings to a character
        AND/OR
        setting emerges through description of place, time and atmosphere	

        4 - effective characterisation: details are selected to create distinct characters
        AND/OR
        Maintains a sense of setting throughout. Details are selected to create a sense of place and atmosphere. convincing dialogue, introspection and reactions to other characters

        Keep in mind the students grade and the essay type. Be more lenient to the lower grades and stricter with higher grades in your scoring. 
        Even though all grades have the same criteria, the same story would score higher if written by a grade 3 vs a grade 5. Same for grade 7 vs grade 9.
        Provide feedback on the input story in terms of what if anything was done well and what can be improved. Try to include examples.
        Keep your response limited to less than 5 sentences and format your response as Feedback: (your feedback) Grade: (your grade)/(Scored out of).
        Remember that your grade cannot exceed 4.
 """,
    )


    chain = LLMChain(llm=llm, prompt=relevance_prompt)

    inputs = {
        "essay": user_response,
        "task_title": title,
        "task_desc": description,
        "grade": grade,
        "essay_type": essay_type,
    }

    # print(essay_type, title, description)
    feedback_from_api = chain.run(inputs)
    print(feedback_from_api)
    return feedback_from_api


# 16. Cohesion (0 - 4)

def check_cohesion_narrative(user_response, title, description, essay_type, grade):
    print("I am in setting narrative")
    print(essay_type, grade, title, description)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    relevance_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc", "grade", "essay_type"],
        template="""You are an narrative writing (story) grader for Naplan. Your inputs are

        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        Students Grade: {grade}

        Essay Type: {essay_type}

        Your task is to grade the provided story on the criteria of 
        Cohesion: The control of multiple threads and relationships over the whole text, achieved through the use of referring words, substitutions, word associations and text connectives. 
        Score Range: 0-4 

        Scoring guide: 
        0 - symbols or drawings
        1 - links are missing or incorrect short script often confusing for the reader
        2 - some correct links between sentences (do not penalise for poor punctuation), 
        most referring words are accurate. reader may occasionally need to re-read and provide their own links to clarify meaning
        3 - cohesive devices are used correctly to support reader understanding, accurate use of referring words,
        meaning is clear and text flows well in a sustained piece of writing
        4 - a range of cohesive devices is used correctly and deliberately to enhance reading, an extended, highly cohesive piece of writing showing continuity of ideas and tightly linked sections of text

        Keep in mind the students grade and the essay type. Be more lenient to the lower grades and stricter with higher grades in your scoring. 
        Even though all grades have the same criteria, the same story would score higher if written by a grade 3 vs a grade 5. Same for grade 7 vs grade 9.
        Provide feedback on the input story in terms of what if anything was done well and what can be improved. Try to include examples.
        Keep your response limited to less than 5 sentences and format your response as Feedback: (your feedback) Grade: (your grade)/(Scored out of).
        Remember that your grade cannot exceed 4.
 """,
    )


    chain = LLMChain(llm=llm, prompt=relevance_prompt)

    inputs = {
        "essay": user_response,
        "task_title": title,
        "task_desc": description,
        "grade": grade,
        "essay_type": essay_type,
    }

    # print(essay_type, title, description)
    feedback_from_api = chain.run(inputs)
    print(feedback_from_api)
    return feedback_from_api