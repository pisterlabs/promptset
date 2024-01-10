from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
import requests
import openai
import pandas as pd
import os
from ast import literal_eval
from supabase.client import Client, create_client
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from decouple import config
import json

from translate import translate_lang

# Config
openai.api_base = config('OPENAI_API_BASE')
openai.api_type = "azure"

openai.api_version = config('OPENAI_API_VERSION')
openai.api_key = config('OPENAI_API_KEY')

supabase_url = config('SUPABASE_URL')
supabase_key = config('SUPABASE_SERVICE_KEY')
SUPABASE_SECRET_KEY = config('SUPABASE_SECRET_KEY')

supabase: Client = create_client(supabase_url, supabase_key)


llm = AzureOpenAI(temperature=0.0,
                  model_name="gpt-35-turbo-instruct",
                  openai_api_version=config('OPENAI_API_VERSION'),
                  openai_api_key = config('OPENAI_API_KEY'),
                deployment_name="lang-chain",)

embeddings = OpenAIEmbeddings(deployment="chaining",
                              openai_api_version=config('OPENAI_API_VERSION'),
                              openai_api_key = config('OPENAI_API_KEY'),
                            openai_api_base="https://lang-chain.openai.azure.com/",
                            openai_api_type="azure",)



# Supabase Request 
headers = {
    'apikey': f"{supabase_key}",
    'Authorization': f"Bearer {supabase_key}"
}

def courseoutline(qa):
    return qa.run('What are the main chapters or sections in this book? Give the response as a list ')

def subtopics(chapter_name, qa):
    return qa.run(f'What are the subtopics covered in {chapter_name}?  Give the response as a list ')

def indepth_subtopics(subtopicname, chapter_name, qa):
    return qa.run(f'Give an in-depth overview of the chapter {subtopicname} in the {chapter_name} chapter ')

def getcourseinfo(course_code, target_language='English'):
    result = {
    }

    url = 'https://hryzlnqorkzfkdrkpgbl.supabase.co/rest/v1/data?select=*'
    # Error handling for the request
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return {}
    df = pd.DataFrame(data)
    course_data = df[df['course_code'] == course_code]

    result['chapter'] = {}

    for _, row in course_data.iterrows():
        sub_topics = literal_eval(row['sub_topics'])[1:]
        chapter = row['chapters']

        if target_language != 'English':
            translated_subtopics = translate_lang(target_language, ' '.join(sub_topics))
            translated_chapter = translate_lang(target_language, chapter)
            
            result['chapter'][translated_chapter] = {'subtopics':translated_subtopics}
        else:
            result['chapter'][chapter] = {'subtopics':sub_topics}
        
    return result

def getcourse(course_code, page_number, target_language='English'):
    result = {}
    url = 'https://hryzlnqorkzfkdrkpgbl.supabase.co/rest/v1/data?select=*'
    # Error handling for the request
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return {}
    df = pd.DataFrame(data)
    df = df[df['course_code'] == course_code].reset_index(drop=True)
    
    sub_topics = df['sub_topics'].iloc[page_number]
    sub_topics = literal_eval(sub_topics)[1:]
    chapter = df['chapters'].iloc[page_number]
    print(len(sub_topics))
    
    vector_store = SupabaseVectorStore(client=supabase, embedding=embeddings, table_name=f"{course_code}documents".lower(), query_name=f"{course_code}match_documents".lower())
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())

    result['course-title'] = chapter
    result['subtopics'] = {}

    for sub_topic in sub_topics:
        indepth_response = indepth_subtopics(sub_topic, chapter, qa)
        if target_language != 'English':
            indepth_response = translate_lang(target_language, indepth_response)
        
        result['subtopics'][sub_topic] = {'indepth_response': indepth_response}

    return result

def get_course_quiz(course_code, target_language='English'):
    result = {'questions': {}}

    vector_store = SupabaseVectorStore(client=supabase, embedding=embeddings, table_name=f"quiz{course_code}documents".lower(), query_name=f"quiz{course_code}match_documents".lower(),)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())

    data = qa.run(f'List 5 questions and the options from this book')
    responses = data.split('\n')

    for i in range(1, len(responses), 6):
        question = responses[i]
        options = responses[i+1: i+5]
        result['questions'][question] = {'options':options}

        vector_store = SupabaseVectorStore(client=supabase, embedding=embeddings, table_name=f"{course_code}documents".lower(), query_name=f"{course_code}match_documents".lower(),)
        answer_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())

        # ans = qa.run(f'Which of a,b,c,or d is the correct answer {str(responses[i: i+5])}?')
        # result['questions'][question] = {'answer':ans}
        # Formulate the question for the answer retrieval
        option_str = ', '.join(options)
        ans_query = f'Which of {option_str} is the correct answer for the question: "{question}"?'
        ans = answer_qa.run(ans_query)
        
        # Update the question's data with the answer
        result['questions'][question]['answer'] = ans
    #result = json.dumps(result, indent = 4) 

    return result

def get_question_response(prompt, course_code, target_language='English'):
    vector_store = SupabaseVectorStore(client=supabase, embedding=embeddings, table_name=f"{course_code}documents".lower(), query_name=f"{course_code}match_documents".lower(),)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())

    data = qa.run(prompt)
    if target_language !='English':
            data = translate_lang(target_language, data)
        
    return data
