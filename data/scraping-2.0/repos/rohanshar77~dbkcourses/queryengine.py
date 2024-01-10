import asyncio
import re
import pinecone
import openai
import tiktoken
from dotenv import load_dotenv
import os
import json

# set up from environment
from pymongo import MongoClient

load_dotenv()
tokenizer = tiktoken.get_encoding("cl100k_base")

# open-ai auth
api_key = os.environ["OPENAI_KEY"]
openai.api_key = api_key

token_limit = 2000
score_lower_bound = 0.75


async def create_context_pinecone(question, undergrad_only):
    
    """
    Create a context for a question by finding the most similar context from the dataframe
    """
    # pinecone auth
    if undergrad_only:
        api_key = os.environ["PINE_CONE_API_KEY_UNDERGRAD"]
        pinecone.init(api_key=api_key, environment="gcp-starter")
        index = pinecone.Index("dbkcourses-undergrad")
    else:
        api_key = os.environ["PINE_CONE_API_KEY_ALL"]
        pinecone.init(api_key=api_key, environment="us-west4-gcp-free")
        index = pinecone.Index("umd-courses")

    # Get the embeddings for the question
    q_embedding = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Find courses given in the question directly
    course_pattern = r'(?i)([A-Z]{4})([0-9]{3})([A-Z]?)'
    courses_in_question = []

    for match in re.finditer(course_pattern, question):
        course_name = match.group(1)
        course_number = match.group(2)
        course_letter = match.group(3) if match.group(3) else ''

        course = course_name + course_number + course_letter
        courses_in_question.append(course.upper())

    # If courses were explicitly stated in question, find those courses in pinecone database
    fetch_response = {}
    if courses_in_question:
        fetch_response = index.fetch(ids=courses_in_question)

    # Find matches related to question embedding
    matches = index.query(
        vector=q_embedding,
        top_k=20,
        include_values=False,
        include_metadata=True
    )

    # Construct relevant_courses from fetch_response
    relevant_courses = {}

    # List to store related_matches
    related_matches_list = []

    for course_id, details in fetch_response.get('vectors', {}).items():
        course = {
            "id": details.get('id', ''),
            "score": '',  # Score not available in fetch_response
            "combined": details.get('metadata', {}).get('combined', ''),
            "description": details.get('metadata', {}).get('description', ''),
            "top_prof": details.get('metadata', {}).get('top_prof', ''),
            "top_rating": details.get('metadata', {}).get('top_rating', ''),
            "title": details.get('metadata', {}).get('title', ''),
            "tokens": details.get('metadata', {}).get('tokens', ''),
            "average_gpa": details.get('metadata', {}).get('average_gpa', ''),
            "credits": details.get('metadata', {}).get('credits', ''),
            "professors": details.get('metadata', {}).get('professors', '')
        }

        # For each explicitly stated course, find related ones
        related_matches = index.query(
            vector=details.get('values', {}),
            top_k=5,
            include_values=False,
            include_metadata=True
        )

        relevant_courses[course_id] = course
        related_matches_list.extend(related_matches.get('matches', []))

    # Prepend related_matches to matches
    matches['matches'] = related_matches_list + matches['matches']

    # Add courses from matches
    for match in matches.get('matches', []):
        if match.get('id', '') not in relevant_courses:
            course = {
                "id": match.get('id', ''),
                "score": match.get('score', ''),
                "combined": match.get('metadata', {}).get('combined', ''),
                "description": match.get('metadata', {}).get('description', ''),
                "title": match.get('metadata', {}).get('title', ''),
                "top_prof": match.get('metadata', {}).get('top_prof', ''),
                "top_rating": match.get('metadata', {}).get('top_rating', ''),
                "tokens": match.get('metadata', {}).get('tokens', ''),
                "average_gpa": match.get('metadata', {}).get('average_gpa', ''),
                "credits": match.get('metadata', {}).get('credits', ''),
                "professors": match.get('metadata', {}).get('professors', '')
            }

            relevant_courses[match.get('id', '')] = course

    context = ""
    token_count = 0

    # Construct context from matches
    for match in matches.get('matches', []):
        if match.get('score', 0) >= score_lower_bound and token_count <= token_limit:
            context += match.get('id', {}) + ": "
            context += match.get('metadata', {}).get('combined', '') + "\n\n"

            token_count += match.get('metadata', {}).get('tokens', 0)

            average_gpa = match.get('metadata', {}).get('average_gpa', '')
            if average_gpa:
                context += "Average GPA (out of 4.0):\n" + str(average_gpa) + "\n\n"

            top_prof = match.get('metadata', {}).get('top_prof', '')
            top_rating = match.get('metadata', {}).get('top_rating', '')
            if top_prof and top_rating:
                context += "Top professor and their rating (out of 5 stars):\n" + top_prof + ": " + str(top_rating) + " Stars\n\n"


    return {"context": context, "courses": list(relevant_courses.values())}


async def answer_chat(
    model="gpt-3.5-turbo",
    question="?",
    debug=False,
    context=""
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        p = f'You are a bot to help students find courses from the University of Maryland Course Catalog. Courses are organized such that the starting digit of the course specifies its "level". For example, a course beginning with 1 (like CMSC131) is an easier lower-level course, while a course beginning with 4 or higher (like CMSC417) is a more difficult upper-level course. When appropriate, prioritize recommending lower-level classes or provide options for both high and low-level classes. The difficulty of a class is also represented by average GPA and professor ratings. Try to prioritize classes with a higher average GPA and professor ratings. Always list the course code if you name a course (for example, include the CMSC131 course code if you recommend Object-Oriented Programming I) and round average GPA to 2 decimal places. I will include context for each query to help you. Use lists when it makes sense. \n\nContext: {context}\n\n---\n\nQuestion: {question} \nAnswer:'
        # Create a completions using the question and context
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": p}]
        )

        # print(response)
        tokens = response["usage"]["total_tokens"]
        print(f"Tokens Used: {tokens}, Cost: {tokens/1000 * 0.002}")
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(e)
        return ""


async def query_response(q, context=""):
    try:
        ans = await answer_chat(question=q, debug=False, context=context)
        # print(ans)
        return ans

    except Exception as e:
        print(e)
        return "An error occurred"


async def start():

    while True:
        user_input = input("Enter a message: ")

        # Check for exit command
        if user_input.lower() == "exit":
            break

        # context = await create_context_pinecone(user_input, False)
        context = await create_context_pinecone(user_input, True)
        resp = await answer_chat(question=user_input, debug=False, context=context["context"])
        print(resp)


if __name__ == '__main__':
    asyncio.run(start())

