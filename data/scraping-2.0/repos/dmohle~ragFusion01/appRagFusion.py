import os
from dotenv import load_dotenv
import openai
import random
import streamlit as st


# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")  # Alternative: Use environment variable
if openai.api_key is None:
    raise Exception("No OpenAI API key found. Please set it as an environment variable or in main.py")


# Function to generate queries using OpenAI's ChatGPT
def generate_queries_chatgpt(original_query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system",
             "content": """You are a wonderful AI assistant who can easily generate multiple search queries based on 
                        a single input query. Your temperature is set to 0.3 and you always think 
                        step by step and rank the documents easily."""},
            {"role": "user", "content": f"Generate multiple search queries related to: {original_query}"},
            {"role": "user", "content": "OUTPUT (5 queries):"}
        ]
    )

    generated_queries = response.choices[0]["message"]["content"].strip().split("\n")
    return generated_queries


# Mock function to simulate vector search, returning random scores
def vector_search(query, all_documents):
    available_docs = list(all_documents.keys())
    random.shuffle(available_docs)
    selected_docs = available_docs[:random.randint(2, 5)]
    scores = {doc: round(random.uniform(0.7, 0.9), 2) for doc in selected_docs}
    return {doc: score for doc, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)}


# Reciprocal Rank Fusion algorithm
def reciprocal_rank_fusion(search_results_dict, k=60):
    fused_scores = {}
    print("Initial individual search result ranks:")
    for query, doc_scores in search_results_dict.items():
        print(f"For query '{query}': {doc_scores}")

    for query, doc_scores in search_results_dict.items():
        for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            previous_score = fused_scores[doc]
            fused_scores[doc] += 1 / (rank + k)
            print(
                f"Updating score for {doc} from {previous_score} to {fused_scores[doc]} based on rank {rank} in query '{query}'")

    reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    print("Final re-ranked results:", reranked_results)
    return reranked_results


# Dummy function to simulate generative output
def generate_output(reranked_results, queries):
    return f"Final output based on {queries} and re-ranked documents: {list(reranked_results.keys())}"


# Predefined set of documents (usually these would be from your search database)
all_documents = {
    "doc1": "marineTroopLeadersGuide.pdf",
    "doc2": "mcdpOneTwo.pdf",
    "doc3": "howTheArmyRuns.pdf",
    "doc4": "MCRP_3-01A_BasicMarksmanship.pdf",
    "doc5": "rangerHandbook.pdf",
    "doc6": "armyCompanyLeader.pdf",
    "doc7": "PAScol Reference Guide_2.pdf",
    "doc8": "RP0103_marineCorpsLeadership.pdf",
    "doc9": "westPointLeadershipLessons.pdf",
    "doc10": "airForceNCOleadership.pdf"
}

# Streamlit bot
st.title("Study bot using RAG Fusion Prompts!")
st.write("""
# Squad Leader's Chatbot
Welcome to the *Basic* App! 
""")

# Streamlit container for user question.
user_question = st.text_input("Please enter your study question here:")
if user_question:
    st.write(f"You asked: {user_question}")

    # Main function
    if __name__ == "__main__":
        original_query = user_question
        generated_queries = generate_queries_chatgpt(original_query)

        all_results = {}
        for query in generated_queries:
            search_results = vector_search(query, all_documents)
            all_results[query] = search_results

        reranked_results = reciprocal_rank_fusion(all_results)

        final_output = generate_output(reranked_results, generated_queries)

        print(final_output)

    with st.spinner("Creating a prompt by inferring what you mean to ask..."):
        response = final_output

    # Display the answer.
    st.subheader("Your Answer:")
    st.write(response)





