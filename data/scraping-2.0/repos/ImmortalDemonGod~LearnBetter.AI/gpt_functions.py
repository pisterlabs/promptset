import json
from knowledgeDB_search import _knowledgedb_search
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def write_file(filename, content):
    sure = input("Do you want to write to " + filename + "? (YES/NO) ")
    if sure == "YES":
        with open(filename, "w") as f:
            f.write(content)
        return "Successfully written file " + filename
    else:
        return "ERROR: You are not allowed to write to this file"


def search(queries: list, num_results: int = 5):
    documents_list = []
    for query in queries:
        documents = _knowledgedb_search(query, num_results)
        # Convert Document objects to a serializable format
        for doc in documents:
            doc_dict = doc if isinstance(doc, dict) else doc.__dict__
            generate_citation(json.dumps(doc_dict))  # Generate citation for each document
            documents_list.append(doc_dict)
    # Convert the list of dictionaries to a string
    return json.dumps(documents_list)


def get_jokes(number_of_jokes):
    return json.dumps(["Why don't scientists trust atoms? Because they make up all the things!", 'How did the computer get wasted? It took screenshots!', "Why don't skeletons fight other skeletons? They don't have the guts!"])


definitions = [
    {
        "name": "get_jokes",
        "description": "Gets jokes from the joke database",
        "parameters": {
            "type": "object",
            "properties": {
                "number_of_jokes": {
                    "type": "number",
                    "description": "Gets the specified number of jokes"
                }
            }
        },
        "required": ["number_of_jokes"]
    },
    {
        "name": "write_file",
        "description": "Writes content to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename to write to"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to file"
                }
            }
        },
        "required": ["filename", "content"]
    },
    {
        "name": "search",
        "description": "Searches the knowledge database",
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of search queries"
                },
                "num_results": {
                    "type": "number",
                    "description": "Number of results to return for each query"
                },
                "context": {
                    "type": "string",
                    "enum": ["default", "programming", "english_writing", "youtube_history"],
                    "description": "The context to use for the search"
                }
            }
        },
        "required": ["queries"]
    },
    {
        "name": "generate_questions",
        "description": "Generates 5 more questions based on the given question",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Question to generate more questions from"
                }
            }
        },
        "required": ["question"]
    },
    {
        "name": "get_citations",
        "description": "Gets all citations",
        "parameters": {
            "type": "object",
            "properties": {}
        },
        "required": []
    },
    {
        "name": "summarize_conversation",
        "description": "Summarizes the conversation using GPT-3 Turbo",
        "parameters": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string"},
                            "content": {"type": "string"}
                        }
                    },
                    "description": "List of messages in the conversation"
                }
            }
        },
        "required": ["messages"]
    },
]



def generate_questions(question):
    prompt = f"I'm going to ask a question. I need you to take my question and use it to generate 5 more questions where if all the questions generated were answered would lead to the greatest final understanding.\nQUESTION: {question}\n"
    response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=100, temperature=0.8)
    questions = response['choices'][0]['text'].strip().split('\n')
    return '\n'.join(questions)

import json

citations_list = []

def get_citations():
    # citations_list.clear()
    return json.dumps(citations_list)


def generate_citation(document):
    document = json.loads(document)
    metadata = document['metadata']
    prompt = f"Generate a citation for a document with the source , date {metadata['date']}, file name {metadata['file_name']}, and file SHA1 {metadata['file_sha1']}, and with the any additional metadata."
    response = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=100, temperature=0.5)
    citation = "ERROR: No choices in response"
    if 'choices' in response:
        citation = response['choices'][0]['text'].strip()
        citations_list.append(citation)
    #print(citations_list)
    return citation

def summarize_conversation(messages):
    summary_prompt = {
        "role": "system",
        "content": "Please act as a neutral conversational summarizer. Your role is to concisely recap the key points from a dialogue between an AI and a user. Focus on capturing the core ideas and relevant details without adding subjective opinions or interpretations. You may rephrase the content in your own words for clarity, while preserving the original meaning. If the dialogue references external sources, indicate those points with placeholders like (Source: Author, date) to note the reference without summarizing unverified content. Your goal is to objectively distill the conversation down to its salient points for the purposes of coherent, personalized dialogue. Avoid introducing biases or assumptions beyond what is directly stated.",
    }
    # Create a new list that includes the summary_prompt and the messages
    messages_with_prompt = [summary_prompt] + messages

    response = openai.ChatCompletion.create(
        messages=messages_with_prompt,
        model="gpt-3.5-turbo-16k",
    )

    if isinstance(response, dict) and 'choices' in response:
        summary = response["choices"][0]["message"]["content"]
    else:
        print("Error: Unexpected response type")
        return

    return summary

