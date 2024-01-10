# ./oai_assistants/query_gpt_4t_from_documents_threads.py
from openai import OpenAI
from credentials import oai_api_key




client = OpenAI(api_key=oai_api_key)


def get_response_from_gpt_4t(question, context):
    """
    Queries the GPT-4T model with a specific question and context.

    Args:
    question (str): The question to be asked.
    context (str): The context to be used for answering the question.

    Returns:
    str: The response from the GPT-4T model.
    """
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "You are the Q&A based on knowledge base assistant.\nYou will always review and refer to the pages included as context. \nYou will always answer from the pages.\nYou will never improvise or create content from outside the files.\nIf you do not have the answer based on the files you will clearly state that and abstain from answering.\nIf you use your knowledge to explain some information from outside the file, you will clearly state that.\n"
            },
            {
                "role": "user",
                "content": f"You will answer the following question with a summary, then provide a comprehensive answer, then provide the references aliasing them as Technical trace: \nquestion: {question}\npages:{context}"
            }
        ],
        temperature=0.5,
        max_tokens=4095,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    answer = response.choices[0].message.content
    return answer


def format_pages_as_context(file_ids):
    """
    Adds specified files to the question's context for referencing in responses.

    Args:
    file_ids (list of str): List of file IDs to be added to the assistant.
    """
    context = ""
    for file_id in file_ids:
        chosen_file_path = f"/Users/roland/code/Nur/content/file_system/{file_id}.txt"
        # Open file and append file to context
        with open(chosen_file_path, 'r') as file:
            context += file.read()
        print(f"File {file_id} appended to context successfully")

    return context


def query_gpt_4t_with_context(question, page_ids):
    """
    Queries the assistant with a specific question, after setting up the necessary context by adding relevant files.

    Args:
    question (str): The question to be asked.
    page_ids (list): A list of page IDs representing the files to be added to the assistant's context.

    Returns:
    list: A list of messages, including the assistant's response to the question.
    """
    # Format the context
    # Ensure page_ids is a list
    if not isinstance(page_ids, list):
        page_ids = [page_ids]
    context = format_pages_as_context(page_ids)
    # Query GPT-4T with the question and context
    response = get_response_from_gpt_4t(question, context)
    return response


if __name__ == "__main__":
    response = query_gpt_4t_with_context("Do we support payment matching in our solution? and if the payment is not matched "
                                 "do we already have a way to notify the client that they have a delayed payment?",
                                 ["458841", "491570"])
    print(response)