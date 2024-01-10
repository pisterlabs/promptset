
from openai import OpenAI
from time import sleep
from dotenv import load_dotenv
import requests
import os

load_dotenv()
client = OpenAI()


def Mistral7x8BResponse(prompt, messages):
    """
    This is expiremental and not used in the demo
    
    """
    s = requests.Session()

    api_base = os.getenv("ANYSCALE_BASE_URL")
    token = os.getenv("ANYSCALE_API_KEY")
    url = f"{api_base}/chat/completions"
    body = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": messages,
        "temperature": 0.7
    }

    with s.post(url, headers={"Authorization": f"Bearer {token}"}, json=body) as resp:
        response = resp.json()
        content = response['choices'][0]['message']['content']
        return content


def OpenAiAssistantResponse(prompt: str, style_of_response: str, quizz_length: int):

    assistantID = "asst_ScGfhT0u4MomilSUCTuKnRWF"

    thread = client.beta.threads.create()

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt,
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistantID,
        instructions="""
        You are a quizz creator that creates quizzes form the provided text ONLY. 
        The questions should be in the style of {style_of_response}.
        If multiple choice, the number of choices should be 4 marked a) b) c) d).
        The quizz length should be {quiz_length}.
        You provide questions and only questions, never bread text or paragraphs that are not questions.
        Do NOT provide the answers to the questions.

        If the answer cannot be found in the articles, write 'I could not find an answer from the provded documents.'
        If the question is not about creating a quizz, write 'I cannot create a quizz from this questiosn.'
        Try your best to always create a quizz.
        Try to make the quiz informative and interesting, something that will teach you if you answer it.
        Respond with onlt the quiz
        """,
    )

    while run.status != "completed":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )
        sleep(2)

    # retrieve and format mesasge
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    message = messages.data[0]

    # Extract the message content
    message_content = message.content[0].text
    annotations = message_content.annotations
    citations = []

    # this is copied from the openai docs
    # Iterate over the annotations and add footnotes
    for index, annotation in enumerate(annotations):
        # Replace the text with a footnote
        message_content.value = message_content.value.replace(
            annotation.text, f' [{index}]')

        # Gather citations based on annotation attributes
        if (file_citation := getattr(annotation, 'file_citation', None)):
            cited_file = client.files.retrieve(file_citation.file_id)
            citations.append(
                f'[{index}] {file_citation.quote} from {cited_file.filename}')
        elif (file_path := getattr(annotation, 'file_path', None)):
            cited_file = client.files.retrieve(file_path.file_id)
            citations.append(
                f'[{index}] Click <here> to download {cited_file.filename}')
            # Note: File download functionality not implemented above for brevity

    # Add footnotes to the end of the message before displaying to user
    message_content.value += '\n' + '\n'.join(citations)
    return message_content.value

def update_openai_api(api_key):
    # print("updating api key", api_key)
    client.api_key = api_key


if __name__ == "__main__":
    print(Mistral7x8BResponse("Ping!"))
