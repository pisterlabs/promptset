from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader
from openai import OpenAI

# pre-specified tokens
company_name = "Ukraine Mental Help"
grant_sys_token = "You are a professional non-profit grant-writer for " + company_name + " who is persuasive through " \
                                                                                         "direct and precise presentation that connects with the grant-writers values."
sm_sys_token = "You are a social media manager for " + company_name + " who is writing an English post"
email_token = "You are writing a mass email template for outreach and/or fundraising purposes."


def load_model():
    embedding = OpenAIEmbeddings(model='text-embedding-ada-002')
    loader_json = TextLoader("UAMH_training_data.jsonl")
    loader_prop = TextLoader("grant_prop.txt")
    loaders = [loader_json, loader_prop]
    return VectorstoreIndexCreator(embedding=embedding).from_loaders(loaders)


def query(model, request_text, mode=None, length=None, values=None, platform=None, previous=None):
    initial_query = query_string = request_text

    if mode == "grants":
        query_string = (query_string + " " + grant_sys_token)
    elif mode == "social media":
        if platform is None:
            query_string = (query_string + " " + sm_sys_token + ".")
        else:
            query_string = (query_string + " " + sm_sys_token + " on " + platform + ".")
    elif mode == "emails":
        query_string = (query_string + " " + email_token)

    if length is not None:
        query_string = (query_string + " Limit answer to " + length + " words.")

    if values is not None:
        query_string = (query_string + " Please embody the values of " + values + ".")

    response = model.query(query_string)

    while response[-1] not in '!.*])' and 6000 > len(response) > 720:
        query_string = query_string + "Previously you've written the following text; continue where you left off: " + response + " Insert the string '****' at the end when you're done, so I can exit this loop."
        new = model.query(query_string)
        response = response + new

    if response[-1] == '*':
        return response[:-4]
    return gpt_4_process(response, initial_query)


def gpt_4_process(response, request_text):
    print("Response" + response)
    print("Query" + request_text)
    client = OpenAI()
    request = ''
    request = request + 'I prompted ada with the prompt: ' + request_text
    request = request + ' The model replied ' + response
    request = request + 'I want you to revise this text in precise accordance with the goals of the query prompt, ' \
                        'since ' \
                        'your writing is more natural than ada'
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an editor for grant-writing materials and marketing, who revises "
                                          "text written by OpenAI's ada embeddings model."},
            {"role": "user", "content": request}
        ]
    )
    return response.choices[0].message.content
