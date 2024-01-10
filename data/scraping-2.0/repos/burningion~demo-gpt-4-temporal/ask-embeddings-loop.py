import pinecone
import openai
import os
import pprint

pp = pprint.PrettyPrinter(indent=2)

openai.api_key = os.environ['OPENAI_API_KEY']
embed_model = "text-embedding-ada-002"

index_name = os.environ['PINECONE_INDEX']
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],  # app.pinecone.io (console)
    environment=os.environ['PINECONE_ENVIRONMENT']  # next to API key in console
)
index = pinecone.GRPCIndex(index_name)

messagesList = []

print("Enter your system prompt context below. As an example it should be something like: \n'you are an experienced frontend developer who cares about readability'")
system_prompt = input("Leave blank for default: ")
if system_prompt == "":
    # system message to 'prime' the model
    primer = f"""You are Q&A bot. A highly intelligent system that answers
    user questions based on the information provided by the user above
    each question. If the information can not be found in the information
    provided by the user you truthfully say "I don't know".

    your answers should be great examples of clean, easy to read code
    """

messagesList.append({"role": "system", "content": system_prompt})
first_prompt = input("Enter your prompt: ")

res = openai.Embedding.create(
    input=[first_prompt],
    engine=embed_model
)

# retrieve from Pinecone
xq = res['data'][0]['embedding']

# get relevant contexts (including the questions)
res = index.query(xq, top_k=5, include_metadata=True)

pp.pprint(res)

contexts = [item['metadata']['text'] for item in res['matches']]

augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+first_prompt
messagesList.append({"role": "user", "content": augmented_query})

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messagesList
)

resContent = response["choices"][0]["message"]["content"]
pp.pprint(resContent)
messagesList.append({"role": "assistant", "content": resContent})

while True:
    next_prompt = input("Enter next prompt (q to quit): ")
    if next_prompt == "q":
        import csv
        with open('chatlog.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, messagesList[0].keys())
            writer.writeheader()
            for message in messagesList:
                writer.writerow(message)
        print("Wrote log of chat to `chatlog.csv`")
        exit()
    
    res = openai.Embedding.create(
        input=[next_prompt],
        engine=embed_model
    )
    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts (including the questions)
    res = index.query(xq, top_k=5, include_metadata=True)
    pp.pprint(res)
    contexts = [item['metadata']['text'] for item in res['matches']]

    augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+first_prompt
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messagesList
    )

    resContent = response["choices"][0]["message"]["content"]
    pp.pprint(resContent)
    messagesList.append({"role": "assistant", "content": resContent})
     
pp.pprint(res)