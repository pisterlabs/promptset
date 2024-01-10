import openai
from transformers import GPT2TokenizerFast
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")

db = client["DataCheck"]
collection = db["connector"]

db2 = client["testsql"]
collection2 = db2["dataset"]

db3 = client["Datacheckmodel"]
collection3 = db3["openaidata"]

openai.api_key = "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"

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

docs = ["phone", "email", "address"]
query = "phone"


def construct_context(query, document):
    return "<|endoftext|>{document}\n\n---\n\nThe above passage is related to: {query}".format(
        document=document, query=query
    )


def get_score(context, query, log_probs, text_offsets) -> float:
    SCORE_MULTIPLIER = 100.0

    log_prob = 0
    count = 0
    cutoff = len(context) - len(query)

    for i in range(len(text_offsets) - 1, 0, -1):
        log_prob += log_probs[i]
        count += 1

        if text_offsets[i] <= cutoff and text_offsets[i] != text_offsets[i - 1]:
            break

    return log_prob / float(count) * SCORE_MULTIPLIER


def search(query, documents, engine):

    prompts = [construct_context(query, doc.lower())
               for doc in [""] + documents]

    resps = openai.Completion.create(
        model=engine,
        prompt=prompts,
        temperature=1.0,
        top_p=1.0,
        max_tokens=0,
        logprobs=0,
        n=1,
        echo=True,
    )

    resps_by_index = {choice["index"]: choice for choice in resps["choices"]}

    scores = [
        get_score(
            prompts[i],
            query,
            resps_by_index[i]["logprobs"]["token_logprobs"],
            resps_by_index[i]["logprobs"]["text_offset"],
        )
        for i in range(len(prompts))
    ]

    # Process results
    scores = [score - scores[0] for score in scores][1:]

    return [
        {
            "object": "search_result",
            "document": document_idx,
            "score": round(score, 3),
        }
        for document_idx, score in enumerate(scores)
    ]


print(search(query=query, documents=docs, engine="davinci"))

# for scores documentation visit: https://platform.openai.com/docs/guides/search/understanding-scores


# =================================================================================================================

documents = collection.find()
# iterate over the documents and print their content
# for document in documents:
#     print(document)

# print(documents[0])

lis1 = list(documents[0].keys())
lis1.pop(0)
print("Lis1:", lis1)


document2 = collection2.find()
lis2 = list(document2[0].keys())
# lis1.remove('_id')
lis2.pop(0)
print(lis2)

changedattributes = {}
maxsimilarity = {}
docs = lis1

for i in lis2:
    maxsim = 0
    idx = -1
    tempdict = search(query=i.lower(), documents=docs, engine="davinci")
    print(tempdict)
    for j in tempdict:
        if (j['score'] > 360.0 and j['score'] > maxsim and (j['document'] not in maxsimilarity)):
            idx = j["document"]
            maxsim = j["score"]
    if i not in changedattributes and idx != -1:
        changedattributes[i] = lis1[idx]
        maxsimilarity[idx] = maxsim

print(changedattributes)
attributes = {}
for i in lis1:
    if i not in attributes:
        attributes[i] = ""

for j in lis2:
    if j not in attributes:
        if j not in changedattributes:
            attributes[j] = ""
dict1 = {}
for i in documents:
    for j in attributes:
        attributes[j] = ""
    templis = list(i.keys())
    templis.pop(0)
    # print(templis)
    # templis.pop(0)
    for j in templis:
        attributes[j] = i[j]
    if ("_id" in attributes):
        del attributes['_id']
    collection3.insert_one(attributes)

for i in document2:
    for j in attributes:
        attributes[j] = ""
    templis = list(i.keys())
    # print("2", templis)
    templis.pop(0)
    for j in templis:
        if j in changedattributes:
            attributes[changedattributes[j]] = i[j]
        else:
            attributes[j] = i[j]
    if ("_id" in attributes):
        del attributes['_id']
    collection3.insert_one(attributes)
