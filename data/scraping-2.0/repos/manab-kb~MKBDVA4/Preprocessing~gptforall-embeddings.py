import os
import json
import openai
from contentprep import completesubtutcontent

embeddingfile = "/Users/manabkb/Desktop/Manab Biswas/Pytch - Summer Internship/Model/Dataset/Extras/Simplified-Dataset/Embeddings/gptforall-embeddings.json"
embed = {}

openai.organization = os.getenv("OPENAIORG")
openai.api_key = os.getenv("OPENAIKEY")

index = 1

for element in completesubtutcontent:

    try: 
        print("Generating embedding for index {0}".format(index))
        embedding = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=str(element)
        )
        
        block = {"Tut_Text" : str(element), "Vector" : embedding['data'][0]['embedding']}
        writeblock = {index: block}
        embed.update(writeblock)

        wf = open(embeddingfile, "w")
        wf.write(json.dumps(embed, indent=4))
        wf.close()
        index += 1

    except:
        index += 1
        pass
