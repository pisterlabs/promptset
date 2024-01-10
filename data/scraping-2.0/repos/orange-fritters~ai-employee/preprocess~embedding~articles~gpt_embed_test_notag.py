import openai
import json
import numpy as np
import os


def get_embedding(text):
    result = openai.Embedding.create(
        engine="text-embedding-ada-002",
        input=text,)
    return result["data"][0]["embedding"]


if __name__ == "__main__":
    # open log.txt and append to the log
    with open("preprocess/embedding/log_notag.txt", "a") as log:
        TEXT = "아르바이트로만 생활하고 있는데 정식으로 취업을 찾고 싶어."

        with open("server/config.json") as config_file:
            config_data = json.load(config_file)
            openai.api_key = config_data["chatgpt"]["secret"]

        embedding = np.load('data/embeds/embeddings_notag.npy')
        embed_json = json.load(open('data/embeds/embeddings_notag.json', 'r'))

        data_len = len(os.listdir('data/notags'))
        embed_len = len(embedding[0])

        embedding = embedding.reshape(data_len, embed_len)

        test_embed = get_embedding(TEXT)
        test_embed = np.array(test_embed)
        test_embed = test_embed.reshape(embed_len, 1)

        cosine_sim = np.dot(embedding, test_embed)
        top_10 = np.argsort(cosine_sim.flatten())[-10:]

        log.write("\n\nTEXT : " + TEXT + "\n")
        for i in top_10[::-1]:
            print(embed_json[str(i)]["title"], " ", embed_json[str(i)]["filename"], " : ", cosine_sim[i][0])
            log.write(embed_json[str(i)]["title"] + " " + embed_json[str(i)]
                      ["filename"] + " : " + str(round(cosine_sim[i][0], 3)) + "\n")
