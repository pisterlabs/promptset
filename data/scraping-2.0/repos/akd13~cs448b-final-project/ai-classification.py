# we used the API as well as chat interface for GPT-4 to create our clusters
from openai import OpenAI

client = OpenAI(
    organization="my-id"
)

stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "I will give you a list of occupations"
                                          " and you come up with 20 clusters that each one of them can belong to"}],
    stream=True,
)
list_occupations = ["Plumber", "Educator", "Scientist"]
num_clusters = 20

clusters = []
for i in range(num_clusters):
    stream.send({"role": "user", "content": "I will give you a list of occupations"
                                            " and you come up with 20 clusters that each one of them can belong to"
                                            + str(list_occupations)})
    response = stream.next()
    clusters = response["choices"][0]["text"]

for occupation in list_occupations:
    stream.send({"role": "user", "content": "I will give you an occupation, assign it to one of the clusters in the "
                                            "list"
                                             + str(occupation) + str(clusters)})
    response = stream.next()
    cluster = response["choices"][0]["text"]
    # save the cluster for the occupation in a file
    with open("occupations.txt", "a") as f:
        f.write(occupation + ":" + cluster + "\n")

