import requests
from langchain.llms import OpenAI
from constants import openai_key
import os

os.environ["OPENAI_API_KEY"]=openai_key
def get_most_technical_complex_repository_for_user(username):
    # Get a list of all public repositories for the user.
    url = "https://api.github.com/users/{}/repos".format(username)
    response = requests.get(url)
    if response.status_code == 200:
        repositories = response.json()
    else:
        return None

    # Initialize the LLM.
    lm = OpenAI(temperature=0.1)
    # Initialize the complexity metrics.
    complexity_metrics = {
        "cyclomatic_complexity": lambda repository: len(repository["complexity"]["cyclomatic"]) if "complexity" in repository else 0,
        "halstead_volume": lambda repository: repository["complexity"]["halstead"]["volume"] if "complexity" in repository else 0,
    }

    # Calculate the technical complexity of each repository.
    technical_complexities = []
    for repository in repositories:
        complexity = 0
        for metric, metric_fn in complexity_metrics.items():
            complexity += metric_fn(repository)
        technical_complexities.append((complexity, repository))

    # Rank the repositories by their technical complexity.
    technical_complexities.sort(key=lambda x: x[0], reverse=True)
    # print(technical_complexities)
    # print(len(technical_complexities))
    # Return the repository with the highest technical complexity.
    for i in technical_complexities:
        repo=i[1]
        print(repo["name"])
    return technical_complexities[0][1]

if __name__ == "__main__":
    username = "ipsitipsi"
    repository = get_most_technical_complex_repository_for_user(username)
    print(repository["name"])
