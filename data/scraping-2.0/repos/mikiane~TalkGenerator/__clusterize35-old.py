# imports
import numpy as np
import pandas as pd
import os
import openai
import time
import requests
from dotenv import load_dotenv
load_dotenv('.env')




openai.api_key = os.environ.get("OPENAI_API_KEY")


# load data
datafile_path = "testembeddings.csv"

df = pd.read_csv(datafile_path)
df["ada_embedding"] = df.ada_embedding.apply(eval).apply(np.array)  # convert string to numpy array
matrix = np.vstack(df.ada_embedding.values)
matrix.shape

from sklearn.cluster import KMeans

n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=5)
kmeans.fit(matrix)
labels = kmeans.labels_
df["Cluster"] = labels

#df.groupby("Cluster").Scenarios.mean().sort_values()

from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, perplexity=10, random_state=5, init="pca", learning_rate=500)
vis_dims2 = tsne.fit_transform(matrix)

x = [x for x, y in vis_dims2]
y = [y for x, y in vis_dims2]

#for category, color in enumerate(["purple", "green", "red", "blue"]):
for category, color in enumerate(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white', 
          'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 
          'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 
          'cornflowerblue', 'cornsilk', 'crimson', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 
          'darkgrey', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 
          'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 
          'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 
          'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 
          'goldenrod', 'gray', 'grey', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 
          'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 
          'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgrey', 'lightgreen', 
          'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 
          'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 
          'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 
          'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 
          'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 
          'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff']):
    xs = np.array(x)[df.Cluster == category]
    ys = np.array(y)[df.Cluster == category]
    plt.scatter(xs, ys, color=color, alpha=0.3)

    avg_x = xs.mean()
    avg_y = ys.mean()

    plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)
plt.title("Clusters identified visualized in language 2d using t-SNE")
plt.savefig('graph.png')

import openai

# Reading a review which belong to each group.
rev_per_cluster = 5

#Fichier de sortie
#file = open("/Users/michel/tmp/scenariosculturisesoutput.txt", "w")


for i in range(n_clusters):
    print(f"Cluster {i} Theme:", end=" ")

    reviews = "\n".join(
        df[df.Cluster == i]
        .structure.str.replace("Title: ", "")
        .str.replace("\n\nContent: ", ":  ")
        .sample(min(rev_per_cluster, len(df[df.Cluster == i])), random_state=5)
        .values
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {os.environ.get("OPENAI_API_KEY")}'
    }
    prompt=f'Générer une structure au même formet qui soit une moyenne des structures. \n\structures :\n"""\n{reviews}\n\n structure moyenne : ""'
    data = {
        'model': 'gpt-4',
        'messages': [
            {'role': 'user', 'content': prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    json_data = response.json()
    
    if 'choices' in json_data and len(json_data['choices']) > 0:
        message = json_data['choices'][0]['message']['content']
        print(message.strip())
    else:
        print("Error: 'choices' key not found or no choices available in the response")
        print(json_data)

    
    
    #file.write(response["choices"][0]["text"].replace("\n", ""))


    sample_cluster_rows = df[df.Cluster == i].sample(min(rev_per_cluster, len(df[df.Cluster == i])), random_state=5)
    summary_scenarios = ""
    for j in range(min(rev_per_cluster, len(sample_cluster_rows.structure.values))):
        summary_scenarios = summary_scenarios + "######### \n\n" + sample_cluster_rows.structure.values[j] + "\n\n"
    # ... rest of your code
 
        prompt=summary_scenarios + "tl;dr"   
        api_key = os.environ.get("OPENAI_API_KEY")
        url = 'https://api.openai.com/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.environ.get("OPENAI_API_KEY")}'
        }
        data = {
            'model': 'gpt-4',
            'messages': [
                {'role': 'user', 'content': prompt}
            ]
        }
        response = requests.post(url, headers=headers, json=data)
        json_data = response.json()

        if 'choices' in json_data and len(json_data['choices']) > 0:
            message = json_data['choices'][0]['message']['content']
            print(message.strip())
        else:
            print("Error: 'choices' key not found or no choices available in the response")
            print(json_data)

        
    #
    #print("Erreur : Echec de la création de la completion après 5 essais")
    #print(response["choices"][0]["text"].replace("\n", ""))
    #file.write(response2["choices"][0]["text"].replace("\n", ""))
    #file.close()
    
    
    
    
    
    