import openai
import os


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

api_key = os.getenv('OPENAI_API_KEY')

import base64
import requests

# OpenAI API Key
#api_key = "YOUR_OPENAI_API_KEY"

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "graph4.png"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
      { 
        "role": "system",
        "content": 
          {
            "type": "text",
            "text":"""你是计算机专家，这是一个读图像中的“图”数据结构的题,请识别图像中的“图”，然后回答图片上的问题。
            A graph is a collection of vertices and edges. An edge is a connection between two vertices (sometimes referred to as nodes). One can draw a graph by marking points for the vertices and drawing lines connecting them for the edges, but the graph is defined independently of the visual representation
            The edges of the above graph have no directions meaning that the edge from one vertex A to another vertex B is the same as from vertex B to vertex A. Such a graph is called an undirected graph. Similarly, a graph having a direction associated with each edge is known as a directed graph.

A path from vertex x to y in a graph is a list of vertices, in which successive vertices are connected by edges in the graph. For example, FGHE is path from F to E in the graph above. A simple path is a path with no vertex repeated. For example, FGHEG is not a simple path.

A graph is connected if there is a path from every vertex to every other vertex in the graph. Intuitively, if the vertices were physical objects and the edges were strings connecting them, a connected graph would stay in one piece if picked up by any vertex. A graph which is not connected is made up of connected components. For example, the graph above has two connected components: {A, B, D} and {C, E, F, G, H}.

A cycle is a path, which is simple except that the first and last vertex are the same (a path from a point back to itself). For example, the path HEGH is a cycle in our example. Vertices must be listed in the order that they are traveled to make the path; any of the vertices may be listed first. Thus, HEGH and EHGE are different ways to identify the same cycle. For clarity, we list the start / end vertex twice: once at the start of the cycle and once at the end.
            """
          },
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "这是一个读“图”的题，请读取图片上图，获取图的数据结构，然后回答图片上的问题，生成邻接表矩阵"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
        ]
      }
    ],
    "max_tokens": 2000
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
response_json=response.json()

content = response_json['choices'][0]['message']['content']
print(content)