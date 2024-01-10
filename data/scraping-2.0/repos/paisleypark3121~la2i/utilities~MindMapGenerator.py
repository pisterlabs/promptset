import json
from openai import OpenAI
import time

import networkx as nx
import matplotlib.pyplot as plt

from io import BytesIO

template = """You are a helpful assistant that generates a coded Mind Map given a specific [topic] and a [context].
Each map has to contain a maximum of 3 concepts and all connections must be labelled.
The output has to be the NetworkX python code needed to produce the mind map. This output has to contain only the code needed without any import.
As an example, the output has to start with: 
mm = nx.Graph(); 
As an example, if the user asks for: 
[topic] atom
[context] An atom is the fundamental building block of matter, consisting of two main components: electrons and nucleus. Electrons are negatively charged subatomic particles that orbit the nucleus in specific energy levels or electron shells; the nucleus is the central, densely packed core of an atom, where most of its mass is concentrated and contains two types of particles: protons (positively charged subatomic particles) and neutrons (electrically neutral subatomic particles).

coded mind map:
G = nx.DiGraph()
G.add_node("atom", label="atom")
G.add_node("nucleus", label="nucleus")
G.add_node("protons", label="protons")
G.add_node("neutrons", label="neutrons")
G.add_node("electrons", label="electrons")
G.add_edge("atom", "nucleus", label="composition")
G.add_edge("nucleus", "protons", label="compositions")
G.add_edge("nucleus", "neutrons", label="composition")
G.add_edge("atom", "electrons", label="composition")
pos = nx.spring_layout(G)
node_labels = nx.get_node_attributes(G, 'label')
node_sizes = {node: len(label) * 500 for node, label in node_labels.items()}
font_size = 14
figure, ax = plt.subplots(figsize=(20, 15))
nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=list(node_sizes.values()), node_color="skyblue",
        font_size=font_size, edge_color="gray", nodelist=list(G.nodes()), ax=ax)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=font_size, font_color='red', ax=ax)'''
[topic]{topic}
[context]{context}

coded mind map:"""


def generateMindMap_mono_topic(name,text,temperature=0,model_name='gpt-4-0613'):

    #print(model_name)

    messages=[]
    messages.append(
        {
        "role": "system",
        "content": template
        }
    )

    user_content="{topic} "+name+" {context} "+text;
    messages.append(
        {
            "role":"user",
            "content": user_content
        }
    )

    #print(messages)

    client=OpenAI()
    
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=2000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    answer=response.choices[0].message.content

    timestamp = int(time.time())
    last2 = timestamp % 100
    suffix = str(last2)

    exec(answer)
    image_bytes_io = BytesIO()
    plt.savefig(image_bytes_io, format="png")
    image_bytes_io.seek(0)
    image_content = image_bytes_io.read()
    image_bytes_io.close()

    return image_content

    # file_name=name+"_"+suffix+".png"
    # answer=answer+"\nfigure.savefig(\""+file_name+"\")"

    # answer = answer.replace("figure", "fig" + suffix)\
    #     .replace("axx", "ax" + suffix)
    
    #print(answer)
    #exec(answer)

    #return file_name  

def test():
    from dotenv import load_dotenv
    load_dotenv()

    name="atom"
    text="i need a Mind Map for the topic: "+name
    response=generateMindMap_mono_topic(name,text)
    print(response)

    # name="solar system"
    # text="i need a Mind Map for the topic: "+name
    # response=generateMindMap_mono_topic(name,text)
    # print(response)

    # name="atom"
    # text="An atom is the basic unit of matter, consisting of a nucleus at its center, composed of positively charged protons and uncharged neutrons. Negatively charged electrons orbit the nucleus in shells or energy levels. The number of protons in the nucleus determines an element's identity and its chemical properties, while the overall number of electrons balances the positive charge of the protons, making the atom electrically neutral.Atoms are the building blocks of all chemical elements and are crucial to understanding the structure and behavior of matter in the universe."
    # response=generateMindMap_mono_topic(name,text)
    # full_response=response+"\nplt.savefig(\""+name+".png\")"
    # #print(full_response)
    # exec(full_response)

#test()