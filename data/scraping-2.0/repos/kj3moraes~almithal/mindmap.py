import os
import openai
import json
import graphviz
import streamlit as st

class MindMap:
    
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
    def get_connections(self, text_chunks_libs:dict) -> list:
        
        state_prompt = open("./prompts/mindmap.prompt")
        PROMPT = state_prompt.read()
        state_prompt.close()
        
        final_connections = []
        for key in text_chunks_libs:
            for text_chunk in text_chunks_libs[key]:
                PROMPT = PROMPT.replace("$prompt", text_chunk)
                
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt = PROMPT,
                    temperature=0.5,
                    max_tokens=2048,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )
                
                relationships = response.choices[0].text
                final_string = '{"relations":' + relationships + '}'
                data = json.loads(final_string)
                relations = data["relations"]
                final_connections.extend(relations)
        return final_connections
            
        
    def generate_graph(self, text_chunks_libs:dict):
        graph = graphviz.Digraph()
        all_connections = self.get_connections(text_chunks_libs)
        for connection in all_connections:
            from_node = connection[0]
            to_node = connection[2]
            graph.edge(from_node, to_node)
        st.graphviz_chart(graph)