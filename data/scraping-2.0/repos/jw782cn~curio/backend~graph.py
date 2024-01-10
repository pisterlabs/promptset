import json
import chatgpt
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain import PromptTemplate

class TreeNode:
    def __init__(self, id, value, name, children=None):
        self.id = id
        # value is the q & a related to this node
        self.value = value
        self.context = []
        self.children = children if children is not None else []
        self.name = name

def tree_to_json(node, x=0, y=0):
    nodes = []
    links = []

    if node is not None:
        node_json = {
            "id": str(node.id),
            "name": node.name,
            "symbolSize": 100,
            "x": x,
            "y": y,
            "value": node.value,
            "category": 0
        }
        nodes.append(node_json)

        y_gap = 10
        for i, child in enumerate(node.children):
            child_nodes, child_links = tree_to_json(child, x + 10, y + (i * y_gap))
            nodes.extend(child_nodes)

            links.append({"source": str(node.id), "target": str(child.id)})
            links.extend(child_links)

    return nodes, links

def convert_tree_to_json(tree, topic):
    nodes, links = tree_to_json(tree)

    json_data = {
        "nodes": nodes,
        "links": links,
        "categories": [{"name": topic}]
    }

    return json.dumps(json_data, indent=4)

def tree_to_text(node, indent=0):
    if node is None:
        return ""

    text = "    " * indent + "+-- " + str(node.id) + " " + node.name + "\n"

    for child in node.children:
        text += tree_to_text(child, indent + 1)

    return text

def get_text_from_file(filename):
    with open(filename, "r") as f:
        return f.read()

class Graph():
    def __init__(self, topic):
        self.topic = topic
        self.root = TreeNode(0, topic, topic)
        self.nodes = {0: self.root}
        # graph is json for frontend
        self.chatgpt = chatgpt.chatgpt()
         
    def get_graph(self):
        '''
        return json file for frontend
        '''
        return convert_tree_to_json(self.root, self.topic)

    def get_text(self):
        '''
        return text file for query
        '''
        return tree_to_text(self.root)

    def insert_node(self, parent_id, name, question, answer):
        '''
        insert new node to tree
        '''
        current_id = len(self.nodes)
        node = TreeNode(current_id, name, name)
        node.context.append(f'question: {question} \n answer: {answer}')
        print("insert node: ", node.id, node.name)
        self.nodes[parent_id].children.append(node)
        self.nodes[current_id] = node
        
    def update_node(self, node_id, question, answer):
        '''
        update existing node
        '''
        node = self.nodes[node_id]
        node.context.append(f'question: {question} \n answer: {answer}')
        print("update node: ", node.id, node.name)

    def update_graph(self, question, answer):
        '''
        update graph with new message (Q&A)
        1. get current knowledge tree structure
        2. ask chatgpt to insert new node or update existing node
        
            For the "insert" operation: 
            { "operation: "insert", "arg1": {parent_id}, "arg2": {summarized_topic} }
            For the "update" operation:
            { "operation: "update", "arg1": {updated_id}, "arg2": "" }
        '''
        system_prompt = get_text_from_file("backend/prompt/update_graph.txt")
        system_message = SystemMessage(content=system_prompt)
        user_prompt = f'current tree: \n {self.get_text()} \n\n===\n\ncurrent q&a:\nquestion:\n{question}\nanswer:\n{answer}'
        user_message = HumanMessage(content=user_prompt)
        messages = [system_message, user_message]
        response = self.chatgpt.chat_with_messages(messages)
        response = json.loads(response.content)
        print(response)
        response["arg1"] = int(response["arg1"])
        # update data
        self.update_data(response, question, answer)
        return response
    
    def update_data(self, operation, question, answer):
        '''
        For the "insert" operation: 
        {
            "operation: "insert",
            "arg1": {parent_id},
            "arg2": {summarized_topic},
        }
        For the "update" operation:
        {
            "operation: "update",
            "arg1": {updated_id},
            "arg2": "",
        }
        '''
        if operation["operation"] == "insert":
            self.insert_node(operation["arg1"], operation["arg2"], question, answer)
        elif operation["operation"] == "update":
            self.update_node(operation["arg1"], question, answer)
        print("updated data")
        
    def get_related_context(self, node_id):
        '''
        return related context for node_id
        '''
        contexts = self.nodes[node_id].context
        print("get related context for node: ", node_id)
        contexts = "\n".join(contexts)
        return contexts