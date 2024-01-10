from swissnyf.pipeline.base_pipeline import BasePipeline
from swissnyf.pipeline.base_pipeline import CodeSynth
import copy
import pickle
import json
import os
from llama_index.embeddings import OpenAIEmbedding
from llama_index.agent import ReActAgent
from llama_index.llms import AzureOpenAI
from abc import abstractclassmethod
from typing import Optional, Dict, List, Tuple, Any
import re
from collections import deque
from tqdm import tqdm
from llama_index.tools.function_tool import FunctionTool
from typing import List
from llama_index.agent.react.formatter import  get_react_tool_descriptions
from llama_index.llms.base import LLM

## Node
class Node:
    def __init__(self, name : str, children : list, previous_tool_name : str, is_api : bool, previous_argument : str):
        self.name = name
        self.children_list = children
        self.previous_tool_name = previous_tool_name
        self.is_api = is_api
        self.previous_argument = previous_argument

        
    def add_child(self, obj):
        self.children_list.append(obj) 

class ParseTree(object):
    def __init__(self):
        pass

    def clean_tree(self, node):
        if not node:
            return node
        child_list = list(filter(lambda x: x.is_api, node.children_list))
        new_child_list = []
        for i, child in enumerate(node.children_list):
            if child.is_api:
                new_child_list.append(self.clean_tree(child))
        node.children_list = child_list
        return node
    

    def print_tree(self, node, prefix="", last=True):
        # global answer
        if node is not None:
            connector = "└── " if last else "├── "
            if node.previous_argument is not None and node.name != 'None':
                self.answer += f"{prefix}{connector}{node.previous_argument}{':'}{node.name}\n"
            elif node.previous_argument is None:
                self.answer += f"{prefix}{connector}{node.name}\n"
            prefix += "    " if last else "│   "
            for i, child in enumerate(node.children_list):
                is_last = i == len(node.children_list) - 1
                self.print_tree(child, prefix, last=is_last)

    def parse(self, arg_tree):
        clean_root = copy.deepcopy(arg_tree.root)
        self.answer = ""
        self.print_tree(clean_root)
        return self.answer

## ActionTree
class ActionTree:

    def __init__(self, root_name):
        self.root = Node(name=root_name,children=[], previous_tool_name=None, is_api=True, previous_argument=None)
        self.nodes = {}
        self.nodes[root_name] = self.root
        self.leafs = []
        
    
    def create_node(self ,node_name, previous_tool_name, is_api, previous_argument) -> Node:
        new_node = Node(node_name, [], previous_tool_name, is_api, previous_argument)
        self.nodes[node_name] = new_node
        return new_node
    
    def add_child(self, child_name:str, previous_tool_name:str, is_api : bool, previous_argument : str) -> None:
        previous_tool = self.nodes[previous_tool_name]
        child_node = self.create_node(child_name, previous_tool_name, is_api, previous_argument)
        previous_tool.add_child(child_node)
        
    def parse_tree(self):
        queue = []
        vis = set()
        lay = set()

        
        queue.append((self.root,0))
        
        output = []
        while queue:
            node, layer = queue.pop(0)

            if layer not in lay:
                lay.add(layer)
            
            if node not in vis:
                api_dict = {}
                if node.is_api:
                    api_dict['tool_name'] = node.name
                    api_dict['arguments'] = []

                vis.add(node)

                if len(node.children_list) == 0:
                    self.leafs.append(node.name) 

                for child in node.children_list:
                    queue.append((child, layer+1)) 
                    if child.name !='None':
                        temp_dict = {}
                        temp_dict['argument_name'] = child.previous_argument
                        temp_dict['argument_value'] = child.name  
                        api_dict['arguments'].append(temp_dict)
                
                if node.is_api:
                    output.append(api_dict)
        output_list = list(reversed(output))
        prev = [out['tool_name'] for out in output_list]
        for out in output_list:
            for arg in out['arguments']:
                if arg['argument_value'] in prev:
                    arg['argument_value'] = f"$$PREV[{prev.index(arg['argument_value'])}]"

        return json.dumps(output_list, indent=4)
        
## ReverseChain
class ReverseChain(BasePipeline):
    
    FIND_LAST_API_PROMPT = """We have N APIs:
                =====
                {}
                =====
                If the query is: "{}"
                Which final API should we use for this instruction?
                Only return API code. Only return one word!
                """

    FIND_LAST_API_W_N_GOALS_PROMPT = """We have N APIs:
                =====
                {}
                =====
                If the query is: "{}"
                Characterize the given query into major goal/goals which require independent output. Individual goals can contain multiple API calls 
                but only split them 
                into two different goals only if both the task have different objective.
                Write these goals in a bulletin format like so:
                - goal 1 which requires API 1 for the final call, this list only contain the API for the Final call and not all the API required 
                  to solve it. 
                - goal 2 which requires API 2 for the final call, this list only contain the API for the Final call and not all the API required 
                  to solve it.

                Finally for individual of these goals give a list of APIs the is required for their Final Calls in the below format
                
                Final_API_List: API_1, API_2

                stop generation right after the above list
                """
    
    ARGUEMENT_EXTRACTOR_PROMPT ="""
                1. You are an argument extractor for an API, which is to be used to solve a given query. 
                2. For a given API, its arguments can be completed in 3 ways:
                             1. The argument value is hidden within some part of the query. 
                             2. The argument value is hidden in another API.
                             3. The allowed argument value is not relevant to the given query, complete it with "None".
                3. You have been given the query and the relevant APIs which you can use for completing the argument value.
                4. For each argument you need to choose the correct way to complete the argument value by understanding the query.
                5. In case argument value is hidden in another API, use the API name as the argument value.
                6. Use your understanding of the query to choose the correct argument completetion way. It heavily depends on the user query.
                7. Generate output in a correct json format with all the argument keys and values in proper data type under double quotes.
                      Example:
                        if argument_value data type is string or int:
                            "argument_name": "argument_value"
                        if argument_value data type is list:
                            "argument_name": "[argument_value]" 
                        if argument_value is None:
                            "argument_name": "None"
                8. Do not generate any other uncesseary text in the except the json output.
                
                Now,
    
                The given API is:
                {}, which was a requirement for {} for the field of {}.

                The Arguments to be extracted are:
                {} 
                The datatype of the Arguments are:
                {}
                The function and arguments description is as follows:
                {}
                The API you can use includes:
                {}

                The given query is: {}

                Arguments : """
   
    
    MERGE_TREES_PROMPT = """ 
                Let's consider a tree which can solve a given query. The parent nodes are a api function call and child nodes are it's argument. 
                The argument value can either be another api function or a some part of the query.

                The tree format is:
 
                Head Node
                ├── child_node_1_argument : child_node_1_argument_value
                │   ├── child_node_1.1_argument : child_node_1.1_argument_value
                │   └── child_node_1.2_argument : child_node_1.2_argument_value
                └── child_node_2_argument : child_node_2_argument_value
                    ├── child_node_2.1_argument : child_node_2.1_argument_value
                    │   └── child_node_2.1.1_argument : child_node_2.1.1_argument_value
                    └── child_node_2.2_argument : child_node_2.2_argument_value

                The above tree is iterated in the reverse order starting with the leaf nodes, the argument value is completed either through some api
                function call or with some information in the query. The child node arguments are completed first then their parent node function is called 
                using those arguments. The tree is traveresed in this fashion until the Head Node is called and results in the response to the query.

                Now, You are given multiple such trees: 
                lets say 
                tree_1: <--Tree Format-->
                tree_2: <--Tree Format-->
                ...

                1. The given trees are possible response to the given query but may or may not be correct.
                2. You have to evaluate all the given trees with respect to the query and generate a final single tree which can completely solve the query.
                3. By evaluating the trees you will be able to select which nodes are required to solve the query.
                4. Efficiently generate the final tree using the relevant nodes, such that there are no "redundant" or "duplicate" function calls.
                5. Read the query properly and then only choose the relevant function from the given trees. Choosing the functions from trees heavily depend on the 
                   query.
                6. Do not generate any child node/arguments that are not in any of the given tree. This is stricty not allowed.
                7. The final tree should be in the same format as the given trees.
                8. You have been provided the names of tools to help you generate the final tree.
                9. Do not generate any other text, only the tree.
                10. Head Node will be the final API call of the final tree. 

                Since you are very smart you can figure this out.
                
                Now,
                the query is : {}
                The list of tree is : {}
                The list of tools is : {}
                The final single tree is:
            
            """

    RETRIEVER_PROMPT = """
                    You are a query modifier. You will be given a query and a tool that has been used to solve the query.
                    You task is to modify the current query such that that the output query can be used to extract 
                    the arguments for the tool you have been given.
                    You should output multiple queries if you feel the multiple arguments need to be extracted.

                     Let's start with an example:
                     Input_Query: Please help Jack book a meeting room for 9am-10am.
                     Input_Tool: BookRoom
                     Input_Tool_Description: BookRoom is a tool used for BookingRooms for a person for a specified time duration.
                     Arguments_to_be_extracted = person_ID: str, room_ID: str, start_time: str, end_time: str
                     Output_Query: What time should Jack book the room?
                     Output_Query: What room should be booked for Jack?

                     Now your inputs are:
                     Input_Query: {input_query}
                     Input_Tool: {input_tool}
                     Arguments_to_be_extracted: {input_tool_desc}
                     now finish the output for:
                     Output_Query:
                     """

    
    code_synth = CodeSynth()
    _reverse_chain_corpus = []
    tool_defs = []
    all_tools = []
    def __init__(self, filter_method, llm):
        self.filter_method = filter_method
        self.llm = llm

    def set_tools(self, tool_descs:List[str], tool_names:List[str]):
        for tool_desc, tool_name in tqdm(zip(tool_descs,tool_names)):
            self.add_tool(tool_desc, tool_name)

    def init_tools(self):
        exec_string_tool_def = "\n\n\n".join(self._reverse_chain_corpus)
        exec(f"""{exec_string_tool_def}""")
    
    def __set_tools(self, tool_defs:List[str] ):
        self.tool_defs = tool_defs
        self._reverse_chain_corpus = self.tool_defs
        
    def add_tool(self, new_tool_desc, tool_name, max_retries=10):
        completed = False
        retries = 0
        while not completed and retries<max_retries:
            try:
                tool_def = self.code_synth.forward(tool_name, new_tool_desc, self.llm)
                formated_tool_def = self.format_tool_def(tool_def, tool_name)
                completed = True
            except Exception as e:
                
                print(f"Retrying with new tool def for {tool_name}, {e}")
                retries+=1
        
        self.tool_defs.append(formated_tool_def)
        self._reverse_chain_corpus.append(formated_tool_def)
        formated_tool_def = formated_tool_def.replace("@update_traverse_dict", "")
        exec(formated_tool_def)
        function_tool = FunctionTool.from_defaults(eval(tool_name))
        self.all_tools.append(function_tool)
    
    def format_tool_def(self, new_tool_def, tool_name):
        idx = new_tool_def.find(f"def {tool_name}")
        if idx==-1:
            raise Exception("This tool is not in a good format")
        if idx == 0 :
            new_tool_def = f"@update_traverse_dict\n{new_tool_def}"
        else:
            new_tool_def = f"{new_tool_def[:idx-1]}@update_traverse_dict\n{new_tool_def[idx:]}"
        return new_tool_def

    def modify_query(self, input: str, input_tool: str, input_tool_desc: str) -> str:
        """generate new queries for reverse chain"""
        prompt_input = self.RETRIEVER_PROMPT.format(input_query=input, input_tool=input_tool, input_tool_desc=input_tool_desc)
        res = self.llm.complete(prompt_input)
        res = res.text
        return res
    
    def fetch_plan_w_goal(self, query):
        fetch_plan_w_goals_prompt = self.FIND_LAST_API_W_N_GOALS_PROMPT.format(" ".join(get_react_tool_descriptions(self.all_tools)), query)
        res = self.llm.complete(fetch_plan_w_goals_prompt)
        find_key_word = "Final_API_List: "
        idx = res.text.find(find_key_word)
        if idx>=0:
            print(res.text)
            final_api_list = list(res.text[idx+len(find_key_word):].split(', '))
            return final_api_list
        else:
            return self.fetch_plan(query)

    def fetch_plan(self, query):
        first_system_prompt = self.FIND_LAST_API_PROMPT.format(" ".join(get_react_tool_descriptions(self.all_tools)), query)
        res = self.llm.complete(first_system_prompt)
        final_api = res.text
        return [final_api]

    def build_tree_for_f_api(self, final_api, query):
        tree = ActionTree(final_api)
        
        tools_req = deque()
        final_api_dict = {}
        final_api_dict['arg_name'] = final_api
        final_api_dict['previous_tool_name'] = None
        final_api_dict['previous_argument'] = None

        tools_req.append(final_api_dict)
        executed = set()

        while len(tools_req) != 0:
            tool = tools_req.popleft()
            tool_name_req = tool['arg_name']
            previous_tool_name = tool['previous_tool_name']
            previous_argument = tool['previous_argument']
            
            possible_tools = list(filter(lambda t: t.metadata.name==tool_name_req, self.all_tools))
            executed.add(tool_name_req)
            
            args_req = possible_tools[0].metadata.to_openai_function()["parameters"]["properties"]
            tool_desc = possible_tools[0]._metadata.description

            modified_queries = list(self.modify_query(query, tool_name_req, args_req).split('\n'))

            if len(modified_queries)==0:
                modified_queries.append(query)

            available_tools_name = []
            for q in modified_queries:
                available_tools_name.extend(self.filter_method.filter(q))
            available_tools_name = list(set(available_tools_name))  
            available_tools = list(filter(lambda t: t.metadata.name in available_tools_name, self.all_tools))
            
            if len(available_tools) == 0:
                print("No Tool can be retrieved for the given query")
                break

            if len(args_req) == 0:
                continue
            
            arg_retreiver = self.ARGUEMENT_EXTRACTOR_PROMPT.format(tool_name_req, previous_tool_name, previous_argument, 
                            list(args_req.keys()),list(args_req.values()), tool_desc, " ".join(get_react_tool_descriptions(available_tools)), query)                 
                                                                   
            res = self.llm.complete(arg_retreiver)
            res = res.text

            try:
                res = json.loads(res)
            except Exception as e:
                print(e)
                return None


            for key, arg in res.items():
                # print()
                if key not in args_req.keys():
                    continue
                flag = False
                for tool in self.all_tools:
                    if tool._metadata.name in arg:
                        arg_dict ={}
                        arg_dict['arg_name'] = tool._metadata.name
                        arg_dict['previous_tool_name'] = tool_name_req
                        arg_dict['previous_argument'] = key
                        tools_req.append(arg_dict)
                        flag = True
                        break
                if flag:    
                    tree.add_child(child_name=tool._metadata.name, previous_tool_name=tool_name_req, is_api=True, previous_argument=key)
                else:
                    tree.add_child(child_name=arg, previous_tool_name=tool_name_req, is_api=False, previous_argument=key)
        return tree
        
    def query(self, query) -> ActionTree:
        final_apis = self.fetch_plan_w_goal(query)
        trees = []
        for final_api in final_apis:
            tree = self.build_tree_for_f_api(final_api, query)
            if tree is not None:
                trees.append(tree)
        merged_tree = self.merge_trees(query, trees).replace("\\n", "\n")
        print(merged_tree)
        parsed_tree = self.parse_merged_tree(merged_tree, [tool.metadata.name for tool in self.all_tools])
        print(parsed_tree)
        print("-------------------------------------------------------------------")
        return merged_tree
    
    def merge_trees(self, query, trees):
        parsed_tree = []
        for tree in trees:
                parsed_tree.append(ParseTree().parse(tree))
        merged_tree = self.llm.complete(self.MERGE_TREES_PROMPT.format(query, parsed_tree, [tool.metadata.name for tool in self.all_tools])).text
        return merged_tree
        
    def parse_merged_tree(self, merge_tree, tool_names):
        STOP_CHAR = '─ '
        list_fn_args = []
        for i, ind in enumerate(merge_tree.split('\n')):
            j = ind.find(STOP_CHAR)
            if j!=-1:
                list_fn_args.append((ind[j+len(STOP_CHAR):], (i,j)) )
        
        def is_terminal(i1, j1, i2, j2):
            if i2>i1:
                if j2 > j1:
                    return False
                else:
                    return True
            else:
                return True
        i = 0
        prev_func_stack = []
        prev_arg_stack = []
        final_answer_list = []
        def fn_in_tool_list(fn_name, tool_list):
            for tool in tool_list:
                if tool in fn_name:
                    return True
            return False
        def get_tool_name(fn_name, tool_list):
            for tool in tool_list:
                if tool in fn_name:
                    return tool
            return ""
        def is_terminal(i1, j1, i2, j2):
            if i2>i1:
                if j2 > j1:
                    return False
                else:
                    return True
            else:
                return True
                
        stack_back_counter = 0
        prev_ind =0
        while i < len(list_fn_args):
            # print(list_fn_args[i])
            
            fn_or_arg = list_fn_args[i]
            fn_arg_list_ind = fn_or_arg[0].split(':')
            if len(fn_arg_list_ind)>2:
                fn_arg_list_ind = (fn_arg_list_ind[0], ":".join(fn_arg_list_ind[1:])) 
            
            if len(fn_arg_list_ind)==1:
                prev_func_stack.append((fn_arg_list_ind[0],fn_or_arg[1], []))
            elif fn_in_tool_list(fn_arg_list_ind[1], tool_names):
                if is_terminal(*prev_func_stack[-1][1],*fn_or_arg[1]):
                    # print("terminal call- last popped")
                    while prev_func_stack[-1][1][1] >= fn_or_arg[1][1]:
                        final_answer_list.append(prev_func_stack[-1])
                        prev_func_stack.pop(-1)
                    
                arg_dict = {
                        "argument_name": fn_arg_list_ind[0],
                        "argument_value": get_tool_name(fn_arg_list_ind[1], tool_names),
                }
                stack_back_counter+=1
                prev_func_stack[-1][-1].append(arg_dict)
                prev_func_stack.append((get_tool_name(fn_arg_list_ind[1], tool_names), fn_or_arg[1], []))
                
                
            else:
                if is_terminal(*prev_func_stack[-1][1],*fn_or_arg[1]):
                    while prev_func_stack[-1][1][1] >= fn_or_arg[1][1]:
                        final_answer_list.append(prev_func_stack[-1])
                        prev_func_stack.pop(-1)
                        
                arg_name = fn_arg_list_ind[0]
                arg_value = fn_arg_list_ind[1]
                arg_dict = {
                        "argument_name": arg_name,
                        "argument_value": arg_value,
                }
                
                prev_func_stack[-1][-1].append(arg_dict)
            i+=1
        
        while len(prev_func_stack)>0:
            final_answer_list.append(prev_func_stack[-1])
            prev_func_stack.pop(-1)
        
        merge_tool_map = {}
        for i, ind in enumerate(final_answer_list):
            merge_tool_map[ind[0]] = i
        for i, ind in enumerate(final_answer_list):
            for j, arg in enumerate(ind[-1]):
                if arg['argument_value'] in merge_tool_map:
                    final_answer_list[i][-1][j]['argument_value'] = f"$$PREV[{merge_tool_map[arg['argument_value']]}]"
        parsed_output = []
        for item in final_answer_list:
            parsed_output.append({"tool_name":item[0], "arguements":item[-1]})
        return json.dumps(parsed_output, indent=4)
        

    def save(self, action_tree, filename):
        file_tree = open(filename, 'wb')
        pickle.dump(action_tree,file_tree)
