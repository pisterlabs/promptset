import json
import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
import networkx as nx
from api import *
import random

class PromptBuilder:

    def __init__(self):
        os.environ["OPENAI_API_KEY"] = openai_key
        self.model = ChatOpenAI(model_name="gpt-3.5-turbo")

        # Data paths and data variables
        self.random_words_path = 'data/random_words.txt'
        self.states_path = "data/states.json"
        self.random_graph_names = []
        self.random_part_names = []
        self.states = []

        self.load_words()

        #Graph Variables    
        self.num_nodes, self.num_sample, self.common, self.unique_dict = None, None, None, None

        #Prompt Variables
        self.input_variables = []

    def initialize(self, num_nodes, num_sample, common, unique_dict):
        self.num_nodes = num_nodes
        self.num_sample = num_sample
        self.common = common
        self.unique_dict = unique_dict

    def load_words(self):
        try:
            words_file = open(self.random_words_path, 'r')
            self.random_words = words_file.read().split()
        except:
            print("Error loading random words")

        try:
            states_file = open(self.states_path, 'r')
            self.states = json.loads(states_file.read())
        except:
            print("Error loading states")
    
    def generate_prompt(self, graphs, all_inferences, unique_dict, common_pairs, type="base"):
        if type == "base":
            self.generate_base_prompt(graphs, all_inferences, unique_dict, common_pairs)
        elif type == "counterfactual":
            self.generate_counterfactual_prompt(graphs, all_inferences, unique_dict, common_pairs)
        elif type == "intervention":
            self.generate_intervention_prompt(graphs, all_inferences, unique_dict, common_pairs)
   
    def generate_base_prompt(self, graphs, all_inferences, unique_dict, common_pairs):
        template = ""
        num_graphs = len(graphs)
        template += "\nThere are {num_graphs} machines. "
        self.input_variables.append("num_graphs")

        #Constructing the title 
        template += "All machines have {num_nodes} parts, " 
        self.input_variables.extend(["num_nodes"])

        #Constructing the title - Enumerating possible parts 
        part_string = "a "
        part_list = ["{random_part_name_"+str(i)+"}, a " for i in range(self.num_nodes)]
        part_list[-1] = part_list[-1][:-4]
        part_string += "".join(part_list)
        template += part_string

        part_variables = ["random_part_name_"+str(i) for i in range(self.num_nodes)]
        self.input_variables.extend(part_variables)

        #Constructing the title  - Enumerating possible states
        state_string = ". Each part can be in one of {num_states} states: "
        state_list = ["{state_"+str(i)+"}, or " for i, state in enumerate(self.states)]
        state_list[-1] = state_list[-1][:-5]

        state_string += "".join(state_list)
        template += state_string

        state_variables = ["state_"+str(i) for i, state in enumerate(self.states)]
        state_variables.insert(0, "num_states")
        self.input_variables.extend(state_variables)

        template += ". "

        #Constructing all machine possibilities
        graph_variables = [] 
        for i, graph in enumerate(graphs):
            machine_exp = "\nIn a {random_graph_name_"+str(i)+"} machine: "
            counter = 0
            if graph.edges == []:
                for j, node in enumerate(graph.nodes):
                    machine_exp += "\n"+str(counter+1)+". when a {random_part_name_"+str(input_var)+"} is {state_0}, no other part changes state."
                    counter += 1

            for j in range(len(all_inferences[graph][0])):
                input_var = tuple(all_inferences[graph][0][j])
                output_var = all_inferences[graph][1][j]
                if input_var != output_var:
                    input_indices = [index for index, value in enumerate(input_var) if value == 1]

                    rel_exp = "\n"+str(counter+1)+". when "
                    for k, index in enumerate(input_indices):
                        if len(input_indices) == 1:
                            rel_exp += "the {random_part_name_"+str(index)+"} is {state_0} - "
                        elif k == len(input_indices) - 1:
                            rel_exp += "and the {random_part_name_"+str(index)+"} is {state_0} - "
                        else:
                            rel_exp += "the {random_part_name_"+str(index)+"} is {state_0}, "

                    output_indices = [index for index, value in enumerate(output_var) if value == 1] 

                    #Avoiding duplication of input and output variables. I.e if part 1 is inflated, we shouidn't say that it becomes inflated again
                    mod_output_indices = []
                    for value in output_indices:
                        if value not in input_indices:
                            mod_output_indices.append(value)
    
                    for k, index in enumerate(mod_output_indices):
                            if len(mod_output_indices) == 1:
                                rel_exp += "the {random_part_name_"+str(index)+"} becomes {state_0}. "
                            elif k == len(mod_output_indices) - 1:
                                rel_exp += "and the {random_part_name_"+str(index)+"} becomes {state_0}. "
                            else:
                                rel_exp += "the {random_part_name_"+str(index)+"} becomes {state_0}, "      
                    machine_exp += rel_exp
                    graph_variables.append("random_graph_name_"+str(i))
                    counter += 1

            for j, non_edge in enumerate(nx.non_edges(graph)):
                input_var = non_edge[0]
                output_var = non_edge[1]
                rel_exp = "\n"+str(counter+1)+". when the {random_part_name_"+str(input_var)+"} is {state_0}, the {random_part_name_"+str(output_var)+"} is unaffected. "
                machine_exp += rel_exp
                graph_variables.append("random_graph_name_"+str(i))
                counter += 1

            template += "\n"+machine_exp

        self.input_variables.extend(graph_variables) 
        
        #Constructing the question
        question = "\n\nLet's say that you have one of these machines, but don't know which one. The "

        #All parts start in off state
        for i in range(self.num_nodes):
            if i == self.num_nodes - 1:
                question += "and {random_part_name_"+str(i)+"} are all {state_1}. "
            else:
                question += "{random_part_name_"+str(i)+"}, "
        
        #Sample the change
        random_graph = random.choice(graphs)
        unique_inferences = unique_dict[random_graph]

        random_inference = random.choice(unique_inferences)
        input = random_inference[0]
        print("Inference:")
        print("Input", input)
        output = random_inference[1]
        print("Output", output)
        input_indices = [index for index, value in enumerate(input) if value == 1]
        output_indices = [index for index, value in enumerate(output) if value == 1]

        mod_output_indices = []
        for value in output_indices:
            if value not in input_indices:
                mod_output_indices.append(value)

        #Write out the change
        question += "Alice {state_transition_0} "
        for i, index in enumerate(input_indices):
            if len(input_indices) == 1:
                question += "the {random_part_name_"+str(index)+"}. "
            elif i == len(input_indices) - 1:
                question += "and the {random_part_name_"+str(index)+"}. "
            else:
                question += "the {random_part_name_"+str(index)+"}, "

        #Case if the input is the same as the output
        if input == output:
            question += "You see that nothing changes. "

        #Case if the input is different from the output
        if input != output:
            question += "You then see that "
        
            for i, index in enumerate(mod_output_indices):
                if i == len(mod_output_indices) - 1:
                    question += "and the {random_part_name_"+str(index)+"} {state_transition_0}. "
                else:
                    question += "the {random_part_name_"+str(index)+"} {state_transition_0}, "
        
        question += "Which machine do you have?"

        #Adding Transition variables
        state_transition_variables = ["state_transition_0"]
        self.input_variables.extend(state_transition_variables)

        template += question

        #Langchain Prompt
        graph_names, part_names, chosen_states, chosen_state_transitions = self.sample_random(num_graphs)
    
        prompt = PromptTemplate(
            input_variables=self.input_variables,
            template=template,
        )

        chain_dict = {}
        chain = LLMChain(llm = self.model, 
                         prompt = prompt)
        
        if "num_graphs" in self.input_variables:
            chain_dict["num_graphs"] = num_graphs
        if "num_nodes" in self.input_variables:
            chain_dict["num_nodes"] = self.num_nodes
        if "num_states" in self.input_variables:
            chain_dict["num_states"] = len(chosen_states)
        for i, word in enumerate(part_names):
            if f"random_part_name_{i}" in self.input_variables:
                chain_dict[f"random_part_name_{i}"] = word
        for i, state in enumerate(chosen_states):
            if f"state_{i}" in self.input_variables:
                chain_dict[f"state_{i}"] = state
        for i, state_transition in enumerate(chosen_state_transitions):
            if f"state_transition_{i}" in self.input_variables:
                chain_dict[f"state_transition_{i}"] = state_transition
        for i, word in enumerate(graph_names):
            if f"random_graph_name_{i}" in self.input_variables:
                chain_dict[f"random_graph_name_{i}"] = word

        print("Prompt:", prompt.format(**chain_dict))
        
        result = chain.run(chain_dict)
        print("\nResponse:", result)

        print("Correct Answer:", graph_names[graphs.index(random_graph)])

    def generate_counterfactual_prompt(self, graphs, all_inferences, unique_dict, common_pairs):
        template = ""
        num_graphs = len(graphs)
        template += "\nThere are {num_graphs} machines. "
        self.input_variables.append("num_graphs")

        #Constructing the title 
        template += "All machines have {num_nodes} parts, " 
        self.input_variables.extend(["num_nodes"])

        #Constructing the title - Enumerating possible parts 
        part_string = "a "
        part_list = ["{random_part_name_"+str(i)+"}, a " for i in range(self.num_nodes)]
        part_list[-1] = part_list[-1][:-4]
        part_string += "".join(part_list)
        template += part_string

        part_variables = ["random_part_name_"+str(i) for i in range(self.num_nodes)]
        self.input_variables.extend(part_variables)

        #Constructing the title  - Enumerating possible states
        state_string = ". Each part can be in one of {num_states} states: "
        state_list = ["{state_"+str(i)+"}, or " for i, state in enumerate(self.states)]
        state_list[-1] = state_list[-1][:-5]

        state_string += "".join(state_list)
        template += state_string

        state_variables = ["state_"+str(i) for i, state in enumerate(self.states)]
        state_variables.insert(0, "num_states")
        self.input_variables.extend(state_variables)

        template += ". "

        #Constructing all machine possibilities
        template += "All parts start off {state_1}."
        graph_variables = [] 
        for i, graph in enumerate(graphs):
            machine_exp = "\nIn a {random_graph_name_"+str(i)+"} machine: "
            counter = 0
            if graph.edges == []:
                for j, node in enumerate(graph.nodes):
                    machine_exp += "\n"+str(counter+1)+". when a {random_part_name_"+str(input_var)+"} is {state_0}, no other part changes state."
                    counter += 1

            for j in range(len(all_inferences[graph][0])):
                input_var = tuple(all_inferences[graph][0][j])
                output_var = all_inferences[graph][1][j]
                if input_var != output_var:
                    input_indices = [index for index, value in enumerate(input_var) if value == 1]

                    rel_exp = "\n"+str(counter+1)+". when "
                    for k, index in enumerate(input_indices):
                        if len(input_indices) == 1:
                            rel_exp += "the {random_part_name_"+str(index)+"} is {state_0} - "
                        elif k == len(input_indices) - 1:
                            rel_exp += "and the {random_part_name_"+str(index)+"} is {state_0} - "
                        else:
                            rel_exp += "the {random_part_name_"+str(index)+"} is {state_0}, "

                    output_indices = [index for index, value in enumerate(output_var) if value == 1] 

                    #Avoiding duplication of input and output variables. I.e if part 1 is inflated, we shouidn't say that it becomes inflated again
                    mod_output_indices = []
                    for value in output_indices:
                        if value not in input_indices:
                            mod_output_indices.append(value)
    
                    for k, index in enumerate(mod_output_indices):
                            if len(mod_output_indices) == 1:
                                rel_exp += "the {random_part_name_"+str(index)+"} becomes {state_0}. "
                            elif k == len(mod_output_indices) - 1:
                                rel_exp += "and the {random_part_name_"+str(index)+"} becomes {state_0}. "
                            else:
                                rel_exp += "the {random_part_name_"+str(index)+"} becomes {state_0}, "      
                    machine_exp += rel_exp
                    graph_variables.append("random_graph_name_"+str(i))
                    counter += 1

            for j, non_edge in enumerate(nx.non_edges(graph)):
                input_var = non_edge[0]
                output_var = non_edge[1]
                rel_exp = "\n"+str(counter+1)+". when the {random_part_name_"+str(input_var)+"} is {state_0}, the {random_part_name_"+str(output_var)+"} is unaffected. "
                machine_exp += rel_exp
                graph_variables.append("random_graph_name_"+str(i))
                counter += 1

            template += "\n"+machine_exp

        self.input_variables.extend(graph_variables)  

        #Constructing the question
        question = "\n\nLet's say that you have one of these machines, but don't know which one. You can take one of two actions to determine the machine."

        template += question

        # Picks a determining action and a meaningless action
        unique_action = []
        for graph in unique_dict:
            unique_action.append([x[0] for x in unique_dict[graph]])
        all_determining_action = self.find_intersection(unique_action)

        common_tuples = []
        for pair in common_pairs:
            if pair[0] != (0,0,0):
                common_tuples.append(pair[0])
        
        determining_action = None
        while True:
            determining_action = random.choice(all_determining_action)
            if determining_action not in common_tuples:
                break
        
        common_action = random.choice(common_tuples)

        #Construct action choices
        random_number = random.choice([0, 1])
        action1 = None
        action2 = None
        correct_action = None
        if random_number == 0:
            action1 = determining_action
            action2 = common_action
            correct_action = "Action 1"
        else:
            action1 = common_action
            action2 = determining_action
            correct_action = "Action 2"
        
        action_1_index = [index for index, value in enumerate(action1) if value == 1]
        action_prompt = "\nAction 1: You can "
        if len(action_1_index) == 0:
            action_prompt += "do nothing. "
        else:
            for i, action in enumerate(action_1_index):
                if i == len(action_1_index) - 1 and len(action_1_index) != 1:
                    action_prompt += "and change the state of {random_part_name_"+str(action)+"} to {state_0}. "
                else:
                    action_prompt += "change the state of {random_part_name_"+str(action)+"} to {state_0}, "
        
        action_prompt += "\nAction 2: You can "
        action_2_index = [index for index, value in enumerate(action2) if value == 1]
        if len(action_2_index) == 0:
            action_prompt += "do nothing. "
        else:
            for i, action in enumerate(action_2_index):
                if i == len(action_2_index) - 1:
                    action_prompt += "and change the state of {random_part_name_"+str(action)+"} to {state_0}. "
                else:
                    action_prompt += "change the state of {random_part_name_"+str(action)+"} to {state_0}, "
        
        template += action_prompt

        template += "\nWhich action would you take?"

        #Langchain Prompt
        graph_names, part_names, chosen_states, chosen_state_transitions = self.sample_random(num_graphs)
    
        prompt = PromptTemplate(
            input_variables=self.input_variables,
            template=template,
        )

        chain_dict = {}
        chain = LLMChain(llm = self.model, 
                         prompt = prompt)
        
        if "num_graphs" in self.input_variables:
            chain_dict["num_graphs"] = num_graphs
        if "num_nodes" in self.input_variables:
            chain_dict["num_nodes"] = self.num_nodes
        if "num_states" in self.input_variables:
            chain_dict["num_states"] = len(chosen_states)
        for i, word in enumerate(part_names):
            if f"random_part_name_{i}" in self.input_variables:
                chain_dict[f"random_part_name_{i}"] = word
        for i, state in enumerate(chosen_states):
            if f"state_{i}" in self.input_variables:
                chain_dict[f"state_{i}"] = state
        for i, state_transition in enumerate(chosen_state_transitions):
            if f"state_transition_{i}" in self.input_variables:
                chain_dict[f"state_transition_{i}"] = state_transition
        for i, word in enumerate(graph_names):
            if f"random_graph_name_{i}" in self.input_variables:
                chain_dict[f"random_graph_name_{i}"] = word

        print("Prompt:", prompt.format(**chain_dict))
        
        result = chain.run(chain_dict)
        print("\nResponse:", result)

        print("Correct Action:", correct_action)

    def generate_intervention_prompt(self, graphs,all_inferences, unique_dict, common_pairs):
        template = ""
        num_graphs = len(graphs)
        template += "\nThere are {num_graphs} machines. "
        self.input_variables.append("num_graphs")

        #Constructing the title 
        template += "All machines have {num_nodes} parts, " 
        self.input_variables.extend(["num_nodes"])

        #Constructing the title - Enumerating possible parts 
        part_string = "a "
        part_list = ["{random_part_name_"+str(i)+"}, a " for i in range(self.num_nodes)]
        part_list[-1] = part_list[-1][:-4]
        part_string += "".join(part_list)
        template += part_string

        part_variables = ["random_part_name_"+str(i) for i in range(self.num_nodes)]
        self.input_variables.extend(part_variables)

        #Constructing the title  - Enumerating possible states
        state_string = ". Each part can be in one of {num_states} states: "
        state_list = ["{state_"+str(i)+"}, or " for i, state in enumerate(self.states)]
        state_list[-1] = state_list[-1][:-5]

        state_string += "".join(state_list)
        template += state_string

        state_variables = ["state_"+str(i) for i, state in enumerate(self.states)]
        state_variables.insert(0, "num_states")
        self.input_variables.extend(state_variables)

        template += ". "

        #Constructing all machine possibilities
        template += "All parts start off {state_1}."
        graph_variables = [] 
        for i, graph in enumerate(graphs):
            machine_exp = "\nIn a {random_graph_name_"+str(i)+"} machine: "
            counter = 0
            if graph.edges == []:
                for j, node in enumerate(graph.nodes):
                    machine_exp += "\n"+str(counter+1)+". when a {random_part_name_"+str(input_var)+"} is {state_0}, no other part changes state."
                    counter += 1

            for j in range(len(all_inferences[graph][0])):
                input_var = tuple(all_inferences[graph][0][j])
                output_var = all_inferences[graph][1][j]
                if input_var != output_var:
                    input_indices = [index for index, value in enumerate(input_var) if value == 1]

                    rel_exp = "\n"+str(counter+1)+". when "
                    for k, index in enumerate(input_indices):
                        if len(input_indices) == 1:
                            rel_exp += "the {random_part_name_"+str(index)+"} is {state_0} - "
                        elif k == len(input_indices) - 1:
                            rel_exp += "and the {random_part_name_"+str(index)+"} is {state_0} - "
                        else:
                            rel_exp += "the {random_part_name_"+str(index)+"} is {state_0}, "

                    output_indices = [index for index, value in enumerate(output_var) if value == 1] 

                    #Avoiding duplication of input and output variables. I.e if part 1 is inflated, we shouidn't say that it becomes inflated again
                    mod_output_indices = []
                    for value in output_indices:
                        if value not in input_indices:
                            mod_output_indices.append(value)
    
                    for k, index in enumerate(mod_output_indices):
                            if len(mod_output_indices) == 1:
                                rel_exp += "the {random_part_name_"+str(index)+"} becomes {state_0}. "
                            elif k == len(mod_output_indices) - 1:
                                rel_exp += "and the {random_part_name_"+str(index)+"} becomes {state_0}. "
                            else:
                                rel_exp += "the {random_part_name_"+str(index)+"} becomes {state_0}, "      
                    machine_exp += rel_exp
                    graph_variables.append("random_graph_name_"+str(i))
                    counter += 1

            for j, non_edge in enumerate(nx.non_edges(graph)):
                input_var = non_edge[0]
                output_var = non_edge[1]
                rel_exp = "\n"+str(counter+1)+". when the {random_part_name_"+str(input_var)+"} is {state_0}, the {random_part_name_"+str(output_var)+"} is unaffected. "
                machine_exp += rel_exp
                graph_variables.append("random_graph_name_"+str(i))
                counter += 1

            template += "\n"+machine_exp

        self.input_variables.extend(graph_variables) 

        #Constructing the question
        question = "\n\nLet's say that you have one of these machines, but don't know which one. What machine do you have?"

        template += question

        #Intervention instructions
        intervention_instructions = "\nNote: You will need to try different actions in order to determine the blickets and the type of machine. Since you cannot see the machine, after you perform an action, wait for the user to give you the corresponding observation. "

        intervention_instructions += "Here are the possible actions:"

        for i, part in enumerate(part_list):
            intervention_instructions += "\nAction "+str(i+1)+": Change {random_part_name_"+str(i)+"} to {state_0}"
        
        intervention_instructions += "\n\nFollow the following pattern to interact with the user:\nThought: [Your thoughts]\nAction: [Your proposed action]\nObservation: [You must wait for the user to reply with the obervation]"

        template += intervention_instructions

        #Langchain Prompt
        graph_names, part_names, chosen_states, chosen_state_transitions = self.sample_random(num_graphs)
    
        prompt = PromptTemplate(
            input_variables=self.input_variables,
            template=template,
        )

        chain_dict = {}
        chain = LLMChain(llm = self.model, 
                         prompt = prompt)
        
        if "num_graphs" in self.input_variables:
            chain_dict["num_graphs"] = num_graphs
        if "num_nodes" in self.input_variables:
            chain_dict["num_nodes"] = self.num_nodes
        if "num_states" in self.input_variables:
            chain_dict["num_states"] = len(chosen_states)
        for i, word in enumerate(part_names):
            if f"random_part_name_{i}" in self.input_variables:
                chain_dict[f"random_part_name_{i}"] = word
        for i, state in enumerate(chosen_states):
            if f"state_{i}" in self.input_variables:
                chain_dict[f"state_{i}"] = state
        for i, state_transition in enumerate(chosen_state_transitions):
            if f"state_transition_{i}" in self.input_variables:
                chain_dict[f"state_transition_{i}"] = state_transition
        for i, word in enumerate(graph_names):
            if f"random_graph_name_{i}" in self.input_variables:
                chain_dict[f"random_graph_name_{i}"] = word

        print("Prompt:", prompt.format(**chain_dict))
        
        result = chain.run(chain_dict)
        print("\nResponse:", result)

    def sample_random(self, num_graphs, num_nodes=3):
        #Sample which random word to use for the graph names 
        random_graph_names = []
        for i in range(num_graphs + 1):
            random_graph_names.append(random.choice(self.random_words))

        #Sample which random word to use for the part names
        random_part_names = []
        for i in range(num_nodes):
            random_part_names.append(random.choice(self.random_words))
        
        #sample which state package to use
        state_package = random.randint(0, len(self.states) - 1)
        chosen_states = (self.states[str(state_package)]["on"], self.states[str(state_package)]["off"])
        chosen_state_transitions = (self.states[str(state_package)]["turning_on"], self.states[str(state_package)]["turning_off"])

        return(random_graph_names, random_part_names, chosen_states, chosen_state_transitions)
    
    def find_intersection(self, lists):
        if not lists:
            return []

        # Start with the first list as the initial intersection
        intersection = lists[0]

        # Iterate through the remaining lists
        for lst in lists[1:]:
            # Compare each element with the elements in the current intersection
            intersection = [x for x in intersection if x in lst]

        return intersection

if __name__ == '__main__':
    p = PromptBuilder()
    p.initialize(3, 1, 1, 1)
    print(p.sample_random(3, 3))
