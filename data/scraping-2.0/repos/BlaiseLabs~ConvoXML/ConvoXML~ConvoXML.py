import sqlite3
from bs4 import BeautifulSoup
import uuid
import re
import random
import openai
import google.generativeai as palm
from .AgentInterface import AgentInterface
from .Agents import TestAgent, TestModerator, PalmAgent, PalmDeveloper, OpenAIAgent, OpenAIDeveloper
from .Context import Context



class ConvoXMLParser:

    """
      Parses xml and creates agents
    """

    # Define a dictionary to store class name to class object mappings
    class_mapping = {
        'TestAgent': TestAgent,
        'TestModerator': TestModerator,
        'PalmAgent': PalmAgent,
        'PalmDeveloper': PalmDeveloper,
        'OpenAIAgent': OpenAIAgent,
        'OpenAIDeveloper': OpenAIDeveloper,
    }

    default_agent = TestAgent

    def __init__(self, xml_string, db_path=None, context=None, agent_classes=[], **kwargs):
        

        self.xml_string = xml_string
        self.context = context or Context()
        self.context.openai_key = kwargs.get('openai_key')
        self.context.palm_key = kwargs.get('palm_key')
        self.db_path = db_path or 'messages.db'
        self.db_connection = self.setup_database(self.db_path)
        for agent_class in agent_classes:
                self.register_agent_class(agent_class)
        self.soup = BeautifulSoup(self.xml_string, features="html.parser")
        self.convo_loop = self.get_convo()    
        self.agents = self.get_agents()
        self.queue = self.parse_queue()

    def get_convo(self):

        convo_loop = self.soup.find(re.compile(r'convoloop', re.IGNORECASE))
        return convo_loop

    def parse_queue(self):
        queue = []
        branch = []
        for child in self.convo_loop.findChildren(recursive=False):
            agent = self.get_agent_by_role(child.name)
            queue.append(agent)
            for grand_child in child.findChildren(recursive=False):
                if grand_child.name.lower() == 'case':
                    nested_agent = self.get_agent_by_role(grand_child.get("role"))
                else:
                    nested_agent = self.get_agent_by_role(grand_child.name)
                nested_agent.thread_id = agent.thread_id
                agent.children.append(nested_agent)
                branch.append(nested_agent)
            if len(branch) > 0:
                queue.append(branch)
                branch = []
        return queue


    def get_agent_by_role(self, name):
        for agent in self.agents:
            if agent.role.lower() == name.lower():
                return agent
        return None


    def register_agent_class(self, AgentClass):
        class_name = AgentClass.__name__
        if class_name in self.class_mapping.keys():
            return None
            #raise ValueError(f"Agent class '{class_name}' already exists in the class mapping.")
        self.class_mapping[class_name] = AgentClass


    def get_agents(self):
        roles = self.soup.find_all('role')
        agents = []
        for role in roles:
            # Create a dictionary to hold role attributes and values
            role_attrs = {'role': role["name"]}

            # Add any additional attributes from the XML dynamically
            for attr in role.attrs:
                role_attrs[attr] = role[attr]

            # Extract specific attributes if they exist
            input_element = role.find('input')
            output_element = role.find('output')

            if input_element:
                role_attrs['input_table'] = list(input_element.get('table'))
                role_attrs['rows'] = input_element.get('rows').split(',')

            if output_element:
                role_attrs['output_table'] = output_element.get('table')

            # Assign the database connection and context
            role_attrs['db_path'] = self.db_path
            role_attrs['context'] = self.context

            # Instantiate the agent with dynamic attributes
            # Instantiate the agent with dynamic attributes using class_mapping
            if 'class' in role.attrs.keys():
                class_name = role['class'][0]
                agent_class = self.class_mapping.get(class_name, self.default_agent)
            else:
                agent_class = self.default_agent
            agent = agent_class(**role_attrs)
            agents.append(agent)

        return agents

    def get_agent_queue(self):
        element_queue = self.parse_queue()
        agent_queue = []
        for idx,action in enumerate(element_queue):
            agent_queue.append(action)
        return agent_queue




    def handle_branch(self, result, branch):
        """
        Handles a branch of the conversation queue
        """
        results = []
        for agent in branch:
            if result in agent.role:
                results.append(agent.execute())
        return results




    def run(self):
        queue = self.queue        
        self.context.exit = False
        while not self.context.exit:
            result = None
            for action in queue:
                if type(action) == list and result != None:
                    self.handle_branch(result, action)
                else:
                    result = action.execute()





    # Setting up the SQLite database and test data
    def setup_database(self, db_path=None):
        db_path = db_path or self.db_path
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        # Create the Messages table
        cursor.execute('''CREATE TABLE IF NOT EXISTS Messages (
                              message_id INTEGER PRIMARY KEY,
                              thread_id TEXT,
                              sender_id INTEGER,
                              content TEXT,
                              timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # Inserting some test data
        cursor.execute("INSERT INTO Messages (thread_id, sender_id, content) VALUES (?, ?, ?)", ('thread1', 1, 'Test message 1'))
        cursor.execute("INSERT INTO Messages (thread_id, sender_id, content) VALUES (?, ?, ?)", ('thread1', 2, 'Test message 2'))
        cursor.execute("INSERT INTO Messages (thread_id, sender_id, content) VALUES (?, ?, ?)", ('thread2', 1, 'Test message 3'))

        return connection
