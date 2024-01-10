# encoding: utf-8
# Author: Yixuan
# 

import sqlite3
import os
import time
#from prompt_toolkit import prompt
import openai
import requests
import json
from wiktionaryparser import WiktionaryParser

class DialogHistoryManager:
    """
    It stores the dialog history in a sqlite database. 
    When a user sends a query, it will read the last 7k tokens from the database, 
    and form them into a list of messages.
    Args:
        prefix: the role of the chatbot 
        file_path: the path of the pdf file, or “index" at the server side 
        thresh: the max number of tokens to be read from the database.
            It's set to 7k because the default model is gpt4-8k.    
    TODO:
    - reset the database
    """
    def __init__(self, 
                 file_path="", 
                 prefix="",
                 thresh=7000):
        self.messages= []
        self.sys_messages = {"role": "system", "content": prefix}
        self.token_count = 0
        self.thresh = thresh
        self.file_path = file_path 
 
        # if file does not exist, create a sqlite database 
        if not os.path.exists(self.file_path):
            self._create_db()
        self._read_dialog()
    
    def _create_db(self):
        """Create a sqlite database 
        Create a table called "history" with four columns: role, content, int id(autoincrement), timestamp
        """
        conn = sqlite3.connect(self.file_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def _read_dialog(self):
        """Read dialog history from the sqlite database
        each row has two columns: role and content
        each row should be represented as a dictionary
        load rows into self.messages
        """
        conn = sqlite3.connect(self.file_path)
        c = conn.cursor()
        c.execute("SELECT role, content FROM history ORDER BY timestamp DESC LIMIT 500")
        # search in reverse order
        rows = c.fetchall()
        for row in rows:
            self.token_count += len(row[1].split())
            # print(self.token_count, row[1])
            if self.token_count > self.thresh:
                break
            self.messages.append({"role": row[0], "content": row[1]})
        self.messages.reverse()

    def form_user_msg(self, query):
        return {"role": "user", "content": query}

    def _add_message(self, role, message):
        self.token_count += len(message.split())
        self.messages.append({"role": role, "content": message})
        conn = sqlite3.connect(self.file_path)
        c = conn.cursor()
        c.execute("INSERT INTO history (role, content) VALUES (?, ?)", (role, message))
        conn.commit()
        conn.close()
    
    def add_assistant_message(self, message):
        self._add_message("assistant", message)
      
    def add_user_message(self, message):
        self._add_message("user", message)

    def get_message_for_api(self):
        # TODO: need another final check, in case the token count is still larger than the threshold
        # first clear the history if the current token count is larger than the threshold
        while self.token_count > self.thresh:
            first_message = self.messages.pop(0)
            self.token_count -= len(first_message["content"].split())
        messages = [self.sys_messages] + self.messages
        return messages

# TODO: global
wikiparser = WiktionaryParser()

class DialogManager:
    """ 
    DialogManager will preprocess the user query, retrieve the latest history, 
    form a list of messages, and send the list to the openai api, and return 
    the response.
    """
    def __init__(self, 
                file_path="", 
                major="",
                thresh=7000) -> None:
        prefix=f"""
            You are a computational linguistics expert and native English speaker. 
            Assist Master's students majored in {major}, 
            non-native with at least C1 English proficiency, 
            by providing clear and straightforward answers 
            to their course-related questions.
            """
        prefix = prefix.strip().replace("\n", "")
        self.history = DialogHistoryManager(file_path=file_path, 
                                            prefix=prefix, 
                                            thresh=thresh)
        
    def get_definition_via_wiktionary(self, query):
        word = wikiparser.fetch(query)
        print(type(word), word)
        if len(word) == 0:
            return ""
        definitions = word[0]["definitions"]
        
        def_text = ""
        for i, definition in enumerate(definitions, 1):
            def_text = ' '.join(definition['text'])
            def_text = ' '.join(definition['text']).replace("\n", "")
            def_text += f"Definition {i}: {def_text}\n"
        
        def_text = def_text.strip()
        if len(def_text) == 0:
            return ""

        # rephrase the definition in an easy-to-understand way
        prompt = f""" Here is the definition of {query} from wikitionary: {def_text}.
        Very shortly summarize them, then select the one relevant to computational linguistics, and rephrase it to be easily understood by the student.
        Start with:
        Some definitions from wiktionary are: ...
        """

        return prompt 
    
        
    def type_explain(self, query):
        """ 
        It first fetches the definition of the query from wiktionary,
        if the result is empty, do few-shot learning. 
        """
        prompt = self.get_definition_via_wiktionary(query)
        if len(prompt) > 0:
            return prompt
        
        prompt = f"""
                Input:  
                Define "morpheme".  
                Output:  
                A morpheme is the smallest meaningful unit in a language.

                Input:  
                Define "syntax".  
                Output:  
                Syntax refers to the rules that govern sentence structure in a language.

                Input:  
                Define "semantic analysis".  
                Output:  
                Semantic analysis involves understanding the meaning of text.

                Input:  
                Define "{query}".
                Output:  
                """
        return prompt

    def type_exemplify(self, query, major):
        """
        "Exemplify" provides an example to the selected text, it aims to help
        students understand the motivation or the goal of some concepts by generating tailored examples for students from different majors. 
        """
        prompt = f"""
        Input: vector space model, Major: Literature and Critical Theory
        Example: Imagine analyzing 'Jane Eyre' and 'Moby Dick'. Using a vector space model, each novel can be represented by a vector where each dimension corresponds to a unique word, enabling a comparison of thematic elements based on word.

        Input: syntactic tree, Major: Computer Science
        Example: Imagine analyzing a complex algorithm's pseudo-code. A syntactic tree can illustrate the hierarchical structure, with main functions as roots and sub-functions as branches, elucidating the code's execution order and dependencies.

        Input: cosine similarity, Major: General Linguistics
        Example: Imagine comparing two sentences: "I love apples" and "Apples are loved by me". Despite the difference in structure, their meaning is similar. Cosine similarity would measure this closeness by considering the sentences as vectors of words and calculating the cosine of the angle between them, indicating how similar the sentences are in content.

        Input: {query}, Major: {major} 
        Exemplify: """

        return prompt
    
    def type_detect(self, query, major):
        return f"""
        
        Input: 'We would need a very, very large number of training examples to estimate that many parameters.'
        CL Terms: training examples, estimate, parameters. Would you like further clarification on any of the identified terms?

        Input: 'Consider a formal version of the English language modeled as a set of strings of words. Is this language a regular language?'
        CL Terms: language model, regular language. Would you like further clarification on any of the identified terms?

        Input: 'Freeze the representation models and use them as feature extractors, or fine-tune the representation models on downstream tasks.'
        CL Terms: freeze, representation models, feature extractors, fine-tune. Would you like further clarification on any of the identified terms?

        Input: {query} 
        CL Terms: """

    def type_simplify(self, query):
        """
        Here it seems like directly providing the task description is better
        than using the few-shot learning examples. Possible reason is manual simpliefied version only reflects the English level of the creator. 
        Hence the generated one could still be too formal for some non-native speakers.
        In our experiment, even the term is from the same study field,
        the non-native speaker might not be familar with its English translation. Answers of only providing the translation tends to break down more terms than few-shot examples do. 
        However, if the original text is about defining a term, preserving the terminology is important. So we still include the "preserving terminology" in the task description.

        Probably it makes more sense to ask the user to provide a simplified version, and then use the few-shot learning examples to help the user.
        """
        return f"""
        Simplify sentences by preserving terminology, simplifying sentence structures, and using basic vocabulary. The goal is to ensure technical accuracy and improved accessibility for non-native speakers.
        Input: {query}
        Simplified Version:
        """
    
    def get_query_type(self, query):
        if query.startswith("::explain::"):
            return "explain", query[11:]
        elif query.startswith("::simplify::"):
            return "simplify", query[12:]
        elif query.startswith("::detect::"):
            return "detect", query[10:]
        elif query.startswith("::eg::"):
            return "exemplify", query[6:] 
        else:
            return "open", query
        
    def preprocess_query(self, query, major):
        """
        Five functions:
        - Explain: a term or a phrase 
        - Exemplify: provide an example to the selected text
        - Detect: detect ambiguious and confusing text
        - Simplify
        - Open question 
        Note that, to reduce token usage, the first four functions would only
        use the current query, and the last function would use the history.
        
        For non-open questions, we need to create a "fake query", otherwise
        the few-shot learning examples could show up in the follow-up
        questions. For example, if the user types "more examples", it could
        sometimes display the example from the few-shot learning.
        help_info acts as the fake query.
        """
        type, query = self.get_query_type(query)
        help_info = ""
        if type == "explain":
            help_info = f"explain the meaning of {query}"
            query = self.type_explain(query)
        elif type == "exemplify":
            help_info = f"provide an example to {query}"
            query = self.type_exemplify(query, major)
        elif type == "detect":
            help_info = f"detect ambiguious and confusing text for my major in {query}"
            query = self.type_detect(query, major)
        elif type == "simplify":
            help_info = f"simplify {query}"
            query = self.type_simplify(query)
        else:
            help_info = query
        return type, help_info, query
    
    def get_response(self, query, major):
        type, help_info, query = self.preprocess_query(query, major)
        # help_info acts as the query that needs to be added to the history
        # cuz some functions could contain long few-shot examples
        self.history.add_user_message(help_info)
        
        # for non-open questions, we directly prompt the api with the query
        if type != "open":
            user_msg = self.history.form_user_msg(query)
            messages = [self.history.sys_messages, user_msg]
        else: 
            messages = self.history.get_message_for_api()
        #print("debug, messages: ", messages)
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.4,
            max_tokens=1000,
            frequency_penalty=0.0,
            stream=True,
        )
        return response
    
    def collect_response(self, response:str):
        self.history.add_assistant_message(response)