import openai, prompts, consts, os, json, re
from tools import serp_api
from colorama import Fore
from collections import deque
from common_utils import count_tokens, split_answer_and_cot, get_oneshots, openai_call
from utils import pinecone_utils, text_processing
import consts
import traceback
import json
import sys
from ast import literal_eval


openai.api_key = consts.OPENAI_API_KEY

one_shots, p_one_shots = get_oneshots()
all_one_shots = one_shots+p_one_shots


class AutonomousAgent:
    def __init__(self, objective):
        (
            self.objective,
            self.working_memory,
            self.chore_prompt,
            self.completed_tasks,
            self.task_id_counter,
            self.openai_call,
            self.task_list,
            self.indexes,
            self.focus,
            self.get_serp_query_result,
            self.get_erp_api_result,
            self.current_task
        ) = (objective, [], prompts.chore_prompt, [], 1, openai_call, deque([]), {},
             "", serp_api.get_serp_query_result, serp_api.get_erp_api_result, "")
    

    def get_current_state(self):
        # filter properties to avoid adiction
        hash = {"self": [nome for nome in dir(self) if not nome.startswith("__") and nome not in "search_in_index get_ada_embedding append_to_index memory_agent repl_agent task_list memory focus indexes"],
                "To-do tasks list": self.task_list,
                "Available indexes": [ind for ind in self.indexes.keys()],
                "self.working_memory": self.working_memory,
                "self.focus": self.focus,
                "current dir": os.listdir(os.getcwd())}
        return hash

    def execution_agent(self, current_task, root=False):
        self.current_task = current_task

        if not root:
            print(Fore.LIGHTRED_EX + "\nExecution Agent call with task:" + Fore.RESET + f"{current_task}")

        if not current_task in [o['task'] for o in one_shots]:
            one_shots_names_and_kw = [f"name: '{one_shot['task']}', task_id: '{one_shot['memory_id']}', keywords: '{one_shot['keywords']}';\n\n" for one_shot in all_one_shots]
            code, cot = split_answer_and_cot(openai_call(
                f"My current task is: {current_task}."
                f"I must choose from 0 to {consts.N_SHOT} most relevant tasks between the following one_shot examples:'\n{one_shots_names_and_kw}'.\n\n"
                f"These oneshots will be injected in execution_agent as instant memories, task memory. I will try to choose {consts.N_SHOT} tasks memories that may help ExA. I will tell the relevant tasks by looking the names and keywords, and imagining what abilities ExA used to produce this memory."
                f"I must write a list({consts.N_SHOT}) cointaining only the memory_ids of the most relevant one_shots, or a empty list. i.e '[\"one_shot example memory_id\"]' or '[]'."
                f"I must read the examples' names and choose from 0 to {consts.N_SHOT} by memory_id. "
                f"""I must answer in the format '\{{"chain of thoughts": "here I put a short reasoning", "answer": ['most relevant memory_id']'}};"""
                f"My answer:", max_tokens=300).strip("'"))
            print(cot)
            pattern = r'\[([^\]]+)\]'
            matches = re.findall(pattern, str(code))
            completion = eval("["+matches[0]+"]") if matches else []
            print(f"\nChosen one-shot example: {completion}\n")
            one_shot_example_names = completion[:consts.N_SHOT] if len(completion) > 0 else None

            prompt = prompts.execution_agent(
                    self.objective,
                    self.completed_tasks,
                    self.get_current_state,
                    current_task,
                    [one_shot for one_shot in all_one_shots if one_shot["memory_id"] in one_shot_example_names] if one_shot_example_names is not None else '',
                    self.task_list, 
                    None

                )
            #print(Fore.LIGHTCYAN_EX + prompt + Fore.RESET)
            changes = openai_call(
                prompt,
                .5,
                4000-self.count_tokens(prompt),
            )


            validate = (literal_eval(changes))["answer"]
            prompt_validate = prompts.validate_agent(None ,current_task, validate)
            reformulated = openai_call(
                prompt_validate,
                .5,
                4000-self.count_tokens(prompt),
            )
            data_reformulated = literal_eval(reformulated)
            if consts.VIEWER:
             print(Fore.LIGHTCYAN_EX + reformulated + Fore.RESET)
            while data_reformulated["status"] != "success":
                prompt = prompts.execution_agent(
                    self.objective,
                    self.completed_tasks,
                    self.get_current_state,
                    current_task,
                    [one_shot for one_shot in all_one_shots if one_shot["memory_id"] in one_shot_example_names] if one_shot_example_names is not None else '',
                    self.task_list, 
                    data_reformulated["report"]

                )
                changes = openai_call(
                    prompt,
                    .5,
                    4000-self.count_tokens(prompt),
                )
                validate = (literal_eval(changes))["answer"]
                prompt_validate = prompts.validate_agent(data_reformulated ,current_task, validate)
                reformulated = openai_call(
                    prompt_validate,
                    .5,
                    4000-self.count_tokens(prompt),
                )
                if reformulated == str(data_reformulated):
                    break
                data_reformulated = literal_eval(reformulated)
                if consts.VIEWER:
                 print(Fore.LIGHTCYAN_EX + reformulated + Fore.RESET)
            
            

            thoughts = literal_eval(changes)["chain of thoughts"]
            print(Fore.LIGHTMAGENTA_EX+f"\n\ncodename ExecutionAgent:"+Fore.RESET+f"\n\n{thoughts}")

            if consts.VIEWER:
                print(f"Command: {changes}")

            # try until complete
            result, code, cot = self.repl_agent(current_task, changes)

            save_task = True
            if consts.USER_IN_THE_LOOP:
                while True:
                    inp = str(input('\nDo you want to save this action in memory? (Y/N)\n>')).lower()
                    if inp in 'y yes n no':
                        if inp[0] == 'n':
                            save_task = False
                        break
            

            if save_task:
                one_shots.append(
                    {
                        "memory_id": "os-{0:09d}".format(len(one_shots)+1),
                        "objective": self.objective,
                        "task": current_task,
                        "thoughts": cot,
                        "code": code,
                        "keywords": ', '.join(eval(openai_call("I must analyze the following task name and action and write a list of keywords.\n"
                                    f"Task name: {current_task};\nAction: {code};\n\n"
                                    f"> I must write a python list cointaing strings, each string one relevant keyword that will be used by ExecutionAgent to retrieve this memories when needed."
                                                     f" i.e: ['search', 'using pyautogui', 'using execution_agent', 'how to x', 'do y']\n"
                                    f"My answer:", max_tokens=2000)))
                    }
                )
                with open("src/memories/one-shots.json", 'w') as f:
                    f.write(json.dumps(one_shots, indent=True, ensure_ascii=False))

        else:
            cot, code = [[o['thoughts'], o['code']] for o in one_shots if o['task'] == current_task][0]
            print(Fore.LIGHTMAGENTA_EX + f"\n\ncodename ExecutionAgent:" + Fore.RESET + f"\nChain of thoughts:")
            print(*cot, sep='\n')
            print(f"\n\nAnswer:\n{code}")

            #action_func = exec(code, self.__dict__)
            #result = self.action(self)
            result = self.execute_action(code)    
            print(result) 

        self.completed_tasks.append(current_task)
        summarizer_prompt = f"I must summarize the 'working memory' and the last events, I must answer as a chain of thoughts, in first person, in the same verb tense of the 'event'. Working memory: {self.working_memory}, event: {cot} result: {result}. " \
                            f"My answer must include the past workig memory and the new events and thoughts. If there's some error or fix in the event I must summarize it as a learning:"
        self.working_memory = openai_call(summarizer_prompt)

        return result
    
    def execute_action(self, code):
        action = getattr(self, code["command"])
        return action(**code["args"])


    def repl_agent(self, current_task, changes):
        code, cot = split_answer_and_cot(changes)
        ct = 1

        reasoning = changes
        while True:
            try:
                result = self.execute_action(code)
                return result, code, cot
            except Exception as e:
                if consts.DEBUG:
                    print(Fore.RED + f"\n\nERROR: {e}\n" + Fore.RESET)
                    print(code)
                    print('\n\n')
                    traceback.print_tb(e.__traceback__)
                    if input('WANNA DEBUG THE CODE? Y/n: ') in ('YysS'):
                        import pdb
                        pdb.set_trace()
                    if input('WANNA CONTINUE? y/N: ') not in ('YysS'):
                        sys.exit(1)

                else:

                    print(Fore.RED + f"\n\nFIXING AN ERROR: {e.__class__.__name__} {e}\n" + Fore.RESET)
                    print(f"{ct} try")

                    prompt = prompts.fix_agent(current_task, code, cot, e)
                    new_code = openai_call(
                        prompt,
                        temperature=0.4,
                    )
                    reasoning += new_code
                    reasoning = openai_call(f"I must summarize this past events as a chain of thoughts, in first person: {reasoning}", max_tokens=1000)
                    # print(new_code, end="\n")
                    try:
                        code, cot = split_answer_and_cot(new_code)
                        action_func = exec(code, self.__dict__)
                        result = self.action(self)
                        return result, code, cot
                    except Exception as e:
                        pass
            ct += 1

    def change_propagation_agent(self, _changes):
        return openai_call(
            prompts.change_propagation_agent(
                self.objective, _changes, self.get_current_state
            ),
            0.7,
            1000,
        )

    def memory_agent(self, caller,  content, goal):
        answer = openai_call(
            prompts.memory_agent(self.objective, caller, content, goal, self.get_current_state)
        )
        answer = answer[answer.lower().index("answer:")+7:]
        action_func = exec(answer.replace("```", ""), self.__dict__)
        result = self.action(self)

    def search_in_index(self, index_name, query, top_k=1000):
        pinecone_utils.search_in_index(self, index_name, query, top_k=1000)

    def get_ada_embedding(self, text):
        pinecone_utils.get_ada_embedding(text)

    def append_to_index(self, content, index_name):
        pinecone_utils.append_to_index(self, content, index_name)

    def count_tokens(self, text):
        return count_tokens(text)

    def process_large_text(self, text, instruction, max_output_length=1000, split_text=None):
        return text_processing.process_large_text(text, instruction, max_output_length=1000, split_text=None)

    def generate_large_text(self, instruction, max_tokens_lenghts=10000):
        return text_processing.generate_large_text(instruction, max_tokens_lenghts=10000)
    
    def erpnext_get_records_list(
        self,
        doctype,
        fields=None,
        filters=None,
        order_by=None,
        limit_start=None,
        limit_page_length=20,
        #parent=None,
        #debug=False,
        #as_dict=True,
        #or_filters=None
        ):
        """Returns a list of records by filters, fields, ordering and limit

        :param doctype: DocType of the data to be queried
        :param fields: fields to be returned. Default is `name`
        :param filters: filter list by this dict
        :param order_by: Order by this fieldname
        :param limit_start: Start at this index
        :param limit_page_length: Number of records to be returned (default 20)"""


       
        return self.get_erp_api_result(
            'get_list',
            doctype=doctype,
            fields=fields,
            filters=filters,
            order_by=order_by,
            limit_start=limit_start, 
            limit_page_length=limit_page_length
        )
    
    
    def erpnext_get_records_count(self, doctype, filters=None):
        """Returns the number of records that matches a filter
        
        :param doctype: DocType of the data to be queried.
        :param filters: filter list by this dict
        """

        return len(self.erpnext_get_records_list(doctype, ['name'], filters))
    
    def erpnext_get_record_exists(self, doctype, filters=None):
        """Return the boolean that indicates if a record exists or not
        
        :param doctype: DocType of the data to be queried.
        :para filters: filter list by this dict
        """

        return bool(self.erpnext_get_records_list(doctype, ['name'], filters))
    
    def erpnext_get_record(self, doctype, name='', filters=None, parent=None):
        """Returns a document by name or filters

        :param doctype: DocType of the document to be returned
        :param name: return document of this `name`
        :param filters: If name is not set, filter by these values and return the first match"""

        return self.get_erp_api_result(
            'get_doc',
            name=name,
            doctype=doctype,
            filters=filters,
            #parent=parent
        )
    
    def erpnext_get_field_value(self, doctype, fieldname, filters=None, as_dict=True, debug=False, parent=None):
        """Returns a field value form a document

        :param doctype: DocType to be queried
        :param fieldname: Field to be returned (default `name`)
        :param filters: dict or string for identifying the record"""

        return self.get_erp_api_result(
            'get_value',
            doctype=doctype,
            fieldname=fieldname,
            filters=filters,
            #as_dict=as_dict,
            #debug=debug,
            #parent=parent
        )
    
    """
    def erpnext_get_field_single_value(self, doctype, field):
        ""Returns a single field value form a document""

        return self.get_erp_api_result(
            'get_single_value',
            doctype=doctype,
            field=field
        )
    """

    def erpnext_set_field_value(self, doctype, docname, fieldname, value=None):
        """Set a field value using get_doc, group of values

        :param doctype: DocType of the document
	    :param name: name of the document
	    :param fieldname: fieldname string or JSON / dict with key value pair
	    :param value: value if fieldname is JSON / dict"""

        return self.get_erp_api_result(
            'set_value',
            doctype=doctype,
            docname=docname,
            fieldname=fieldname,
            value=value
        )
    
    def erpnext_insert_doc(self, doc=None):
        """Insert a document
        
        param doc: JSON or dict object to be inserted"""

        return self.get_erp_api_result(
            'insert',
            doc=doc
        )
    
    def erpnext_insert_many_docs(self, docs=None):
        """Insert multiple documents

	    :param docs: JSON or list of dict objects to be inserted in one request"""

        return self.get_erp_api_result(
            'insert_many',
            docs=docs
        )

    def erpnext_update_doc(self, doc):
        """Update (save) an existing document

	    :param doc: JSON or dict object with the properties of the document to be updated"""

        return self.get_erp_api_result(
            'update',
            doc=doc
        )
    
    def erpnext_rename_doc(self, doctype, old_name, new_name, merge=False):
        """Rename document

        :param doctype: DocType of the document to be renamed
	    :param old_name: Current `name` of the document to be renamed
	    :param new_name: New `name` to be set"""

        return self.get_erp_api_result(
            'rename_doc',
            doctype=doctype,
            old_name=old_name,
            new_name=new_name,
            merge=merge
        )


    def erpnext_cancel_doc(self, doctype, name):
        """Cancel a document
	    :param doctype: DocType of the document to be cancelled
	    :param name: name of the document to be cancelled"""

        return self.get_erp_api_result(
            'cancel',
            doctype=doctype,
            name=name
        )
    

    def erpnext_delete_doc(self, doctype, name):
        """Delete a remote document

        :param doctype: DocType of the document to be deleted
        :param name: name of the document to be deleted"""       

        return self.get_erp_api_result(
            'delete',
            doctype=doctype,
            name=name
        )
    
    def erpnext_bulk_update_doc(self, docs):
        """Bulk update documents

	    :param docs: JSON list of documents to be updated remotely. Each document must have `docname` property"""

        return self.get_erp_api_result(
            'bulk_update',
            docs=docs
        )
    
    def erpnext_has_permission_doc(self, doctype, docname, perm_type="read"):
        """Returns a JSON with data whether the document has the requested permission
        
        :param doctype: DocType of the document to be checked
	    :param docname: `name` of the document to be checked
	    :param perm_type: one of `read`, `write`, `create`, `submit`, `cancel`, `report`. Default is `read`"""

        return self.get_erp_api_result(
            'has_permission',
            doctype=doctype,
            docname=docname,
            perm_type=perm_type
        )
    
    def erpnext_get_doc_permissions(self, doctype, docname):
        """Returns an evaluated document permissions dict like `{"read":1, "write":1}`

	    :param doctype: DocType of the document to be evaluated
	    :param docname: `name` of the document to be evaluated"""

        return self.get_erp_api_result(
            'get_doc_permissions',
            doctype=doctype,
            docname=docname
        )
    
    def erpnext_get_password_property(self, doctype, name, fieldname):
        """Return a password type property. Only applicable for System Managers

	    :param doctype: DocType of the document that holds the password
	    :param name: `name` of the document that holds the password
	    :param fieldname: `fieldname` of the password property"""

        return self.get_erp_api_result(
            'get_password',
            doctype=doctype,
            name=name,
            fieldname=fieldname
        )
    
    def erpnext_get_js_code(self, items):
        """Return a password type property. Only applicable for System Managers

	    :param doctype: DocType of the document that holds the password
	    :param name: `name` of the document that holds the password
	    :param fieldname: `fieldname` of the password property"""

        return self.get_erp_api_result(
            'get_js',
            items=items
        )
    
    def erpnext_get_default_time_zone(self):
        """Return default time zone"""

        return self.get_erp_api_result(
            'get_time_zone'
        )

    def erpnext_attach_file_to_document(
        self,
        filename=None,
        filedata=None,
        doctype=None,
        docname=None,
        folder=None,
        decode_base64=False,
        is_private=None,
        docfield=None):

        """Attach a file to Document

	    :param filename: filename e.g. test-file.txt
	    :param filedata: base64 encode filedata which must be urlencoded
	    :param doctype: Reference DocType to attach file to
	    :param docname: Reference DocName to attach file to
	    :param folder: Folder to add File into
	    :param decode_base64: decode filedata from base64 encode, default is False
    	:param is_private: Attach file as private file (1 or 0)
	    :param docfield: file to attach to (optional)"""

        return self.get_erp_api_result(
            'attach_file',
            filename=filename,
            filedata=filedata,
            doctype=doctype,
            docname=docname,
            folder=folder,
            decode_base64=decode_base64,
            is_private=is_private,
            docfield=docfield
        )
    
    def erpnext_verify_document_amended(self, doctype, docname):
        """Checks if the document has been changed"""

        return self.get_erp_api_result(
            'is_document_amended',
            doctype=doctype,
            docname=docname
        )
    
    def erpnext_validate_link(self, doctype: str, docname: str, fields=None):
        """Check if the link is valid"""

        return self.get_erp_api_result(
            'validate_link',
            doctype=doctype,
            docname=docname,
            fields=fields
        )
    
    def erpnext_insert_doc_return_object(self, doc):
        """Inserts document and returns parent document object with appended child document
	    if `doc` is child document else returns the inserted document object

        :param doc: doc to insert (dict)"""

        return self.get_erp_api_result(
            'insert_doc',
            doc=doc
        )
    
    def erpnext_delete_doc_child_table(self, doctype, name):
        """Deletes document
	    if doctype is a child table, then deletes the child record using the parent doc
	    so that the parent doc's `on_update` is called"""

        return self.get_erp_api_result(
            'delete_doc',
            doctype=doctype,
            name=name
        )

    
    
