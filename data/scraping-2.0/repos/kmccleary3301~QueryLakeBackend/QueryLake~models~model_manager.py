from .langchain_sse import CustomStreamHandler, ThreadedGenerator, raiseErrorInGenerator
# from . import Exllama

from langchain.callbacks.manager import CallbackManager
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from datetime import datetime

import threading
import copy

from sqlmodel import Session, select
from ..database.sql_db_tables import model_query_raw, access_token, chat_session_new, model, chat_entry_model_response, chat_entry_user_question, model
import time
import json
from langchain.llms import LlamaCpp
from .exllama_langchain import Exllama
from .exllamav2_langchain import ExllamaV2
from ..api.user_auth import get_user, get_openai_api_key
from openai import OpenAI
import tiktoken
from ..instruction_templates import latex_basic_system_instruction

def num_tokens_from_string(string: str, model : str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

default_context_wrapper = """Use and cite the following pieces of context to answer requests with accurate information.
Do not cite anything that is not provided here. Do not make up a source, link a source, or write out the identifier of a source.
<context>
{{context}}
</context>"""

default_sys_instr_pad = ""

def add_context(system_instructions, context_segments, context_wrapper=None):
    if len(context_segments) == 0:
        return system_instructions
    if context_wrapper is None:
        context_wrapper = default_context_wrapper
    context_list = ""
    for i, segment in enumerate(context_segments):
        context_list += "%d. %s\n\n" % (i, segment["document"])
    context_wrapper = context_wrapper.replace("{{context}}", context_list)

    return system_instructions + "\n" + context_wrapper

def construct_params(database: Session, model_id):
    if type(model_id) is str:
        model_entry_db = database.exec(select(model).where(model.name == model_id)).first()
        sys_instr_pad = model_entry_db.system_instruction_wrapper
        usr_entry_pad = model_entry_db.user_question_wrapper
        bot_response_pad = model_entry_db.bot_response_wrapper
        context_pad = model_entry_db.context_wrapper
        return sys_instr_pad, usr_entry_pad, bot_response_pad, context_pad
    else:
        sys_instr_pad = "<<INSTRUCTIONS>>\n{system_instruction}\n<</INSTRUCTIONS>>\n\n"
        usr_entry_pad = "<<USER_QUESTION>>\n{question}\n<</USER_QUESTION>>\n\n<<ASSISTANT>>\n"
        bot_response_pad = "{response}\n<</ASSISTANT>>\n\n"
        context_pad = default_context_wrapper
        return sys_instr_pad, usr_entry_pad, bot_response_pad, context_pad

def construct_chat_history(max_tokens : int, 
                           token_counter, 
                           sys_instr_pad : str, 
                           usr_entry_pad : str, 
                           bot_response_pad : str, 
                           chat_history: list,
                           system_instruction : str,
                           new_question : str,
                           minimum_context_room : int = 1000):
    """
    Construct model input, trimming from beginning until minimum context room is allowed.
    chat history should be ordered from oldest to newest, and entries in the input list
    should be pairs of the form (user_question, model_response).
    """
    system_instruction_prompt = sys_instr_pad.replace("{system_instruction}", system_instruction)
    sys_token_count = token_counter(system_instruction_prompt)

    new_question_formatted = usr_entry_pad.replace("{question}", new_question)

    chat_history_new, token_counts = [], []
    for entry in chat_history:
        new_entry = usr_entry_pad.replace("{question}", entry[1])+bot_response_pad.replace("{response}", entry[1])
        token_counts.append(token_counter(new_entry))
        chat_history_new.append(new_entry)
        # print("%40s %d" % (new_entry[:40], token_counts[-1]))
    # print(max_tokens)
    token_count_total = sys_token_count + token_counter(new_question_formatted)
    construct_prompt_array = []
    token_counts = token_counts[::-1]
    for i, entry in enumerate(chat_history_new[::-1]):
        token_count_tmp = token_counts[i]
        if (token_count_total+token_count_tmp) > (max_tokens - minimum_context_room):
            break
        token_count_total += token_counts[i]
        construct_prompt_array.append(entry)
    
    final_result = system_instruction_prompt + "".join(construct_prompt_array[::-1]) + new_question_formatted

    # print("FINAL PROMPT")
    # print(final_result)
    return final_result



class LLMEnsemble:
    def __init__(self, database: Session, default_llm : str, global_settings : dict) -> None:
        self.max_instances = 1
        self.llm_instances = []
        self.default_llm = default_llm
        self.global_settings = global_settings
        self.make_new_instance(database, default_llm)

    def make_new_instance(self, database, model_name):
        new_model = {
            "lock": False,
            "handler": CustomStreamHandler(None),
        }

        model_db_entry = database.exec(select(model).where(model.name == model_name)).first()

        parameters = json.loads(model_db_entry.default_settings)

        parameters["callback_manager"] = CallbackManager([new_model["handler"]])

        if (model_db_entry.necessary_loader == "exllama"):
            new_model["model"] = Exllama(**parameters)
        elif (model_db_entry.necessary_loader == "exllama_v2"):
            new_model["model"] = ExllamaV2(**parameters)
        elif (model_db_entry.necessary_loader == "llama_cpp"):
            new_model["model"] = LlamaCpp(**parameters)
        
        self.llm_instances.append(new_model)

    def delete_model(self, model_index : int) -> None:
        """
        Properly deletes a model and clears the memory.
        """
        if str(type(self.llm_instances[model_index]["model"].client)) == "ExLlama":
             self.llm_instances[model_index]["model"].client.free_unmanaged()
             del self.llm_instances[model_index]["model"]
             del self.llm_instances[model_index]

    def choose_llm_for_request(self, model_name : str = None):
        """
        This class is structured to cycle multiple instances of LLMs
        to handle heavy server load. This method is designed to select the most available
        llm instance from the ensemble.
        """
        print("Requested LLM:", model_name)
        if model_name is None:
            model_name = self.default_llm

        return 0
    
    def validate_parameters(self, parameters : dict) -> bool:
        """
        Should return true or false if new model parameters are valid.
        NTF
        """
        return True

    def chain(self, 
              username : str,
              password_prehash : str,
              database: Session,
              history : list,
              parameters : dict = None,
              model_choice : str = None,
              context=None,
              session_hash : str = None,
              organization_hash_id : str = None,
              provided_generator : ThreadedGenerator = None):
        """
        This function is for a model request. It creates a threaded generator, 
        substitutes in into the models callback manager, starts the function
        llm_thread in a thread, then returns the threaded generator.
        """

        if model_choice is None:
            model_choice = self.default_llm
        print("Calling Model")

        if provided_generator is None:
            provided_generator = ThreadedGenerator()
        
        if not (parameters is None or parameters == {}):
            if not self.validate_parameters(parameters):
                return None
        print(type(model_choice), str(type(model_choice)), model_choice)
        if type(model_choice) is str or model_choice is None:
            model_index = self.choose_llm_for_request(model_name=model_choice)
            token_counter = self.llm_instances[model_index]["model"].get_num_tokens
            kwargs = {"model_index": model_index}
            self.llm_instances[model_index]["handler"].gen = provided_generator
            model_entry_db = database.exec(select(model).where(model.name == model_choice)).first()
            if model_entry_db is None:
                model_entry_db = database.exec(select(model).where(model.name == self.default_llm)).first()
            system_instruction_base = model_entry_db.default_system_instruction
            if model_entry_db.necessary_loader == "llama_cpp":
                max_tokens = self.llm_instances[model_index]["model"].max_tokens
            else:
                max_tokens = self.llm_instances[model_index]["model"].config.max_seq_len
        else:
            print("External model used:")
            print(model_choice)
            assert model_choice[0] in self.global_settings["external_model_providers"], "Invalid External Provider"
            assert model_choice[1] in self.global_settings["external_model_providers"][model_choice[0]], "Invalid External Model"
            
            def token_counter_tiktoken(prompt : str):
                return num_tokens_from_string(prompt, model_choice[-1])
            token_counter = token_counter_tiktoken
            kwargs = get_openai_api_key(database, username, password_prehash, organization_hash_id=organization_hash_id)
            if "organization_id" in kwargs:
                kwargs["organization"] = kwargs["organization_id"]
            system_instruction_base = latex_basic_system_instruction
            max_tokens = self.global_settings["external_model_providers"][model_choice[0]][model_choice[1]]

        if history[0]["role"] == "system":
            system_instruction_base = history[0]["content"]
        else:
            history = [{"role": "system", "content": system_instruction_base}] + history
        

        sys_instr_pad, usr_entry_pad, bot_response_pad, context_pad = construct_params(database, model_choice)
        if not context is None and len(context) > 0:
            system_instruction_base = add_context(system_instruction_base, context, context_wrapper=context_pad)
        
        
        

        # system_instruction_prompt = model_entry_db.system_instruction_wrapper.replace("{system_instruction}", system_instruction_base)
        
        # bot_responses_previous = database.exec(select(chat_entry_model_response).where(chat_entry_model_response.chat_session_id == session.id)).all()
        # bot_responses_previous = sorted(bot_responses_previous, key=lambda x: x.timestamp)

        chat_history_raw = []
        
        for i in range(1, len(history)-1, 2):
            chat_history_raw.append([history[i]["content"], history[i+1]["content"]])

        # for bot_response in bot_responses_previous:
        #     question_previous = database.exec(select(chat_entry_user_question).where(chat_entry_user_question.id == bot_response.chat_entry_response_to)).first()
        #     chat_history_raw.append([question_previous.content, bot_response.content])
        
        
        if not session_hash is None:
            session = database.exec(select(chat_session_new).where(chat_session_new.hash_id == session_hash)).first()
            if len(chat_history_raw) == 0:
                session.title = history[len(history)-1]["content"].split(r"[.|?|!|\n|\t]")[-1]
        else:
            session = None

        

        
        if type(model_choice) is str: # Use a local model
            prompt_medium = construct_chat_history(
                max_tokens=max_tokens,
                token_counter=token_counter,
                sys_instr_pad=sys_instr_pad,
                usr_entry_pad=usr_entry_pad,
                bot_response_pad=bot_response_pad,
                chat_history=chat_history_raw,
                system_instruction=system_instruction_base,
                new_question=history[len(history)-1]["content"]
            )
            threading.Thread(target=self.llm_thread, args=(provided_generator, 
                                                           username,
                                                           database, 
                                                           prompt_medium,
                                                           history[len(history)-1]["content"],
                                                           parameters, 
                                                           context,
                                                           model_index), kwargs={"session": session}).start()
        else: # Use an OpenAI Model
            # all_history = []
            # all_history.append({"role": "system", "content": system_instruction_base})
            # for entry in chat_history_raw:
            #     all_history.append({"role": "user", "content": entry[0]})
            #     all_history.append({"role": "assistant", "content": entry[1]})
            # all_history.append({"role": "user", "content": history[len(history)-1]})
            threading.Thread(target=self.llm_thread_openai, args=(database,
                                                                  kwargs,
                                                                  history,
                                                                  history[len(history)-1]["content"],
                                                                  context,
                                                                  model_choice[1]), kwargs={"g": provided_generator, "session": session}).start()
        return provided_generator


    def llm_thread_openai(self, 
                          database : Session,
                          openai_keys,
                          chat_history : str,
                          single_question : str,
                          context,
                          model_choice : str,
                          session : chat_session_new = None,
                          g : ThreadedGenerator = None):
        """
        Call OpenAI Model with streaming in a thread.
        """
        print("Calling openAI model")
        if not session is None:
            question_entry = chat_entry_user_question(
                chat_session_id=session.id,
                timestamp=time.time(),
                content=single_question,
            )
            database.add(question_entry)
            database.commit()
            database.flush()
        print("Openai args:", openai_keys)
        start_time = time.time()
        client = OpenAI(**openai_keys)
        response = ""
        for chunk in client.chat.completions.create(
            model=model_choice,
            messages=chat_history,
            stream=True
        ):
            content = chunk.choices[0].delta.content
            if not content is None:
                if not g is None:
                    g.send(content)
                response += content
        end_time = time.time()
        if not g is None:
            # g.send("-DONE-")
            g.close()

        response_token_count = num_tokens_from_string(response, model_choice)
        request_data = {
            "prompt": json.dumps(chat_history),
            "response" : response,
            "response_size_tokens": response_token_count,
            "prompt_size_tokens": sum([num_tokens_from_string(msg["content"], model_choice) for msg in chat_history]) ,
            "model": model_choice,
            "timestamp": time.time(),
            "time_taken": end_time-start_time,
            "model_settings": "{}"
        }
        print("OpenAI response %.2f tokens/s" % (response_token_count/(end_time-start_time)))
        new_request = model_query_raw(**request_data)
        database.add(new_request)
        database.commit()

        if not session is None:
            model_response = chat_entry_model_response(
                chat_session_id=session.id,
                timestamp=time.time(),
                content=response,
                chat_entry_response_to=question_entry.id,
                sources=json.dumps(context) if not context is None and len(context) > 0 else None
                #model query raw id.
            )
            database.add(model_response)
            database.commit()
        return response

    def llm_thread(self, 
                   g, 
                   username,
                   database: Session,
                   full_query : str,
                   single_question : str,
                   parameters,
                   context,
                   model_index : int,
                   session : chat_session_new = None):
        """
        This function is run in a thread, outside of normal execution.
        """
        first_of_session = False
        while self.llm_instances[model_index]["lock"] == True:
            time.sleep(0.005)
        try:
            previous_values = {}
            # if not (parameters is None or parameters == {}):
            #     for key, _ in parameters.items():
            #         previous_values[key] = self.llm_instances[model_index]["model"].__dict__[key]
            #     self.llm_instances[model_index]["model"].refresh_params(parameters)
            
            

            start_time = time.time()
            self.llm_instances[model_index]["lock"] = True
            prompt_template = PromptTemplate(input_variables=["question"], template="{question}")
            # final_prompt = prompt_template.format(question=prompt)
            llm_chain = LLMChain(prompt=prompt_template, llm=self.llm_instances[model_index]["model"])
            # llm_chain.run({"question": prompt, "system_instruction": system_instruction})

            # print("final_prompt")
            # print(prompt_medium)
            
            try:
                response = llm_chain.run(full_query)
                end_time = time.time()

                # print(response)

                tokens_add = self.llm_instances[model_index]["handler"].tokens_generated
                first_key = database.exec(select(access_token).where(access_token.author_user_name == username)).first()

                first_key.tokens_used += tokens_add
                database.commit()
                request_data = {
                    "prompt": full_query,
                    "response" : response,
                    "response_size_tokens": self.llm_instances[model_index]["handler"].tokens_generated,
                    "prompt_size_tokens": self.llm_instances[model_index]["model"].get_num_tokens(full_query),
                    "model": self.llm_instances[model_index]["model"].model_path.split("/")[-1],
                    "timestamp": time.time(),
                    "time_taken": end_time-start_time,
                    "model_settings": str(self.llm_instances[model_index]["model"].__dict__),
                    # "author": user_name,
                    "access_token_id": first_key.id
                }
                new_request = model_query_raw(**request_data)
                database.add(new_request)
                database.commit()

                
                # new_user = sql_db.User(**user_data)
                if not session is None:
                    question_entry = chat_entry_user_question(
                        chat_session_id=session.id,
                        timestamp=time.time(),
                        content=single_question,
                    )
                    database.add(question_entry)
                    database.commit()
                    database.flush()

                    model_response = chat_entry_model_response(
                        chat_session_id=session.id,
                        timestamp=time.time(),
                        content=response,
                        chat_entry_response_to=question_entry.id,
                        sources=json.dumps(context) if not context is None and len(context) > 0 else None
                        #model query raw id.
                    )
                    database.add(model_response)
                    database.commit()
            except AssertionError as e:
                g.send("<<<FAILURE>>> | Model Context Length Exceeded")
        finally:
            self.llm_instances[model_index]["handler"].tokens_generated = 0
            self.llm_instances[model_index]["model"].callback_manager = None
            g.close()
            self.llm_instances[model_index]["handler"].gen = None
            self.llm_instances[model_index]["lock"] = False
            # if previous_values != {}: # Reset the model parameters to normal if they were changed.
            #     # self.llm_instances[model_index]["model"].__dict__.update(previous_values)
            #     self.llm_instances[model_index]["model"].refresh_params(previous_values)
            del previous_values

    def single_sync_chat(self,
                         database : Session,
                         username : str, 
                         password_prehash : str, 
                         history : str,
                         context : list = None,
                         parameters : dict = None,
                         model_choice : str = None,
                         organization_hash_id : str = None):
        """
        This is called in a thread synchronously for execution.
        """
        user = get_user(database, username, password_prehash)

        if model_choice is None:
            model_choice = self.default_llm
        print("Calling Model")
        
        if not (parameters is None or parameters == {}):
            if not self.validate_parameters(parameters):
                return None
        print(type(model_choice), str(type(model_choice)), model_choice)
        if type(model_choice) is str or model_choice is None:
            model_index = self.choose_llm_for_request(model_name=model_choice)
            token_counter = self.llm_instances[model_index]["model"].get_num_tokens
            kwargs = {"model_index": model_index}
            model_entry_db = database.exec(select(model).where(model.name == model_choice)).first()
            if model_entry_db is None:
                model_entry_db = database.exec(select(model).where(model.name == self.default_llm)).first()
            system_instruction_base = model_entry_db.default_system_instruction
            if model_entry_db.necessary_loader == "llama_cpp":
                max_tokens = self.llm_instances[model_index]["model"].max_tokens
            else:
                max_tokens = self.llm_instances[model_index]["model"].config.max_seq_len
        else:
            print("External model used:")
            print(model_choice)
            assert model_choice[0] in self.global_settings["external_model_providers"], "Invalid External Provider"
            assert model_choice[1] in self.global_settings["external_model_providers"][model_choice[0]], "Invalid External Model"
            
            def token_counter_tiktoken(prompt : str):
                return num_tokens_from_string(prompt, model_choice[-1])
            token_counter = token_counter_tiktoken
            kwargs = get_openai_api_key(database, username, password_prehash, organization_hash_id=organization_hash_id)
            system_instruction_base = latex_basic_system_instruction
            max_tokens = self.global_settings["external_model_providers"][model_choice[0]][model_choice[1]]

        if history[0]["role"] == "system":
            system_instruction_base = history[0]["content"]
        else:
            history = [{"role": "system", "content": system_instruction_base}] + history

        sys_instr_pad, usr_entry_pad, bot_response_pad, context_pad = construct_params(database, model_choice)
        if not context is None and len(context) > 0:
            system_instruction_base = add_context(system_instruction_base, context, context_wrapper=context_pad)
        
        chat_history_raw = []
        
        for i in range(1, len(history)-1, 2):
            chat_history_raw.append([history[i]["content"], history[i+1]["content"]])

        if type(model_choice) is str: # Use a local model
            first_key = database.exec(select(access_token).where(access_token.author_user_name == username)).first()
            prompt_medium = construct_chat_history(
                max_tokens=max_tokens,
                token_counter=token_counter,
                sys_instr_pad=sys_instr_pad,
                usr_entry_pad=usr_entry_pad,
                bot_response_pad=bot_response_pad,
                chat_history=chat_history_raw,
                system_instruction=system_instruction_base,
                new_question=history[len(history)-1]["content"]
            )
            prompt_template = PromptTemplate(input_variables=["question"], template="{question}")
            llm_chain = LLMChain(prompt=prompt_template, llm=self.llm_instances[model_index]["model"])

            start_time = time.time()
            response = llm_chain.run(prompt_medium)
            end_time = time.time()

            request_data = {
                "prompt": prompt_medium,
                "response" : response,
                "response_size_tokens": self.llm_instances[model_index]["handler"].tokens_generated,
                "prompt_size_tokens": self.llm_instances[model_index]["model"].get_num_tokens(prompt_medium),
                "model": self.llm_instances[model_index]["model"].model_path.split("/")[-1],
                "timestamp": time.time(),
                "time_taken": end_time-start_time,
                "model_settings": str(self.llm_instances[model_index]["model"].__dict__),
                # "author": user_name,
                "access_token_id": first_key.id
            }
            new_request = model_query_raw(**request_data)
            database.add(new_request)
            database.commit()
        else:
            response = self.llm_thread_openai(database,
                                              kwargs,
                                              history,
                                              history[len(history)-1]["content"],
                                              context,
                                              model_choice[1])
        return response

        