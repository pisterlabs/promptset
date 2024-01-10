import pandas as pd
import json
import logging
import time
import re
import string
from abc import ABC, abstractmethod
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import promptlayer
from openai import OpenAI
from sqlalchemy import update
from sqlalchemy.dialects.mysql import insert
from timeout_function_decorator import timeout

from prompts.templates import llama2Templates
from src.sqldb import HallucinationDb
from src.responseparser import LLmResponseParser
from prompts import templates
from src.constants import *
from src.ollamaclient import OllamaClient

GPT_MODEL_NAME = "gpt-3.5-turbo"
GPT_3_5_TURBO_EVAL = 2

NO_HALLUCINATION = 0
HALLUCINATION = 1
IRRELEVANT = 2

HUMAN_EVAL = 1
ACCEPTANCE_REFLECTION = 2
DEFINITION_REFLECTION = 3
CODE_CHECK = 4

GPT_MODEL_LIST = ["gpt-3.5-turbo", "gpt-4-1106-preview"]

FUNCTION_EVALUATOR_ID = 10

class QAMetadata:
    def __init__(self, row: pd.Series) -> None:
        self.answer_id = row["answer_id"]
        self.answer = row["answer"]
        self.question_id = row["question_id"]
        self.question = row["question"]
        self.replacement = row["replacement"]
        self.replacement_id = row["replacement_id"]
        self.term_list = [Term(row["secondary_id"], row["secondary"], row["secondary_meaning"], row["secondary_source"], False)]
        if row["replacement_type"]:
            self.term_list.append(Term(row["replacement_id"], row["replacement"], row["replacement_meaning"], row["replacement_source"], False))       
        else:
            self.term_list.append(Term(row["nonexistent_id"], row["nonexistent"], "", 0, True))


class Term:
    def __init__(self, term_id:int, name:str, meaning:str, term_source:int, isHallucinative:bool) -> None:
        self.term_id = term_id
        self.name = name
        self.meaning = meaning
        self.term_source = term_source
        self.isHallucinative = isHallucinative


class AnswerEvaluator():
    def __init__(self, evaluator_model_name, settings):
        self.settings = settings
        self.db = HallucinationDb(settings)

        if evaluator_model_name in GPT_MODEL_LIST:
            self.model = GptEvaluator(evaluator_model_name, self.db, settings)
        else:
            self.model = OpenEvaluator(evaluator_model_name, self.db, settings)
        
        self.answers_table = self.db.GetTableDefinition(self.db.TERMS_ANSWERS_TABLE)
        self.eval_table = self.db.GetTableDefinition(self.db.TERMS_ANSWERS_EVAL_TABLE)
        self.verbose = True
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def evaluate_model(self, model_under_test:str, verbose:bool = False, sampled_size:bool = False, reversed:bool = False, half:bool = False):
        answers_df = self.retrieve_answers(model_under_test, verbose, sampled_size)
        if reversed: answers_df = answers_df.iloc[::-1]        
        if half: answers_df = answers_df.iloc[len(answers_df)//2:]
        self.process_evaluation(answers_df, verbose)

    def evaluate_df(self, answers_df:pd.DataFrame):
        self.logger.info(f"Processing evaluation of {answers_df.shape[0]} questions with {self.model.model_name} model...")
        
        answers_df["evaluations"] = answers_df.apply(lambda row: self.evaluate_row(row), axis=1)
        answers_df = answers_df.explode("evaluations")
        answers_df = answers_df.reset_index(drop=True)
        answers_df[["eval_term", 'eval_label', 'eval_type', 'reflection']] = pd.DataFrame(answers_df['evaluations'].tolist(), index=answers_df.index)
        answers_df = answers_df.drop(columns=["evaluations"])

        return answers_df

    def evaluate_row(self, row:pd.Series):
        self.eval_labels = []
        self.eval_types = []
        self.eval_terms = []
        self.reflections = []

        metadata = QAMetadata(row)
        self.logger.info(f"\n\n***\n\nAnswer-{metadata.answer_id}: Question-{metadata.question_id}: {metadata.question}")
        self.single_evaluate_unconnected(metadata, metadata.term_list[0])
        self.single_evaluate_unconnected(metadata, metadata.term_list[1])
        return [{"eval_term":term, "eval_label": label, "eval_type": type, "reflection": reflection} for term, label, type, reflection in zip(self.eval_terms, self.eval_labels, self.eval_types, self.reflections)]

    def retrieve_answers(self, model_under_test:str, verbose:bool = False, sampled_size:bool = False):
        self.verbose = verbose
        self.logger.info(f"Retrieving answers from {model_under_test} model...")
        model_under_test_id = self.model.GetModelId(model_under_test)
        self.logger.info(f"Model ID: {model_under_test_id}")
        answers_table = self.db.GetTableDefinition(self.db.COMBINED_TERMS_ANSWERS)
        self.logger.info(f"Table: {answers_table}")
        answers_query = answers_table.select(
            ).where(answers_table.c.answer_source_id == model_under_test_id)
        if sampled_size:
            answers_query = answers_query.where(answers_table.c.question_id.in_(sampled_question_ids))
        self.logger.info(f"Query: {answers_query.compile(compile_kwargs={'literal_binds': True})}")
        return pd.read_sql(answers_query, self.db.sql.connection)
    
    def evaluate_term_accepted(self, model_under_test:str, verbose:bool = False, sampled_size:bool = False):
        answers_df = self.retrieve_answers(model_under_test, verbose, sampled_size)
        for index, row in answers_df.iterrows():
            self.metadata = QAMetadata(row)
            for term in self.metadata.term_list:
                self.term = term
                self.logger.info(f"\n---\nQuestion-{self.metadata.question_id}: Answer-{self.metadata.answer_id}: {self.term.name} ({self.term.term_id}) is being evaluated...")
                self.single_evaluate_accepted(self.metadata, term)

    def evaluate_term_meanings(self, model_under_test:str, verbose:bool = False, sampled_size:bool = False):
        answers_df = self.retrieve_answers(model_under_test, verbose, sampled_size)
        for index, row in answers_df.iterrows():
            self.metadata = QAMetadata(row)
            for term in self.metadata.term_list:
                self.term = term
                self.logger.info(f"\n---\nQuestion-{self.metadata.question_id}: Answer-{self.metadata.answer_id}: {self.term.name} ({self.term.term_id}) is being evaluated...")
                self.single_meaning_evaluate(self.metadata, term)
            

    def evaluate_term_usage(self, model_under_test:str, verbose:bool = False, sampled_size:bool = False):
        answers_df = self.retrieve_answers(model_under_test, verbose, sampled_size)
        for index, row in answers_df.iterrows():
            self.metadata = QAMetadata(row)
            for term in self.metadata.term_list:
                self.term = term
                self.logger.info(f"\n---\nQuestion-{self.metadata.question_id}: Answer-{self.metadata.answer_id}: {self.term.name} ({self.term.term_id}) is being evaluated...")
                self.check_term_used()
            raise Exception("stop")

    def revise_from_df(self, answers_df:pd.DataFrame, verbose:bool = True):
        self.verbose = verbose
        self.eval_ids_to_change = []
        self.evals_to_add = []
        self.label_change_count = 0
        self.missing_inclusion_eval_count = 0

        for index, row in answers_df.iterrows():
            self.metadata = QAMetadata(row)
            for term in self.metadata.term_list:
                self.term = term

                inclusion_label = self.inclusion_revise_process()

                if inclusion_label == IRRELEVANT: continue

                self.single_evaluate_accepted()
                if term.term_source > 0: self.single_meaning_evaluate()

    def process_evaluation(self, answers_df:pd.DataFrame, verbose:bool = False):
        self.verbose = verbose
        self.logger.info(f"Processing evaluation with {self.model.model_name} model...")
        self.answers_df = answers_df
        for index, row in answers_df.iterrows():
            metadata = QAMetadata(row)

            self.logger.info(f"\n\n***\n\nAnswer-{metadata.answer_id}: Question-{metadata.question_id}: {metadata.question}")

            self.single_evaluate(metadata, metadata.term_list[0])
            self.single_evaluate(metadata, metadata.term_list[1])

    def single_evaluate(self, metadata:QAMetadata, term:Term):
        self.metadata = metadata
        self.term = term

        self.logger.info(f"\n---\nQuestion-{self.metadata.question_id}: Answer-{self.metadata.answer_id}: {self.term.name} ({self.term.term_id}) is being evaluated...")

        if self.checked_before():
            self.logger.info(f"Answer-{self.metadata.answer_id} : {self.term.name} has been checked before.")
            return
        if not self.check_term_used():
            return
        
        acceptance_response, eval_label = self.model.check_term_accepted(self.metadata, self.term)
        self.insert_eval(eval_label, ACCEPTANCE_REFLECTION, acceptance_response)

        self.logger.info(f"Answer-{self.metadata.answer_id}: Acceptance evaluation label for {self.term.name} is {eval_label}.")

        if not term.isHallucinative:
            meaning_response, meaning_eval = self.model.check_term_meaning(self.metadata, self.term)
            self.insert_eval(meaning_eval, DEFINITION_REFLECTION, meaning_response)
            self.logger.info(f"Answer-{self.metadata.answer_id}: Meaning evaluation label for {self.term.name} is {eval_label}.")

    def single_evaluate_unconnected(self, metadata:QAMetadata, term:Term):
        self.metadata = metadata
        self.term = term
        self.logger.info(f"\n---\nQuestion-{self.metadata.question_id}: Answer-{self.metadata.answer_id}: {self.term.name} ({self.term.term_id}) is being evaluated...")

        term_included = self.check_inclusion(self.metadata.answer, self.term.name)
        term_usage_label = NO_HALLUCINATION if term_included else IRRELEVANT
        self.eval_labels.append(term_usage_label)
        self.eval_types.append(CODE_CHECK)
        self.reflections.append("")
        self.eval_terms.append(self.term.name)

        if not term_included: return

        acceptance_response, eval_label = self.model.check_term_accepted(self.metadata, self.term)
        self.reflections.append(acceptance_response)
        self.eval_labels.append(eval_label)
        self.eval_types.append(ACCEPTANCE_REFLECTION)
        self.eval_terms.append(self.term.name)

        self.logger.info(f"Answer-{self.metadata.answer_id}: Acceptance evaluation label for {self.term.name} is {eval_label}.")

        if not term.isHallucinative:
            meaning_response, meaning_eval = self.model.check_term_meaning(self.metadata, self.term)
            self.reflections.append(meaning_response)
            self.eval_labels.append(meaning_eval)
            self.eval_types.append(DEFINITION_REFLECTION)
            self.eval_terms.append(self.term.name)

            self.logger.info(f"Answer-{self.metadata.answer_id}: Meaning evaluation label for {self.term.name} is {eval_label}.")

    def single_meaning_evaluate(self, metadata:QAMetadata=None, term:Term=None):
        if metadata != None: self.metadata = metadata
        if term !=None: self.term = term

        if self.term.isHallucinative:
            return
        if self.metadata.answer.lower().count(self.term.name.lower()) == 0:
            return
        query = self.eval_table.select().where(self.eval_table.c.answer_id == self.metadata.answer_id).where(self.eval_table.c.term_id == self.term.term_id).where(self.eval_table.c.model_id == self.model.model_id).where(self.eval_table.c.term_source == self.term.term_source).where(self.eval_table.c.eval_type_id == DEFINITION_REFLECTION)
        query_result = self.db.sql.execute(query)
        if query_result.rowcount > 0:
            return

        meaning_response, meaning_eval = self.model.check_term_meaning(self.metadata, self.term)
        self.insert_eval(meaning_eval, DEFINITION_REFLECTION, meaning_response)
        self.logger.info(f"Answer-{self.metadata.answer_id}: Meaning evaluation label for {self.term.name} is {meaning_eval}.")

    def single_evaluate_accepted(self, metadata:QAMetadata=None, term:Term=None):
        if metadata != None: self.metadata = metadata
        if term !=None: self.term = term

        self.logger.info(f"\n---\nQuestion-{self.metadata.question_id}: Answer-{self.metadata.answer_id}: {self.term.name} ({self.term.term_id}) is being evaluated...")

        if self.metadata.answer.lower().count(self.term.name.lower()) == 0:
            return
        query = self.eval_table.select().where(self.eval_table.c.answer_id == self.metadata.answer_id).where(self.eval_table.c.term_id == self.term.term_id).where(self.eval_table.c.model_id == self.model.model_id).where(self.eval_table.c.term_source == self.term.term_source).where(self.eval_table.c.eval_type_id == ACCEPTANCE_REFLECTION)
        query_result = self.db.sql.execute(query)
        if query_result.rowcount > 0:
            return
        
        acceptance_response, eval_label = self.model.check_term_accepted(self.metadata, self.term)
        self.insert_eval(eval_label, ACCEPTANCE_REFLECTION, acceptance_response)
        if self.evals_to_add != None: self.evals_to_add.append((metadata, term))
        self.logger.info(f"Answer-{self.metadata.answer_id}: Acceptance evaluation label for {self.term.name} is {eval_label}.")

    def insert_eval(self, eval_label:int, eval_type_id:int, reflection:str):
        query = self.eval_table.insert().values(
            answer_id = self.metadata.answer_id,
            eval_label = eval_label,
            eval_type_id = eval_type_id,
            reflection = reflection,
            term_source = self.term.term_source,
            term_id = self.term.term_id,
            model_id = self.model.model_id
        )
        self.db.sql.execute(query)

    def check_term_used(self):
        term_included = self.check_inclusion(self.metadata.answer, self.term.name)
        term_usage_label = NO_HALLUCINATION if term_included else IRRELEVANT

        if self.term_usage_checked_before():
            self.logger.info(f"Answer-{self.metadata.answer_id} : {self.term.name} has been usage checked before.")
            return term_included

        term_usage_query = self.eval_table.insert().values(
            answer_id = self.metadata.answer_id,
            eval_label = term_usage_label,
            eval_type_id = CODE_CHECK,
            term_source = self.term.term_source,
            term_id = self.term.term_id,
            model_id = FUNCTION_EVALUATOR_ID
        )
        self.db.sql.execute(term_usage_query)

        self.logger.info(f"Answer-{self.metadata.answer_id}: {self.term.name} is used: {str(term_included)}.")

        return term_included

    def inclusion_revise_process(self):
        inclusion = self.check_inclusion(self.metadata.answer, self.term.name)

        if inclusion: term_usage_label = NO_HALLUCINATION
        else: term_usage_label = IRRELEVANT

        evals_query = self.inclusion_eval_filter(self.eval_table.select())
        
        stored_row = self.db.sql.execute(evals_query).fetchone()
        if stored_row == None:
            self.evals_to_add.append((self.metadata.answer_id, term_usage_label, 4, "", self.term.term_source, self.term.term_id, 10))
            self.logger.info(f"missed evaluation: {self.term.name} for answer {self.metadata.answer_id} consider adding to db")
            #insert_statement = insert(eval_table).values(answer_id=self.metadata.answer_id, eval_label=term_usage_label, eval_type_id=4, reflection="", 
            #                                              term_source=self.term.term_source, term_id=self.term.term_id, model_id=10)
            #db.sql.execute(insert_statement)
            return term_usage_label

        stored_eval = stored_row.eval_label

        if term_usage_label != stored_eval:
            self.eval_ids_to_change.append(stored_row.id)
            update_statement = self.inclusion_eval_filter(update(self.eval_table)).values(eval_label = term_usage_label)
            self.db.sql.execute(update_statement)
            self.logger.info(f"term: {self.term.name} new label: {term_usage_label} old label: {stored_eval} answer id: {self.metadata.answer_id}")
        return term_usage_label


    def inclusion_eval_filter(self, statement):
        return (statement.where(self.eval_table.c.answer_id == self.metadata.answer_id)
                        .where(self.eval_table.c.term_source == self.term.term_source)
                        .where(self.eval_table.c.term_id == self.term.term_id)
                        .where(self.eval_table.c.eval_type_id == 4))

    def remove_punctuation(self, input_string):
        translator = str.maketrans("", "", string.punctuation)
        result = input_string.translate(translator)
        return result

    def clean_text(self, text:str) -> str:
        # delete text in parentheses 
        text = re.sub(r'\([^)]*\)', '', text).strip().lower()
        text = text.replace("-", " ").replace("\"", "").replace("the", "").replace("this", "")
        text = self.remove_punctuation(text)
        text = " ".join(text.split())
        return text

    def check_inclusion(self, text:str, subtext:str) -> bool:
        if (text.lower().count(subtext.lower())>0): return True

        cleaned_subtext = self.clean_text(subtext)
        cleaned_text = self.clean_text(text)

        if (cleaned_text.count(cleaned_subtext) >0): return True
        return False

    def checked_before(self):
        query = self.eval_table.select().where(self.eval_table.c.answer_id == self.metadata.answer_id).where(self.eval_table.c.term_id == self.term.term_id).where(self.eval_table.c.model_id == self.model.model_id).where(self.eval_table.c.term_source == self.term.term_source)
        query_result = self.db.sql.execute(query)
        #self.logger.info(query.compile(compile_kwargs={"literal_binds": True}))
        #self.logger.info(query_result.fetchall())
        return query_result.rowcount > 0
    
    def term_usage_checked_before(self):
        query = self.eval_table.select(
            ).where(self.eval_table.c.answer_id == self.metadata.answer_id
            ).where(self.eval_table.c.term_id == self.term.term_id
            ).where(self.eval_table.c.model_id == FUNCTION_EVALUATOR_ID
            ).where(self.eval_table.c.term_source == self.term.term_source
            ).where(self.eval_table.c.eval_type_id == CODE_CHECK)
        query_result = self.db.sql.execute(query)
        self.logger.info(query.compile(compile_kwargs={"literal_binds": True}))
        return query_result.rowcount > 0

class EvaluatorModel(ABC):
    def __init__(self, model_name:str, db:HallucinationDb, settings) -> None:
        self.db = db
        self.settings = settings
        self.model_name = model_name
        self.models_table = self.db.GetTableDefinition(self.db.MODELS_TABLE)
        self.model_id = self.GetModelId()
        self.system_prompt_certainty = templates.certainty_reflection_system
        self.system_prompt_meaning = templates.meaning_reflection_system
        self.parser = LLmResponseParser()


    @abstractmethod
    def check_term_accepted(self, metadata:QAMetadata, term:Term):
        pass

    @abstractmethod
    def check_term_meaning(self, metadata:QAMetadata, term:Term):
        pass
        
    def GetModelId(self, model_name:str = None):
        if model_name is None: model_name = self.model_name

        query = self.models_table.select().where(self.models_table.c.name == model_name)
        query_result = self.db.sql.execute(query)

        if query_result.rowcount == 0:
            insert_query = self.models_table.insert().values(name = model_name)
            insert_result = self.db.sql.execute(insert_query)
            model_id = insert_result.inserted_primary_key[0]
        else:
            model_id = query_result.fetchone()[0]
        return model_id

    def parse_certainty_response(self, response_str:str, term:Term) -> tuple[str, int]:
        try:
            output = self.parser.parse_response(response_str)
            if output["certainty"].lower() == "unknown":
                if term.isHallucinative: eval_label = NO_HALLUCINATION
                else: eval_label = IRRELEVANT
            else:
                eval_label = int((output["certainty"].lower() == "mentioned") == (term.term_source == 0))

            # delete this line later
            #response_str = response_str.replace("WARNING: Failed to parse response: ", "").replace("<|im_end|>", "").strip()
        except Exception as e:
            eval_label = NO_HALLUCINATION
            logging.warning(f"Failed to parse response: {e} \n {response_str}")
            #uncomment this later
            response_str = "WARNING: Failed to parse response: " + response_str
        return response_str, eval_label

    def parse_meaning_response(self, response_str:str, term:Term) -> tuple[str, int]:
        try:
            output = self.parser.parse_response(response_str)
            eval_label = int(str(output["verified"]).lower() == "false")
                
            # delete this line later
            #response_str = response_str.replace("WARNING: Failed to parse response: ", "").replace("<|im_end|>", "").strip()
        except Exception as e:
            eval_label = NO_HALLUCINATION
            logging.warning(f"Failed to parse response: {e} \n {response_str}")
            #uncomment this later
            response_str = "WARNING: Failed to parse response: " + response_str
        return response_str, eval_label

class GptEvaluator(EvaluatorModel):
    def __init__(self, model_name: str, db: HallucinationDb, settings):
        super().__init__(model_name, db, settings)
        
        self.__check_settings(settings)
        self.client = OpenAI(api_key=settings.openai_api_key)

        #promptlayer.api_key = settings.promptlayer_api_key
        #self.openai = promptlayer.openai
        #self.openai.api_key = settings.openai_api_key

    def __check_settings(self, settings):
        if not settings.promptlayer_api_key:
            raise ValueError("Promptlayer API key is not set")
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key is not set")
        
    def try_gptapi_call(self, messages, temperature=0, model=GPT_MODEL_NAME):
        for i in range(3):
            try:
                return self.inner_api_call(messages, temperature, model)
            except TimeoutError as exc:
                logging.exception(f"Timeout: {exc}\nRetrying...")
                time.sleep(2)
                continue
            except Exception as exc:
                logging.exception(f"Exception: {exc}")
                time.sleep(6)
                continue

    @timeout(120)
    def inner_api_call(self, messages, temperature, model):
        return self.client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=messages)

    def check_term_accepted(self, metadata:QAMetadata, term:Term) -> tuple[str, int]:
        messages = self.get_certainty_prompt(metadata, term)
        response = self.try_gptapi_call(messages)
        response_str = response.choices[0].message.content
        return self.parse_certainty_response(response_str, term)
      
    def check_term_meaning(self, metadata:QAMetadata, term:Term):
        messages = self.get_meaning_verification_prompt(metadata, term)
        response = self.try_gptapi_call(messages)
        response_str = response.choices[0].message.content
        return self.parse_meaning_response(response_str, term)
    
    def get_certainty_prompt(self, metadata:QAMetadata, term:Term):
        user_prompt = templates.certainty_reflection_user.format(
            question = metadata.question,
            answer = metadata.answer,
            term = term.name
        )
        messages = [{"role": "system", "content": self.system_prompt_certainty},
                        {"role": "user", "content": user_prompt}]
        return messages
    
    def get_meaning_verification_prompt(self, metadata:QAMetadata, term:Term):
        user_prompt = templates.meaning_reflection_user.format(
            question = metadata.question,
            answer = metadata.answer,
            term = term.name,
            term_definition = term.meaning
        )
        messages = [{"role": "system", "content": self.system_prompt_meaning},
                        {"role": "user", "content": user_prompt}]
        return messages

class OpenEvaluator(EvaluatorModel):
    def __init__(self, model_name, db, settings):
        super().__init__(model_name, db, settings)
        self.load_model()

    def load_model(self):
        self.model = OllamaClient(model_name=self.model_name)
        
    def check_term_accepted(self, metadata:QAMetadata, term:Term):
        messages = self.get_certainty_prompt(metadata, term)
        response = self.try_model_call(messages)
        return self.parse_certainty_response(response, term)

    def check_term_meaning(self, metadata:QAMetadata, term:Term):
        messages = self.get_meaning_verification_prompt(metadata, term)
        response = self.try_model_call(messages)
        return self.parse_meaning_response(response, term)

    def try_model_call(self, messages):
        for i in range(3):
            try:
                return self.model_call(messages)
            except TimeoutError as exc:
                logging.exception(f"Timeout: {exc}")
                return "WARNING! Timeout."
            except Exception as exc:
                logging.exception(f"Exception: {exc}")
                time.sleep(6)
                continue
        return "WARNING! Model failure."

    @timeout(500)
    def model_call(self, messages):
        return self.model.generate(messages, raw=True) 

    def get_certainty_prompt(self, metadata:QAMetadata, term:Term):
        user_prompt = templates.certainty_reflection_user.format(
            question = metadata.question,
            answer = metadata.answer,
            term = term.name
        )
        messages = self.wrap_llama_template([user_prompt], self.system_prompt_certainty)
        return messages

    def get_meaning_verification_prompt(self, metadata:QAMetadata, term:Term):
        user_prompt = templates.meaning_reflection_user.format(
            question = metadata.question,
            answer = metadata.answer,
            term = term.name,
            term_definition = term.meaning
        )
        messages = self.wrap_llama_template([user_prompt], self.system_prompt_meaning)
        return messages

    def wrap_llama_template(self, prompts:list[str], system_message: str = None, replies:list[str] = []):
        llama_templates = llama2Templates(system_message)
        return llama_templates.generate_message(prompts, replies)