import json
import os
import threading

import openai
import requests
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

from dome.auxiliary.DAO import DAO
from dome.auxiliary.enums.intent import Intent
from dome.config import (PNL_GENERAL_THRESHOLD, USELESS_EXPRESSIONS_FOR_INTENT_DISCOVERY, TIMEOUT_MSG_PARSER,
                         DEBUG_MODE, USE_PARSER_CACHE, HUGGINGFACE_TOKEN, WHERE_CLAUSE_WORDS, INTENT_MAP)
import re


class AIEngine(DAO):
    def get_db_file_name(self) -> str:
        return "kdb.sqlite"

    GENERAL_BOT_CONTEXT = "The context here is around a chatbot that updates the data model of its system " \
                          "using the messages received from the end user in Natural Language.\nSo the " \
                          "user can send messages to the chatbot with the following intents categories:\n- " \
                          "GREETING: when the user starts an interaction with the chatbot (v.g.: 'hello'; " \
                          "'good morning'; 'hi'; etc.)\n- GOODBYE: when the user ends interaction with the " \
                          "chatbot (v.g.: 'bye bye'; 'thank you'; 'goodbye'; etc.)\n- HELP: when the user is " \
                          "asking for help to understand how must interact with the chatbot (v.g.: 'help me'; " \
                          "'please, help'; 'I need some help.'; etc.)\n- CONFIRMATION: when the user is " \
                          "confirming some operation. (v.g: 'ok') \n- CANCELLATION: when the user intends to " \
                          "cancel some operation. (v.g: 'cancel').\n- CRUD: when the user is asking for a " \
                          "CRUD type operation. CRUD operations refers to the four basic operations a " \
                          "software application should be able to perform: Create, Read, Update, and Delete. " \
                          "(v.g.: 'add a student with name=Anderson'; 'for the student with name=Anderson, " \
                          "update the age for 43'; 'get the teachers with name Paulo Henrique.'; etc.)"

    def __init__(self, AC):
        super().__init__()
        self.__AC = AC  # Autonomous Controller Object
        self.__pipelines = {}

        # adding specialized pipelines/models
        self.__addToPipeline('text-similarity', SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'))

    # sentiment analysis
    def msgIsPositive(self, msg) -> bool:
        response = self.getPipeline('sentiment-analysis')(msg)
        # return True if positive or False if negative
        return response[0]['label'] == 'POSITIVE'

    def posTagMsg(self, msg, model="vblagoje/bert-english-uncased-finetuned-pos", aggregation_strategy=None):
        # https://huggingface.co/vblagoje/bert-english-uncased-finetuned-pos
        # configure the pipeline
        token_classifier = self.getPipeline(pipeline_name="token-classification",
                                            pipeline_key="posTag-m_" + model + "as_" + str(aggregation_strategy),
                                            model=model, aggregation_strategy=aggregation_strategy)

        considered_msg = msg.lower()
        tokens = token_classifier(considered_msg)

        if aggregation_strategy is None:
            for i in range(len(tokens) - 1, -1, -1):
                if i > 0 and tokens[i]['word'].startswith('##'):
                    # merge the token that word starts with ## (e.g. ##ing) with the previous token
                    tokens[i - 1]['word'] += tokens[i]['word'][2:]
                    tokens[i - 1]['end'] = tokens[i]['end']
                    tokens[i]['entity'] = None
                    tokens[i]['word'] = None
                elif 0 < i < (len(tokens) - 1) and tokens[i]['word'] == '-' and tokens[i - 1]['entity'] == 'NOUN' and \
                        tokens[i + 1]['entity'] == 'NOUN':
                    # merge the token that is a hyphen with the previous and next token
                    tokens[i - 1]['word'] += tokens[i]['word'] + tokens[i + 1]['word']
                    tokens[i - 1]['end'] = tokens[i + 1]['end']
                    tokens[i]['entity'] = None
                    tokens[i]['word'] = None
                    tokens[i + 1]['entity'] = None
                    tokens[i + 1]['word'] = None
                elif 0 < i < len(tokens) and tokens[i - 1]['entity'] == 'ADJ' and \
                        tokens[i]['entity'] == 'NOUN':
                    # merge the token that is a noun with the previous token
                    tokens[i - 1]['word'] += ' ' + tokens[i]['word']
                    tokens[i - 1]['end'] = tokens[i]['end']
                    tokens[i - 1]['entity'] = 'NOUN'
                    tokens[i]['entity'] = None
                    tokens[i]['word'] = None
                elif tokens[i]['word'] == 'delete' and tokens[i]['entity'] == 'PROPN':
                    # to solve bug about delete expression that the model recognizes as PROPN
                    tokens[i]['entity'] = 'VERB'

        return tokens

    def get_entities_map(self) -> dict:
        return self.__AC.get_entities_map()

    def get_all_attributes(self) -> set:
        attributes = set()
        for class_key in self.__AC.get_entities_map().keys():
            for att_on_model in self.__AC.get_entities_map()[class_key].getAttributes():
                attributes.add(att_on_model.name)
        return attributes

    def add_alternative_entity_name(self, entity_name, alternative):
        pass  # for evaluation avoid to add to database
        self._execute_query("INSERT OR IGNORE INTO synonymous(entity_name, alternative) VALUES (?,?)",
                            (entity_name, alternative,))

    # get entity_name by alternative name from database
    def get_entity_name_by_alternative(self, alternative) -> str:
        query_result = self._execute_query_fetchone("SELECT entity_name FROM synonymous WHERE alternative = ?",
                                                    (alternative,))
        if query_result is None:
            return None
        # else
        return query_result['entity_name']

    def entitiesAreSimilar(self, entity_name, alternative, threshold=PNL_GENERAL_THRESHOLD) -> bool:
        # if the texts are equal, return True
        if entity_name == alternative:
            return True
        cached_entity_name = self.get_entity_name_by_alternative(alternative)
        if entity_name == cached_entity_name:
            return True
        if cached_entity_name is not None:
            return False
        # else test similarity
        model = self.getPipeline("text-similarity")
        # replacing '_' by ' ' to improve similarity
        considered_entity_name = entity_name.replace('_', ' ')
        # Compute embedding for both texts
        embedding_1 = model.encode(considered_entity_name, convert_to_tensor=True)
        embedding_2 = model.encode(alternative, convert_to_tensor=True)
        result = util.pytorch_cos_sim(embedding_1, embedding_2)[0][0].item()
        if result > threshold:
            self.add_alternative_entity_name(entity_name, alternative)
            return True
            # else
        return False

    def question_answerer_remote(self, question, context, options=None):

        def __call_hf(input_text):
            API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xl"
            headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}
            payload = {"inputs": input_text, "options": {"use_cache": True, "wait_for_model": True}}
            __response = requests.post(API_URL, headers=headers, json=payload)
            return __response.json()[0]['generated_text'].strip()

        def __call_openai(question, context, options=None):
            openai.api_key = os.getenv("OPENAI_API_KEY")
            main_messages = [{"role": "system", "content": context},
                         {"role": "user", "content": "answer me only with the answer in a string format"},
                         {"role": "user", "content": "The error messages must follow this format: 'dome_openai_error_message = {error message detail}'"},
                         {"role": "user", "content": "When the model can't find the answer or not understand the question, always answer me in the same format as the error messages."},
                         {'role': 'user', 'content': question}]
            if options is not None:
                main_messages.append({"role": "system", "content": "options: %s" % options})

            def _do_request(messages):
                return openai.ChatCompletion.create(
                    # model="text-davinci-003",
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0,
                ).choices[0].message.content.strip()

            _response = _do_request(main_messages)

            if _response.startswith('dome_openai_error_message = '):
                return None

            # dealing with corner cases
            check_messages = [{"role": "system", "content": "I need to check if the language model response is valid or not. If it is not valid, the text will inform that the model can't be able to answer the original question."},
                              {"role": "system", "content": "Examples of INVALID responses: "
                                                            "\n-I'm sorry, I didn't understand what you are asking for. Can you please clarify?"
                                                            "\n-The entity class cannot be determined based on the user's message 'get'. Please provide more information or context to determine the entity class."
                                                            "\n-I'm sorry, I didn't understand the question. Could you please provide more context or clarify what you are asking?"
                               },
                              {"role": "user", "content": "Answer only with 'yes' or 'no'"},
                              {"role": "user", "content": "OPTIONS: [yes, no]"},
                              {'role': 'user', 'content': 'Answer me (with "yes" or "no") if the following language model response is valid or not: ' + _response}]

            check_response = _do_request(check_messages).lower()
            if check_response in ['no', 'no.']:
                return None

            # clean the response from the prompt
            _response = _response.replace("The entity class that the user's current message refers to is ", '')
            if '=' in _response:
                _response = _response.split('=')[1]
            if 'is "' in _response:
                _response = _response.split('is "')[1]
            if 'is \'' in _response:
                _response = _response.split('is \'')[1]
            _response = _response.strip('.').strip("'").strip('"').strip()
            return _response

        def prompt(question_, context_, options_=None):
            # input_text = '-QUESTION: %s-CONTEXT: %s-OPTIONS: %s' % (__question + '\n', fact + '\n', options)
            input_text = '-QUESTION: %s-CONTEXT: %s' % (question_ + '\n', context_)
            if options_:
                input_text += '\n-OPTIONS: %s' % options_
            if DEBUG_MODE:
                print('PROMPT -------------------')
                print(input_text)
                print('--------------------------')
            return __call_openai(question_, context_, options_)
            # return __call_hf(input_text)

        response_str = prompt(question, context, options)
        response = {"answer": response_str}

        if DEBUG_MODE:
            print('RESPONSE -----------------')
            print(response)
            print('--------------------------')

        return response

    def question_answerer_local(self, question, context):
        models = ['deepset/roberta-base-squad2',
                  'distilbert-base-cased-distilled-squad',
                  'deepset/minilm-uncased-squad2']

        # iterate over models to find the best answer (the one with the highest score)
        best_answer = None
        best_score = 0
        for model in models:
            answer = self.__get_question_answer_pipeline(model)(question, context)
            if answer['score'] > PNL_GENERAL_THRESHOLD:
                # return immediately if the answer is good enough
                return answer
            # else
            if answer['score'] > best_score:
                best_answer = answer
                best_score = answer['score']

        return best_answer

    def __get_question_answer_pipeline(self, model):
        return self.getPipeline(pipeline_name='question-answering', model=model,
                                pipeline_key='question-answering-m_' + model)

    def get_zero_shooter_pipeline(self):
        return self.getPipeline(pipeline_name="zero-shot-classification", model="facebook/bart-large-mnli")

    def getPipeline(self, pipeline_name, model=None, config=None, aggregation_strategy=None, pipeline_key=None):
        if pipeline_name not in self.__pipelines:
            self.__addToPipeline(pipeline_name, pipeline(pipeline_name, model=model, config=config,
                                                         aggregation_strategy=aggregation_strategy),
                                 pipeline_key=pipeline_key)
        return self.__pipelines[pipeline_key if pipeline_key else pipeline_name]

    def __addToPipeline(self, pipeline_name, pipeline_object, pipeline_key=None):
        self.__pipelines[pipeline_key if pipeline_key else pipeline_name] = pipeline_object

    def add_parser_cache(self, user_msg, intent, entity_class, attributes, filter_attributes):
        # the Parser Cache stores the intent, the entity class, and the attributes map
        # considering an exact match with the user message
        self._execute_query("INSERT or IGNORE INTO parser_cache(user_msg, user_msg_len, processed_intent, "
                            "processed_class, processed_attributes, processed_filter_attributes) VALUES (?,?,?,?,?,?)",
                            (user_msg.lower(), len(user_msg), str(intent), entity_class,
                             json.dumps(attributes, default=str) if attributes else None,
                             json.dumps(filter_attributes, default=str) if filter_attributes else None))

    def get_parser_cache(self, user_msg):
        return self._execute_query_fetchone("SELECT * FROM vw_considered_parser_cache WHERE user_msg = ?",
                                            (user_msg.lower(),))

    # Wrapper class for encapsulate parsing services
    class __MsgParser:
        @staticmethod
        def get_bot_context():
            return AIEngine.GENERAL_BOT_CONTEXT

        def __init__(self, user_msg, aie_obj) -> None:
            self.user_msg = user_msg
            # removing double spaces from self.user_msg
            self.user_msg = re.sub(' +', ' ', self.user_msg)
            self.__AIE = aie_obj
            self.intent = None
            self.entity_class = None
            self.attributes = None
            self.filter_attributes = None  # where clause attributes

            # verifying if there is cache for the user_msg in database
            cached_parser = None
            if USE_PARSER_CACHE:
                cached_parser = self.__AIE.get_parser_cache(self.user_msg)

            if cached_parser:
                self.intent = Intent(cached_parser['considered_intent'])
                self.entity_class = cached_parser['considered_class']
                # set self.attributes as a dict from the string loaded from cached_parser['considered_attributes'] json
                if cached_parser['considered_attributes']:
                    self.attributes = json.loads(cached_parser['considered_attributes'])
                if cached_parser['considered_filter_attributes']:
                    self.filter_attributes = json.loads(cached_parser['considered_filter_attributes'])
            else:
                # pos-tagging the user_msg
                self.tokens = self.__AIE.posTagMsg(user_msg)
                # build a tokens type map
                self.tokens_by_type_map = {}
                for token in self.tokens:
                    if not (token['entity'] in self.tokens_by_type_map):
                        self.tokens_by_type_map[token['entity']] = []
                    self.tokens_by_type_map[token['entity']].append(token)

                self.question_answerer = self.__AIE.question_answerer_remote

                # discovering of the intent
                self.intent = self.__getIntentFromMsg()

                # discovering of the entity class
                if self.intent in (Intent.DELETE, Intent.ADD, Intent.READ, Intent.UPDATE):
                    if self.tokens[0]['entity'] != 'VERB' and self.tokens[0]['word'] == self.intent:
                        # adjusting the first token to be a verb, considering the intent (see test_corner_case_13)
                        self.tokens[0]['entity'] = 'VERB'

                    self.entity_class = self.__get_entity_class_from_msg()

                # discovering of the attributes
                if self.entity_class:
                    self.attributes, self.filter_attributes = self.__get_attributes_from_msg()

            # saving the cache in database
            if not cached_parser:
                self.__AIE.add_parser_cache(user_msg, self.intent, self.entity_class,
                                            self.attributes, self.filter_attributes)

        def __getIntentFromMsg(self) -> Intent:
            # get the intent from the user_msg
            # the intent is the most likely class of the zero-shot-classification pipeline
            # considering the user_msg as a text to classify
            # and the intents as the possible classes
            # the intent is the class with the highest probability

            # finding if the user's message intention in a "one-word" way
            considered_msg = self.user_msg.lower()
            # clear some problematic or useless expressions from user_msg for discovery the intent
            for useless_expression in USELESS_EXPRESSIONS_FOR_INTENT_DISCOVERY:
                considered_msg = considered_msg.replace(useless_expression, "")
            # seeking for direct commands
            if self.tokens_by_type_map and 'VERB' in self.tokens_by_type_map:
                first_verb = self.tokens_by_type_map['VERB'][0]['word']
                # finding in msg a direct command
                candidate_intent = Intent.fromString(first_verb)
                if candidate_intent:
                    return candidate_intent
            # else:
            candidate_intent = Intent.fromString(considered_msg)
            if candidate_intent:
                return candidate_intent
            # else: the one-word way is not enough to discover the intent

            # test if the user_msg make some sense
            question = "Does the user message make any sense?"
            context = self.get_bot_context() + "\nSo, the first step is to discover if the user's current message " \
                                               "makes some sense considering that context.\nThe user's current " \
                                               "message is: '" + self.user_msg + "'"
            options = "Yes, No"
            question_answer = self.question_answerer(question, context, options)
            if question_answer['answer'] == 'No':
                return Intent.MEANINGLESS
            # else: the user_msg makes some sense

            # finding if the user's message intention refers to a CRUD operation or not
            question = "What is the type of CRUD operation the user's current message refers to?"
            context = self.get_bot_context() + "\nSo, answer me what is the type of CRUD operation " \
                                               "the user's current message refers to. \nThe user's " \
                                               "current message is: '" + self.user_msg + "'"
            options = "CREATE, READ, UPDATE, DELETE"
            question_answer = self.question_answerer(question, context, options)
            if question_answer['answer'] == 'CREATE':
                return Intent.ADD
            elif question_answer['answer'] == 'READ':
                return Intent.READ
            elif question_answer['answer'] == 'UPDATE':
                return Intent.UPDATE
            elif question_answer['answer'] == 'DELETE':
                return Intent.DELETE
            else:  # the user's message intention does not refer to a CRUD operation
                question = "Is the user's message intention refers to a greeting?"
                context = self.get_bot_context() + "\nSo, answer me if the user's current message refers to a " \
                                                   "greeting. \nFollow some examples of greetings:\n- 'hello'; 'good " \
                                                   "morning'; 'hi';\nThe user's current message is: '" + \
                          self.user_msg + "'"
                options = "Yes, No"
                question_answer = self.question_answerer(question, context, options)
                if question_answer['answer'] == 'Yes':
                    return Intent.GREETING
                else:
                    question = "Is the user's message intention refers to a goodbye?"
                    context = self.get_bot_context() + "\nSo, answer me if the user's current message refers to a " \
                                                       "goodbye. \nFollow some examples of goodbye:\n- 'bye bye'; " \
                                                       "goodbye. \nThe user's current message is: '" + self.user_msg + \
                              "'"
                    question_answer = self.question_answerer(question, context, options)
                    if question_answer['answer'] == 'Yes':
                        return Intent.GOODBYE
                    else:
                        question = "Is the user's message intention refers to a help?"
                        context = self.get_bot_context() + "\nSo, answer me if the user's current message refers to a " \
                                                           "help. \nFollow some examples of help:\n- 'help me'; help. " \
                                                           "\nThe user's current message is: '" + self.user_msg + "'"
                        question_answer = self.question_answerer(question, context, options)
                        if question_answer['answer'] == 'Yes':
                            return Intent.HELP
                        else:
                            question = "Is the user's message intention refers to a confirmation?"
                            context = self.get_bot_context() + "\nSo, answer me if the user's current message refers " \
                                                               "to a confirmation. \nFollow some examples of " \
                                                               "confirmation:\n- 'ok'; 'yes'; 'yep'. \nThe user's " \
                                                               "current message is: '" + self.user_msg + "'"
                            question_answer = self.question_answerer(question, context, options)
                            if question_answer['answer'] == 'Yes':
                                return Intent.CONFIRMATION
                            else:
                                question = "Is the user's message intention refers to a cancellation?"
                                context = self.get_bot_context() + "\nSo, answer me if the user's current message " \
                                                                   "refers to a cancellation. \nFollow some examples " \
                                                                   "of cancellation:\n- 'cancel'; 'no'; 'nope'. \n" \
                                                                   "The user's current message is: '" + self.user_msg + \
                                          "'"
                                question_answer = self.question_answerer(question, context, options)
                                if question_answer['answer'] == 'Yes':
                                    return Intent.CANCELLATION
            # else
            return Intent.MEANINGLESS

        def __entities_are_similar(self, entity1, entity2) -> bool:
            return self.__AIE.entitiesAreSimilar(entity1, entity2)

        def __get_entity_class_from_msg(self) -> str:
            question = "What is the entity class that the user's message refers to?"
            context = self.get_bot_context() + "\nThe user's message intent is '" + self.intent.name + "'."
            context += "\nNow, the chatbot need discover the entity class that the user's message refers to, " \
                       "considering that context."
            options = ''

            # adding the candidates
            candidates = []
            attributes = self.__AIE.get_all_attributes()

            # iterate over the tags after the verb token (intent)
            for token in self.tokens:
                if (token['entity'] == 'NOUN' and  # only nouns
                        not (token['word'] in attributes)):  # not a known attribute
                    candidates.append(token['word'])
                    context += "\nPerhaps the entity class may be this: " + token['word'] + ". "

            # adding current classes
            for class_key in self.__AIE.get_entities_map():
                for candidate in candidates:
                    if self.__entities_are_similar(class_key, candidate):
                        context += "\nThere already is an entity class named: " + class_key

            context += "\nSo, answer me what is the entity class that the user's current message refers to." \
                       "\nThe user's current message is: '" + self.user_msg + "'."

            options = ", ".join(candidates)

            response = self.__AIE.question_answerer_remote(question, context, options)
            entity_class_candidate = response['answer']

            if not entity_class_candidate:
                return None

            if entity_class_candidate == 'CRUD' or entity_class_candidate == self.intent \
                    and entity_class_candidate != 'show':  # show is a special case (v.g. show me the shows)
                # it's an error. Probably the user did not inform the entity class in the right way.
                return None
            # else
            cached_entity_class = self.__AIE.get_entity_name_by_alternative(entity_class_candidate)
            if cached_entity_class:
                return cached_entity_class
            # else
            # add the entity class to the cache
            # update the entity_class_candidate replacing special characters for '_'
            entity_class_candidate_original = entity_class_candidate
            entity_class_candidate = re.sub(r'[^a-zA-Z0-9_]', '_', entity_class_candidate)

            if self.intent in [Intent.ADD, Intent.UPDATE]:
                self.__AIE.add_alternative_entity_name(entity_class_candidate, entity_class_candidate)
                self.__AIE.add_alternative_entity_name(entity_class_candidate, entity_class_candidate_original)

            return entity_class_candidate

        def __get_attributes_from_msg(self) -> tuple:
            processed_attributes = {}
            CONTEXT_PREFIX = "This is the user command: '"
            where_clause = None
            where_clause_attributes = {}
            where_clause_idx_start = len(self.user_msg)
            where_clause_idx_end = -1
            user_msg_without_where_clause = self.user_msg

            def generate_example(sentence, answer) -> str:
                return "\nFor example, if the user's message is \"" + sentence + \
                    "\", your answer must be \"" + answer + "\"."

            def generate_options():
                _options = ''
                _options_list = []
                valid_tokens = {'NOUN', 'PROPN', 'NUM'}
                # for each token in self.tokens, generate all possible options
                for _i in range(len(self.tokens)):
                    if self.tokens[_i]['entity'] != 'NOUN' or \
                            self.tokens[_i]['word'] in WHERE_CLAUSE_WORDS or \
                            self.tokens[_i]['word'] in INTENT_MAP['UPDATE'] or \
                            self.__entities_are_similar(self.entity_class, self.tokens[_i]['word']):
                        continue
                    # else
                    valid_sentence = False
                    for _j in range(_i+1, len(self.tokens)):
                        if self.tokens[_j]['word'] in WHERE_CLAUSE_WORDS or \
                                self.tokens[_j]['word'] in INTENT_MAP['UPDATE']:
                            break
                        if self.tokens[_j]['entity'] in valid_tokens and \
                                not self.__entities_are_similar(self.entity_class, self.tokens[_i]['word']):
                            valid_sentence = not valid_sentence  # true only each couple of valid tokens
                            if valid_sentence:
                                # generate the sentence using the token start and end to preserve the case of the words
                                _options_list.append(self.user_msg[self.tokens[_i]['start']:self.tokens[_j]['end']])
                                _options += '"' + _options_list[-1] + '", '
                return _options[:-2], _options_list

            if self.intent == Intent.UPDATE:
                options, options_list = generate_options()
                question = "Considering the follow user's message:\n\"" + self.user_msg + '"'
                question += "\nIn that user's message, what type of '" + self.entity_class + \
                            "' must be updated? Give me the answer as a exact subsentence of the user's message."
                question += "You must consider that the user's message is an SQL dialect near natural human language. "
                question += "\nYour answer must include at least one noun that corresponds to the filter's field name " \
                            "and one expression (one or more words) that corresponds to the field's value."
                question += '\nSo, complete to me: "Update ' + self.entity_class + ' for all ' + \
                            self.entity_class + ' with ?"'
                question += "\nThe operation I've already found out: it's an " + str(self.intent)
                question += ".\nThe name of the table is '" + self.entity_class + "'."
                question += "\nNow I'm trying to figure out the snippet of the user's message that would most likely " \
                            "correspond to a 'where' clause in an SQL command."
                question += "\nI want to know the pairs of field names and values I must apply to filter the "
                question += str(self.intent) + " operation in the table '" + self.entity_class + "'."
                question += generate_example("update " + self.entity_class + " set name = 'John' where id = 1",
                                             "id = 1")
                question += generate_example("update the book setting the title to 'The Lord of the Rings' "
                                             "when the author is 'J. R. R. Tolkien'", "author is 'J. R. R. Tolkien'")
                question += generate_example("for students with name='Anderson', set name='Anderson Silva' and age=45",
                                             "name='Anderson'")
                question += generate_example('Update students setting the age to 42 when name is Anderson',
                                             'name is Anderson')
                question += "\nYou must choose one and only one of the fragments in options below."
                for option in options_list:
                    question += "\n- " + option
                context = "user's message = " + self.user_msg

                where_clause = self.question_answerer(question, context, options)
                where_clause_idx_start = self.user_msg.find(where_clause['answer'])

                if where_clause_idx_start > -1:
                    where_clause_idx_end = where_clause_idx_start + len(where_clause['answer'])
                    where_clause = where_clause['answer']
                    user_msg_without_where_clause = self.user_msg.replace(where_clause, '')

            def idx_in_where_clause(idx):
                return where_clause and (where_clause_idx_start <= idx <= where_clause_idx_end)

            def __get_attributes_context(attribute_target, cur_token) -> str:
                fragment_long = self.user_msg[cur_token['start']:]
                fragment_short = self.user_msg[cur_token['end']:]
                considered_attributes = processed_attributes
                att_context = CONTEXT_PREFIX + self.user_msg + "'. "

                if where_clause:
                    att_context = CONTEXT_PREFIX + user_msg_without_where_clause + "'. "

                att_context += "\nThe intent of the user command is '" + str(self.intent) + "'. "
                att_context += "\nThe entity class is '" + str(self.entity_class) + "'. "
                if idx_in_where_clause(cur_token['start']):
                    fragment_long = where_clause
                    fragment_short = where_clause[where_clause.find(attribute_target) + len(attribute_target):]
                    considered_attributes = where_clause_attributes
                    att_context = ''

                att_context += '\nsentence fragment="' + fragment_long + '";'
                att_context += '\n The answer is a substring of "' + fragment_short + '".'
                att_context += '\n"' + attribute_target + '" is the name of a field in database.'
                att_context += '\nI\'m trying discover the value of the "' + attribute_target + \
                               '" in the sentence fragment.'
                att_context += '\nIn another words, complete to me "' + attribute_target + '=" ?'

                for att_name, att_value in considered_attributes.items():
                    att_context += "\nThe field '" + att_name + "' has the value '" + att_value + "'. "
                    att_context += " So, '" + attribute_target + "' is not '" + att_value + "'! "

                return att_context

            # finding the index of the entity class name in the message
            entity_class_token_idx = -1
            for i, token_i in enumerate(self.tokens):
                # advance forward until the token of the entity class is found
                if self.__AIE.get_entity_name_by_alternative(token_i['word']) == self.entity_class:
                    entity_class_token_idx = i
                    break

            # iterate over the tokens and find the attribute names and values
            j = 0
            while j < len(self.tokens):
                # advance forward until the token of the type "NOUN" is found (i.e. the first noun it is an
                # attribute name)
                token_j = None
                while j < len(self.tokens) and token_j is None:
                    if self.tokens[j]['entity'] == 'NOUN' and j != entity_class_token_idx:
                        token_j = self.tokens[j]
                    else:
                        j += 1

                if token_j:
                    # found the first noun after token_j. It is the attribute name
                    attribute_name = token_j['word']
                    # check if the attribute name is in the attributes set
                    if self.intent != Intent.UPDATE and attribute_name in processed_attributes:
                        # the attribute name is already in the attributes list. It's an error.
                        break
                    # ask by the attribute value using question-answering pipeline
                    response = self.question_answerer(question="What is the '" + attribute_name +
                                                               "' in the sentence fragment?"
                                                               "\nAnswer me with the exact substring of the sentence fragment." \
                                                               "\nAnswer me with only the value of the attribute."
                                                      , context=__get_attributes_context(attribute_name, token_j))

                    if not response['answer']:
                        # the question-answering pipeline was not able to find the answer
                        break

                    # update the j index to the next token after the attribute value
                    # get the end index in the original msg
                    att_value_idx_end = self.user_msg.find(response['answer'], token_j['end'])
                    if att_value_idx_end > -1:
                        att_value_idx_end += len(response['answer'])

                    if att_value_idx_end <= token_j['end']:
                        # inconsistency in the answer (see test.test_corner_case_10)
                        break
                    # else: all right
                    # add the pair of attribute name and attribute value in the result list
                    attribute_value = response['answer'].strip()
                    # clean the attribute value
                    if not attribute_value[0].isalnum():  # see test.test_add_5()
                        attribute_value = attribute_value[1:]
                    if attribute_value[-1] in ['"', "'"]:
                        attribute_value = attribute_value[:-1]

                    # update the attribute_name replacing special characters for '_'
                    attribute_name = re.sub(r'[^a-zA-Z0-9_]', '_', attribute_name)

                    # add the attribute pair to the correspondent map
                    if idx_in_where_clause(att_value_idx_end):
                        where_clause_attributes[attribute_name] = attribute_value
                    else:
                        processed_attributes[attribute_name] = attribute_value

                    if att_value_idx_end > -1:
                        # advance for the next token
                        while j < len(self.tokens) and self.tokens[j]['end'] <= att_value_idx_end:
                            j += 1
                else:
                    # no noun found after token_j. It is the end of the attribute list
                    break

            if self.intent == Intent.UPDATE and (not processed_attributes or not where_clause_attributes):
                # inconsistency in the answer, return none
                return None, None

            return processed_attributes, where_clause_attributes

        def get_tokens_by_type(self, entityType) -> list:
            if entityType in self.tokens_by_type_map:
                return self.tokens_by_type_map[entityType]
            # else
            return []

    def get_msg_parser(self, msg) -> __MsgParser:
        # Start the parser as a process
        msg_parser = None

        def set_parser():
            nonlocal msg_parser
            msg_parser = self.__MsgParser(msg, self)

        thread = threading.Thread(target=set_parser, name="MsgParser", daemon=True)
        # starting the thread and join with timeout to avoid deadlocks
        thread.start()
        thread.join(TIMEOUT_MSG_PARSER)

        if thread.is_alive():
            raise Exception("Timeout in MsgParser")

        return msg_parser

    # get all valid parser cache registers from database
    def get_all_considered_parser_cache(self) -> list:
        return self._execute_query("SELECT * FROM vw_considered_parser_cache")
