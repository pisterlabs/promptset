import string
import numpy as np
import utils
from utils import AnswerMapping
from nltk.corpus import stopwords
from models import OpenAIGPT


class BaseAlgorithm:
    defn = "An entity is an object, place, individual, being, title, proper noun or process that has a distinct and " \
           "independent existence. The name of a collection of entities is also an entity. Adjectives, verbs, numbers, " \
           "adverbs, abstract concepts are not entities. Dates, years and times are not entities"

    chatbot_init = "You are an entity recognition system. "
    entity_token_task = "In the sentence '[sent]'. The phrase '[token]' is an entity of type [type]. In one line explain why. \nAnswer: The phrase '[token]' is an entity of type [type] because"
    nonentity_token_task = "In the sentence '[sent]'. The phrase '[token]' is not an entity. In one line explain why. \nAnswer: The phrase '[token]' is not an entity because"

    # if [] = n then there are O(n^2) phrase groupings

    def __init__(self, model_fn=None, split_phrases=False, identify_types=True, resolve_disputes=True):
        self.defn = self.defn
        self.para = None
        self.model_fn = model_fn
        self.split_phrases = split_phrases
        self.exemplar_task = None
        self.format_task = None
        self.whole_task = None
        self.identify_types = identify_types
        self.resolve_disputes = resolve_disputes

    def set_para(self, para):
        self.para = para

    def set_model_fn(self, model_fn):
        self.model_fn = model_fn

    @staticmethod
    def clean_output(answers, typestrings=None):
        if typestrings is None:
            answers = list(set(answers))
            for trivial in ["", " ", ".", "-"] + stopwords.words('english'):
                while trivial in answers:
                    answers.remove(trivial)
        else:
            new_answers = []
            new_typestrings = []
            for i, ans in enumerate(answers):
                if ans in new_answers:
                    continue
                if ans in ["", " ", ".", "-"] + stopwords.words('english'):
                    continue
                new_answers.append(ans)
                new_typestrings.append(typestrings[i])
        for i in range(len(answers)):
            ans = answers[i]
            if "(" in ans:
                ans = ans[:ans.find("(")]
            ans = ans.strip().strip(''.join(string.punctuation)).strip()
            answers[i] = ans
        if typestrings is None:
            return answers
        else:
            return answers, typestrings


class Algorithm(BaseAlgorithm):
    def perform_span(self, true_tokens=None, verbose=False):
        assert self.identify_types and not self.split_phrases
        answers, typestrings, metadata = self.perform(verbose=verbose, deduplicate=False)
        return self.parse_span(answers, typestrings, metadata, true_tokens=true_tokens)

    def parse_span(self, answers, typestrings, metadata, true_tokens=None):
        para = self.para.lower()
        if true_tokens is not None:
            para_words = [token.lower() for token in true_tokens]
        else:
            para_words = para.split(" ")
        span_pred = ["O" for word in para_words]
        completed_answers = []
        split_tokens = ["'s", ":"]
        for i, answer in enumerate(answers):
            answer = answer.strip().lower()  # take any whitespace out and lowercase for matching
            if "(" in answer:
                answer = answer[:answer.find("(")].strip()  # in case some type annotation is stuck here
            types = typestrings[i]
            if "(" in types and ")" in types:
                types = types[types.find("(") + 1:types.find(")")]
            else:
                continue
            answer_token_split = answer
            for token in split_tokens:
                answer_token_split = (" "+token).join(answer_token_split.split(token))
            exists = answer in para or answer_token_split in para
            answer_multi_word = len(answer.split(" ")) > 1
            if not exists:
                continue
            if not answer_multi_word:
                if answer not in para_words:
                    continue
                multiple = para.count(answer) > 1
                if not multiple:  # easiest case word should be in para_words only once
                    index = para_words.index(answer)
                else:  # must find which occurance this is
                    n_th = completed_answers.count(answer.strip()) + 1
                    index = utils.find_nth_list(para_words, answer, n_th)
                if span_pred[index] == "O":
                    if "-" in types:  # then its FEWNERD
                        span_pred[index] = types
                    else:
                        span_pred[index] = "B-" + types
                completed_answers.append(answer)
            else:
                for token in split_tokens:
                    if token in answer:
                        answer = (" "+token).join(answer.split(token))
                answer_words = answer.split(" ")
                multiple = para.count(answer) > 1
                n_th = completed_answers.count(answer.strip()) + 1
                index = utils.find_nth_list_subset(para_words, answer, n_th)
                end_index = index + len(answer_words)
                if "-" in types:  # then its FEWNERD
                    span_pred[index] = types
                else:
                    span_pred[index] = "B-" + types
                for j in range(index+1, end_index):
                    if "-" in types:  # then its FEWNERD
                        span_pred[j] = types
                    else:
                        span_pred[j] = "I-" + types
                completed_answers.append(answer)
        return span_pred, metadata

    def perform(self, verbose=True, deduplicate=True):
        """

        :param model:
        :param paragraph:
        :return:
        """
        if isinstance(self.model_fn, OpenAIGPT):
            if not self.identify_types:
                if self.model_fn.is_chat():
                    answers, metadata = self.perform_chat_query(verbose=verbose)
                else:
                    answers, metadata = self.perform_single_query(verbose=verbose)
            else:
                if self.model_fn.is_chat():
                    answers, typestrings, metadata = self.perform_chat_query(verbose=verbose)
                else:
                    answers, typestrings, metadata = self.perform_single_query(verbose=verbose)
        else:
            if not self.identify_types:
                answers, metadata = self.perform_single_query(verbose=verbose)
            else:
                answers, typestrings, metadata = self.perform_single_query(verbose=verbose)
        if not self.identify_types:
            answers = list(set(answers))
        if self.split_phrases:
            new_answers = []
            if self.identify_types:
                new_typestrings = []
            for i, answer in enumerate(answers):
                if " " not in answer:
                    new_answers.append(answer)
                    if self.identify_types:
                        new_typestrings.append(typestrings[i])
                else:
                    minis = answer.split(" ")
                    for mini in minis:
                        new_answers.append(mini)
                        if self.identify_types:
                            new_typestrings.append(typestrings[i])
            answers = new_answers
            if self.identify_types:
                typestrings = new_typestrings
        if deduplicate:
            if self.identify_types:
                answers, typestrings = BaseAlgorithm.clean_output(answers, typestrings)
            else:
                answers = BaseAlgorithm.clean_output(answers)
        if not self.identify_types:
            return answers, metadata
        else:
            return answers, typestrings, metadata

    def perform_single_query(self, verbose=True):
        if self.exemplar_task is not None:
            task = self.defn + "\n" + self.exemplar_task + f" '{self.para}' \nAnswer:"
            output = self.model_fn(task)
            final = AnswerMapping.exemplar_format_list(output, identify_types=self.identify_types, verbose=verbose)
        else:
            task = self.defn + "\n" + self.format_task + f"\nParagraph: {self.para} \nAnswer:"
            output = self.model_fn(task)
            final = AnswerMapping.exemplar_format_list(output, identify_types=self.identify_types, verbose=verbose)
        if self.identify_types:
            final, typestrings = final
        if not self.identify_types:
            return final, output
        else:
            return final, typestrings, output

    def perform_chat_query(self, verbose=True):
        if self.exemplar_task is not None:
            system_msg = self.chatbot_init + self.defn + " " + self.whole_task
            msgs = [(system_msg, "system")]
            for exemplar in self.exemplars:
                if "Answer:" not in exemplar:
                    raise ValueError(f"Something is wrong, exemplar: \n{exemplar} \n Does not have an 'Answer:'")
                ans_index = exemplar.index("Answer:")
                msgs.append((exemplar[:ans_index+7].strip(), "user"))
                msgs.append((exemplar[ans_index+7:].strip(), "assistant"))
            msgs.append((f"\nParagraph: {self.para} \nAnswer:", "user"))
            output = self.model_fn(msgs)
            final = AnswerMapping.exemplar_format_list(output, identify_types=self.identify_types, verbose=verbose)
        else:
            system_msg = self.chatbot_init + self.defn + " " + self.format_task
            msgs = [(system_msg, "system"), (f"\nParagraph: {self.para} \nAnswer:", "user")]
            output = self.model_fn(msgs)
            final = AnswerMapping.exemplar_format_list(output, identify_types=self.identify_types, verbose=verbose)
        if self.identify_types:
            final, typestrings = final
        if not self.identify_types:
            return final, output
        else:
            return final, typestrings, output

    def get_annotation(self, token, ner_label):
        if ner_label == "O":
            task_string = self.nonentity_token_task.replace("[sent]", self.para)
            task_string = task_string.replace("[token]", token)
        else:
            task_string = self.entity_token_task.replace("[sent]", self.para)
            task_string = task_string.replace("[token]", token)
            task_string = task_string.replace("[type]", ner_label)
        if self.model_fn.is_chat():
            msgs = [(self.defn, "system"), (task_string, "user")]
            output = self.model_fn(msgs)
        else:
            task_string = self.defn + "\n" + task_string
            output = self.model_fn(task_string)
        return output

    def generate_annotations(self, tokens, ner_labels, max_falses=3):
        false_indices = []
        annots = []
        for i, token in enumerate(tokens):
            if ner_labels[i] != "O":
                annot = self.get_annotation(token, ner_labels[i])
                annots.append(annot)
            else:
                if token.strip().strip(string.punctuation).strip() == '' \
                        or token.strip() in stopwords.words('english') or token.isnumeric():
                    annots.append(None)
                else:
                    false_indices.append(i)
                    annot = self.get_annotation(token, "O")
                    annots.append(annot)
        if len(false_indices) > max_falses:
            false_indices = np.random.choice(false_indices, max_falses, replace=False)
            false_indices.sort()
        annot_str = "Answer: \n"
        no = 1
        for i, token in enumerate(tokens):
            if annots[i] is None:
                pass
            else:
                appendage = "\n" + f"{no}. {token} | {ner_labels[i] != 'O'} | {annots[i]}"
                if ner_labels[i] != 'O':
                    if ner_labels[i][:2] in ["B-", "I-"]:
                        label = ner_labels[i][2:]
                    else:
                        label = ner_labels[i]
                    annot_str = annot_str + appendage + f"({label})"
                    no += 1
                else:
                    if i in false_indices:
                        annot_str = annot_str + appendage
                        no += 1
        return annot_str


class MultiAlgorithm(Algorithm):

    def perform_span(self, true_tokens=None, resolve_disputes=False, verbose=False):
        assert self.identify_types and not self.split_phrases
        answers, typestrings, metadata = self.perform(verbose=verbose, deduplicate=False)
        span_pred, metadata = self.parse_span(answers, typestrings, metadata, query=True, true_tokens=true_tokens, verbose=verbose)
        return span_pred, metadata

    def parse_span(self, answers, typestrings, metadata, true_tokens=None, query=False, verbose=False):
        para = self.para.lower()
        if true_tokens is not None:
            para_words = [token.lower() for token in true_tokens]
        else:
            para_words = para.split(" ")
        span_pred = ["O" for word in para_words]
        completed_answers = []
        split_tokens = ["'s", ":"]
        for i, answer in enumerate(answers):
            answer = answer.strip().lower()  # take any whitespace out and lowercase for matching
            if "(" in answer:
                answer = answer[:answer.find("(")].strip()  # in case some type annotation is stuck here
            if not self.resolve_disputes and query:
                types = self.get_type(answer, verbose=verbose)
                if types == -1:
                    types = typestrings[i]
                    if "(" in types and ")" in types:
                        types = types[types.find("(") + 1:types.find(")")]
                    else:
                        continue
            else:
                types = typestrings[i]
                if "(" in types and ")" in types:
                    types = types[types.find("(") + 1:types.find(")")]
                else:
                    continue
                if self.resolve_disputes:
                    other_types = self.get_type(answer, verbose=verbose)
                    if types != other_types:
                        types = self.resolve_dispute(answer, types, other_types, verbose=verbose)
                        if types == -1:
                            types = typestrings[i]
                            if "(" in types and ")" in types:
                                types = types[types.find("(") + 1:types.find(")")]
                            else:
                                continue

            answer_token_split = answer
            for token in split_tokens:
                answer_token_split = (" "+token).join(answer_token_split.split(token))
            exists = answer in para or answer_token_split in para
            answer_multi_word = len(answer.split(" ")) > 1
            if not exists:
                continue
            if not answer_multi_word:
                if answer not in para_words:
                    continue
                multiple = para.count(answer) > 1
                if not multiple:  # easiest case word should be in para_words only once
                    index = para_words.index(answer)
                else:  # must find which occurance this is
                    n_th = completed_answers.count(answer.strip()) + 1
                    index = utils.find_nth_list(para_words, answer, n_th)
                if span_pred[index] == "O":
                    if "-" in types:  # then its FEWNERD
                        span_pred[index] = types
                    else:
                        span_pred[index] = "B-" + types
                completed_answers.append(answer)
            else:
                for token in split_tokens:
                    if token in answer:
                        answer = (" "+token).join(answer.split(token))
                answer_words = answer.split(" ")
                multiple = para.count(answer) > 1
                n_th = completed_answers.count(answer.strip()) + 1
                index = utils.find_nth_list_subset(para_words, answer, n_th)
                end_index = index + len(answer_words)
                if "-" in types:  # then its FEWNERD
                    span_pred[index] = types
                else:
                    span_pred[index] = "B-" + types
                for j in range(index+1, end_index):
                    if "-" in types:  # then its FEWNERD
                        span_pred[j] = types
                    else:
                        span_pred[j] = "I-" + types
                completed_answers.append(answer)
        return span_pred, metadata

    def get_type(self, phrase, verbose=False):
        task = self.type_task
        afterphrase = f"Entity Phrase: {phrase}"
        if self.model_fn.is_chat():
            exemplars = self.type_exemplars
            answer = self.template_chat_query(task, exemplars, afterphrase, verbose=verbose)
        else:
            task = self.type_task_exemplars
            answer = self.template_single_query(task, afterphrase, verbose=verbose)
        if "(" in answer and ")" in answer:
            start = answer.find("(")
            end = answer.find(")")
            return answer[start+1:end]
        else:
            return -1

    def resolve_dispute(self, phrase, option1, option2, verbose=False):
        task = self.dispute_task
        afterphrase = f"Entity Phrase: {phrase}, Options: ({option1}), ({option2})"
        if self.model_fn.is_chat():
            exemplars = self.dispute_exemplars
            answer = self.template_chat_query(task, exemplars, afterphrase, verbose=verbose)
        else:
            task = self.dispute_task_exemplars
            answers = self.template_single_query(task, afterphrase, verbose=verbose)
        if "(" in answer and ")" in answer:
            start = answer.find("(")
            end = answer.find(")")
            return answer[start+1:end]
        else:
            return -1

    def template_chat_query(self, task, exemplars, afterphrase, verbose=False):
        system_msg = self.chatbot_init + self.defn + " " + task
        msgs = [(system_msg, "system")]
        for exemplar in exemplars:
            if "Answer:" not in exemplar:
                raise ValueError(f"Something is wrong, exemplar: \n{exemplar} \n Does not have an 'Answer:'")
            ans_index = exemplar.index("Answer:")
            msgs.append((exemplar[:ans_index+7].strip(), "user"))
            msgs.append((exemplar[ans_index+7:].strip(), "assistant"))
        msgs.append((f"\nParagraph: {self.para} \n{afterphrase} \nAnswer:", "user"))
        output = self.model_fn(msgs)
        if verbose:
            print(output)
        return output

    def template_single_query(self, task, afterphrase, verbose=False):
        task = self.defn + "\n" + task + f" '{self.para}' \n{afterphrase} \nAnswer:"
        output = self.model_fn(task)
        if verbose:
            print(output)
        return output


class Config:
    cot_format = """
    Format: 
    
    1. First Candidate | True | Explanation why the word is an entity (entity_type)
    2. Second Candidate | False | Explanation why the word is not an entity (entity_type)
    """

    no_tf_format = """
    1. First Entity | Explanation why the word is an entity (entity_type)
    2. Second Entity | Explanation why the word is not an entity (entity_type)
    """

    tf_format = """
    Format: 

    1. First Candidate | True | (entity_type)
    2. Second Candidate | False | (entity_type)
    """

    exemplar_format = """
    Format:    
    
    1. First Entity | (entity_type)
    2. Second Entity | (entity_type)
    """

    def set_config(self, alg, exemplar=True, coT=True, tf=True, defn=True):
        if isinstance(alg, MultiAlgorithm):
            coT = False
            tf = False
            exemplar = True
            type_task = "Q: Given the paragraph below and the entity phrase, identify what type the entity is \nParagraph:"
            alg.type_exemplars = self.type_exemplars
            exemplar_construction = ""
            for exemplar in self.type_exemplars:
                exemplar_construction = exemplar_construction + type_task + "\n"
                exemplar_construction = exemplar_construction + exemplar + "\n"
            exemplar_construction = exemplar_construction + type_task + "\n"
            alg.type_task_exemplars = exemplar_construction
            alg.type_task = type_task

            dispute_task = "Q: Given the paragraph below, the entity phrase and two proposed entity types, identify what the actual type of the entity is \nParagraph:"
            alg.dispute_exemplars = self.dispute_exemplars
            exemplar_construction = ""
            for exemplar in self.dispute_exemplars:
                exemplar_construction = exemplar_construction + dispute_task + "\n"
                exemplar_construction = exemplar_construction + exemplar + "\n"
            exemplar_construction = exemplar_construction + dispute_task + "\n"
            alg.dispute_task_exemplars = exemplar_construction
            alg.dispute_task = dispute_task
        if defn:
            alg.defn = self.defn
        else:
            alg.defn = ""
        if not exemplar:
            alg.exemplar_task = None
            if coT:
                if tf:
                    whole_task = "Q: Given the paragraph below, identify a list of possible entities " \
                                 "and for each entry explain why it either is or is not an entity. Answer in the format: \n"

                    alg.format_task = whole_task + self.cot_format
                else:
                    whole_task = "Q: Given the paragraph below, identify a list of entities " \
                                 "and for each entry explain why it is an entity. Answer in the format: \n"

                    alg.format_task = whole_task + self.no_tf_format

            else:
                whole_task = "Q: Given the paragraph below, identify the list of entities " \
                             "Answer in the format: \n"

                if not tf:
                    alg.format_task = whole_task + self.exemplar_format
                else:
                    alg.format_task = whole_task + self.tf_format
        else:
            alg.format_task = None
            if coT:
                if tf:
                    whole_task = "Q: Given the paragraph below, identify a list of possible entities " \
                                 "and for each entry explain why it either is or is not an entity. \nParagraph:"
                    alg.whole_task = whole_task
                    alg.exemplars = self.cot_exemplars
                    exemplar_construction = ""
                    for exemplar in self.cot_exemplars:
                        exemplar_construction = exemplar_construction + whole_task + "\n"
                        exemplar_construction = exemplar_construction + exemplar + "\n"
                    exemplar_construction = exemplar_construction + whole_task + "\n"
                    alg.exemplar_task = exemplar_construction
                else:
                    whole_task = "Q: Given the paragraph below, identify a list of entities " \
                                 "and for each entry explain why it is an entity. \nParagraph:"
                    alg.whole_task = whole_task
                    alg.exemplars = self.no_tf_exemplars
                    exemplar_construction = ""
                    for exemplar in self.no_tf_exemplars:
                        exemplar_construction = exemplar_construction + whole_task + "\n"
                        exemplar_construction = exemplar_construction + exemplar + "\n"
                    exemplar_construction = exemplar_construction + whole_task + "\n"
                    alg.exemplar_task = exemplar_construction
            else:
                whole_task = "Q: Given the paragraph below, identify the list of entities \nParagraph:"
                exemplar_construction = ""
                if not tf:
                    e_list = self.exemplars
                else:
                    e_list = self.tf_exemplars
                alg.whole_task = whole_task
                alg.exemplars = e_list
                for exemplar in e_list:
                    exemplar_construction = exemplar_construction + whole_task + "\n"
                    exemplar_construction = exemplar_construction + exemplar + "\n"
                exemplar_construction = exemplar_construction + whole_task + "\n"
                alg.exemplar_task = exemplar_construction

    def autogenerate_annotations(self, alg, texts, tokens, labels, max_examples=3):
        cot_exemplars = []
        for i in range(len(texts[:max_examples])):
            text = texts[i]
            token = tokens[i]
            label = labels[i]
            alg.set_para(text)
            exemplar = text + "\n" + alg.generate_annotations(token, label)
            cot_exemplars.append(exemplar)
        self.cot_exemplars = cot_exemplars




class ConllConfig(Config):
    defn = "An entity is a person (PER), title, named organization (ORG), location (LOC), country (LOC) or nationality (MISC)." \
           "Names, first names, last names, countries are entities. Nationalities are entities even if they are " \
           "adjectives. Sports, sporting events, adjectives, verbs, numbers, " \
                "adverbs, abstract concepts, sports, are not entities. Dates, years and times are not entities. " \
           "Possessive words like I, you, him and me are not entities. " \
           "If a sporting team has the name of their location and the location is used to refer to the team, " \
           "it is an entity which is an organisation, not a location"

    defn = "An entity is a person (PER), title, named organization (ORG), location (LOC), country (LOC) or nationality (MISC)."

    cot_exemplar_1 = """
    After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 .
    
    Answer:
    1. bowling | False | as it is an action
    2. Somerset | True | Somerset is used as a sporting team here, not a location hence it is an organisation (ORG)
    3. 83 | False | as it is a number 
    4. morning | False| as it represents a time of day, with no distinct and independant existence
    5. Grace Road | True | the game is played at Grace Road, hence it is a place or location (LOC)
    6. Leicestershire | True | is the name of a cricket team that is based in the town of Leicestershire, hence it is an organisation (ORG). 
    7. first innings | False | as it is an abstract concept of a phase in play of cricket
    8. England | True | as it is a place or location (LOC)
    9. Andy Caddick | True | as it is the name of a person. (PER) 
    """
    cot_exemplar_2 = """
    Their stay on top , though , may be short-lived as title rivals Essex , Derbyshire and Surrey all closed in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire .
    
    Answer:
    1. Their | False | as it is a possessive pronoun
    2. stay | False | as it is an action
    3. title rivals | False | as it is an abstract concept
    4. Essex | True |  Essex are title rivals is it a sporting team organisation not a location (ORG)
    5. Derbyshire | True |  Derbyshire are title rivals is it a sporting team organisation not a location (ORG)
    6. Surrey | True |  Surrey are title rivals is it a sporting team organisation not a location (ORG)
    7. victory | False | as it is an abstract concept
    8. Kent | True |  Kent lost to Nottinghamshire, it is a sporting team organisation not a location (ORG)
    9. Nottinghamshire | True |  Kent lost to Nottinghamshire, it is a sporting team organisation not a location (ORG)
    
    """

    cot_exemplar_3 = """
    But more money went into savings accounts , as savings held at 5.3 cents out of each dollar earned in both June and July .
    
    Answer:
    1. money | False | as it is not a named person, organization or location
    2. savings account | False | as it is not a person, organization or location
    3. 5.3 | False | as it is a number
    4. June | False | as it is a date
    5. July | False | as it is a date
    """

    cot_exemplars = [cot_exemplar_1, cot_exemplar_2, cot_exemplar_3]

    no_tf_exemplar_1 = """
        After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 .
        
        Answer:
        1. Somerset | Somerset is used as a sporting team here, not a location hence it is an organisation (ORG)
        2. Grace Road | the game is played at Grace Road, hence it is a place or location (LOC)
        3. Leicestershire | is the name of a cricket team that is based in the town of Leicestershire, hence it is an organisation (ORG). 
        4. England | as it is a place or location (LOC)
        5. Andy Caddick | as it is the name of a person. (PER) 
        """
    no_tf_exemplar_2 = """
        Their stay on top , though , may be short-lived as title rivals Essex , Derbyshire and Surrey all closed in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire .
        
        Answer:
        1. Essex | since Essex are title rivals is it a sporting team organisation not a location (ORG)
        2. Derbyshire | since Derbyshire are title rivals is it a sporting team organisation not a location (ORG)
        3. Surrey | since Surrey are title rivals is it a sporting team organisation not a location (ORG)
        4. Kent | since Kent lost to Nottinghamshire, it is a sporting team organisation not a location (ORG)
        5. Nottinghamshire | since Kent lost to Nottinghamshire, it is a sporting team organisation not a location (ORG)
        """

    no_tf_exemplar_3 = """
        But more money went into savings accounts , as savings held at 5.3 cents out of each dollar earned in both June and July .

        Answer:
        1. 

        """
    no_tf_exemplars = [no_tf_exemplar_1, no_tf_exemplar_2, no_tf_exemplar_3]

    tf_exemplar_1 = """
        After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 .

        Answer:
        1. bowling | False | None 
        2. Somerset | True | (ORG)
        3. 83 | False | None
        4. morning | False | None
        5. Grace Road | True | (LOC)
        6. Leicestershire | True | (ORG)
        7. first innings | False | None
        8. England | True | (LOC)
        9. Andy Caddick | True | (PER)
        """
    tf_exemplar_2 = """
        Their stay on top , though , may be short-lived as title rivals Essex , Derbyshire and Surrey all closed in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire .

        Answer:
        1. Their | False | None
        2. stay | False | None
        3. title rivals | False | None
        4. Essex | True | (ORG)
        5. Derbyshire | True | (ORG)
        6. Surrey | True | (ORG)
        7. victory | False | None
        8. Kent | True | (ORG)
        9. Nottinghamshire | True | (ORG)

        """

    tf_exemplar_3 = """
        But more money went into savings accounts , as savings held at 5.3 cents out of each dollar earned in both June and July .

        Answer:
        1. money | False | None
        2. savings account | False | None
        3. 5.3 | False | None
        4. June | False | None
        5. July | False | None

        """
    tf_exemplars = [tf_exemplar_1, tf_exemplar_2, tf_exemplar_3]

    exemplar_1 = """
        After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 .
        
        Answer:
        1. Somerset | (ORG)
        2. Grace Road | (LOC)
        3. Leicestershire | (ORG). 
        4. England | (LOC)
        5. Andy Caddick | (PER) 
    """
    exemplar_2 = """
        Their stay on top , though , may be short-lived as title rivals Essex , Derbyshire and Surrey all closed in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire .
        
        Answer:
        1. Essex | (ORG)
        2. Derbyshire | (ORG)
        3. Surrey | (ORG)
        4. Kent | (ORG)
        5. Nottinghamshire | (ORG)
    """

    exemplar_3 = """
    But more money went into savings accounts , as savings held at 5.3 cents out of each dollar earned in both June and July .

    Answer:
    1. 
    """
    exemplars = [exemplar_1, exemplar_2, exemplar_3]

    type_exemplar_1 = """
    After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 .

    Entity Phrase: Somerset
    Answer: Somerset is used as a sporting team here, not a location hence it is an organisation (ORG)
    
    Entity Phrase: England
    Answer: England is a country hence it is a location (LOC)
    
    Entity Phrase: Grace Road
    Answer: at Grace Road indicates this is a location or venue (LOC)    
    """

    type_exemplar_2 = """
    Their stay on top , though , may be short-lived as title rivals Essex , Derbyshire and Surrey all closed in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire .
    
    Entity Phrase: Essex
    Answer: As they are tital rivals, Essex is a sports team and not a location (ORG)
    
    Entity Phrase: Nottinghamshire
    Answer: As Nottinghamshire defeated Kent, this is a sports team not a location (ORG)
    """
    type_exemplars = [type_exemplar_1, type_exemplar_2]

    dispute_exemplar_1 = """
    After bowling Somerset out for 83 on the opening morning at Grace Road , Leicestershire extended their first innings by 94 runs before being bowled out for 296 with England discard Andy Caddick taking three for 83 .

    Entity Phrase: Somerset, Options: [(LOC), (ORG)]
    Answer: Somerset is used as a sporting team here, not a location hence it is an organisation (ORG)

    Entity Phrase: England, Options: [(LOC), (PER)]
    Answer: England is a country hence it is a location not a person (LOC)

    Entity Phrase: Grace Road, Options: [(LOC), (ORG)]
    Answer: at Grace Road indicates this is a location or venue (LOC)    
    """

    dispute_exemplar_2 = """
    Their stay on top , though , may be short-lived as title rivals Essex , Derbyshire and Surrey all closed in on victory while Kent made up for lost time in their rain-affected match against Nottinghamshire .

    Entity Phrase: Essex, Options: [(LOC), (ORG)]
    Answer: As they are tital rivals, Essex is a sports team and not a location (ORG)

    Entity Phrase: Nottinghamshire, Options: [(LOC), (ORG)]
    Answer: As Nottinghamshire defeated Kent, this is a sports team not a location (ORG)
    """
    dispute_exemplars = [dispute_exemplar_1, dispute_exemplar_2]


class GeniaConfig(Config):
    defn = "An entity is a protein (protein), group of proteins (protein), DNA, RNA, Cell Type (cell_type) or Cell Line (cell_line). " \
           "Abstract concepts, processes and adjectives are not entities"

    cot_exemplar_1 = """
        In primary T lymphocytes we show that CD28 ligation leads to the rapid intracellular formation of reactive oxygen intermediates ( ROIs ) which are required for CD28 -mediated activation of the NF-kappa B / CD28-responsive complex and IL-2 expression

        Answer:
        1. primary T lymphocytes | True | as they are a kind of cell type (cell_type) 
        2. CD28 | True | CD28 is one of the proteins expressed on T cells (protein)
        3. reactive oxygen intermediates ( ROIs ) | False | as they are not a protein, DNA, RNA, Cell Type or Cell Line
        4. NF-kappa B | True | Nuclear factor kappa B (NF-κB) is an ancient protein transcription factor (protein)
        5. CD28-responsive complex | True | it is a complex of the protein (protein)
        6. IL-2 | True | as it is a protein (protein)
        """
    cot_exemplar_2 = """
        The peri-kappa B site mediates human immunodeficiency virus type 2 enhancer activation in monocytes but not in T cells

        Answer:
        1. peri-kappa B site | True | as it is a  is a cis-acting element that is a DNA (DNA)
        2. human immunodeficiency virus type 2 enhancer | True | as it is a DNA (DNA)
        3. Activation | False | as it is a process
        4. monocytes | True | as they are a type of cell (cell_type)
        5. T cells | True | as they are a type of cell (cell_type)
        
        """
    cot_exemplars = [cot_exemplar_1, cot_exemplar_2]

    no_tf_exemplar_1 = """
        In primary T lymphocytes we show that CD28 ligation leads to the rapid intracellular formation of reactive oxygen intermediates ( ROIs ) which are required for CD28 -mediated activation of the NF-kappa B / CD28-responsive complex and IL-2 expression

        Answer:
        1. primary T lymphocytes | as they are a kind of cell type (cell_type) 
        2. CD28 | CD28 is one of the proteins expressed on T cells (protein)
        3. NF-kappa B | Nuclear factor kappa B (NF-κB) is an ancient protein transcription factor (protein)
        4. CD28-responsive complex | it is a complex of the protein (protein)
        5. IL-2 | as it is a protein (protein)
        
        """
    no_tf_exemplar_2 = """
        The peri-kappa B site mediates human immunodeficiency virus type 2 enhancer activation in monocytes but not in T cells

        Answer:
        1. peri-kappa B site | as it is a  is a cis-acting element that is a DNA (DNA)
        2. human immunodeficiency virus type 2 enhancer | as it is a DNA (DNA)
        3. monocytes | as they are a type of cell (cell_type)
        4. T cells | as they are a type of cell (cell_type)

        """
    no_tf_exemplars = [no_tf_exemplar_1, no_tf_exemplar_2]

    tf_exemplar_1 = """
        In primary T lymphocytes we show that CD28 ligation leads to the rapid intracellular formation of reactive oxygen intermediates ( ROIs ) which are required for CD28 -mediated activation of the NF-kappa B / CD28-responsive complex and IL-2 expression

        Answer:
        1. primary T lymphocytes | True | (cell_type) 
        2. CD28 | True | (protein)
        3. reactive oxygen intermediates ( ROIs ) | False | None
        4. NF-kappa B | True | (protein)
        5. CD28-responsive complex | True | (protein)
        6. IL-2 | True | (protein)
        """
    tf_exemplar_2 = """
        The peri-kappa B site mediates human immunodeficiency virus type 2 enhancer activation in monocytes but not in T cells

        Answer:
        1. peri-kappa B site | True | (DNA) 
        2. human immunodeficiency virus type 2 enhancer | True | (DNA)
        3. Activation | False | None
        4. monocytes | True | (cell_type)
        5. T cells | True | (cell_type)

        """
    tf_exemplars = [tf_exemplar_1, tf_exemplar_2]

    exemplar_1 = """
        In primary T lymphocytes we show that CD28 ligation leads to the rapid intracellular formation of reactive oxygen intermediates ( ROIs ) which are required for CD28 -mediated activation of the NF-kappa B / CD28-responsive complex and IL-2 expression

        Answer:
        1. primary T lymphocytes | (cell_type) 
        2. CD28 | (protein)
        3. NF-kappa B | (protein)
        4. CD28-responsive complex | (protein)
        5. IL-2 | (protein)
        """
    exemplar_2 = """
        The peri-kappa B site mediates human immunodeficiency virus type 2 enhancer activation in monocytes but not in T cells

        Answer:
        1. peri-kappa B site | (DNA) 
        2. human immunodeficiency virus type 2 enhancer | (DNA)
        3. monocytes | (cell_type)
        4. T cells | (cell_type)
        """
    exemplars = [exemplar_1, exemplar_2]


class TweetNERConfig(Config):
    defn = "An entity is a corporation (corporation), the name of a creative work made by humans (creative_work), event (event), names of real or fictional people (person), names of real or fictional locations (location), products (product) or groups that are not a corporation (group). Any set of words in the format {@words inside@} is a twitter handle and the entire phrase should be considered"

    #defn = "An entity is words that refer to names of people: (e.g. Virginia Wade). Include punctuation in the middle of names. Fictional people can be included as long as they are referenced by name (e.g. Harry Potter) (person). Names of locations: (e.g. France). Include punctuation in the middle of names. Fictional people can be included as long as they are referenced by name (e.g. Hogwarts) (location). Names of groups: (e.g. Nirvana, San Diego Padres), fictional groups can be included (group). Names of events: (e.g. Christmas, Super Bowl) (event). Name of products (e.g. iPhone), it must be something one can touch and it has to be refereed to by the official name (product). Creative work (e.g. Bohemian Rhapsody), should be created by a human (creative_work) or a corporation (e.g. Google) (coorporation)"

    cot_exemplar_1 = """
        # ReformPoliceNG {@Nigeria Police Force@} not # EndSARS the entire {@Nigeria Police Force@} is the problem not only the SARS operatives,i remember sometime in 2017,it was the {@UEFA Champions League@} season,around 10.30-11.00pm,myself and two of my friends watched the game a street opposite my street,on our way back ,

        Answer:
        1. {@Nigeria Police Force@} | True | as a police force is a corporate organization (corporation)
        2. SARS operatives | False | as it is not the name of a group or corporation
        3. {@UEFA Champions League@} | True | it is a football tournament refered to as an event not a corporation (event)
        """
    cot_exemplar_2 = """
        Soulja Boy Tell'em - Crank That ( Soulja Boy ) ( Official Music Video ) {{URL}} via {@YouTube@} {@Soulja Boy Draco 🌹@}
        
        Answer:
        1. Soulja Boy | True | this is the name of a rapper (person)
        2. Tell'em - Crank That | True | this is the name of a song created by Soulja Boy (creative_work)
        3. Official Music Video | True | as it is a work of art created by a human (creative_work)
        4. {{URL}} | False | as urls are not entities
        5. {@YouTube@} | True | as YouTube is a product (product)
        6. {@Soulja Boy Draco 🌹@} | True | as it is the twitter handle of a person (person)
        """

    cot_exemplar_3 = """
    The story of how working class folks , diehard fans from rival football clubs across England , all joined together in solidarity to defeat the greed of billionaires like {@The Boston Globe@} / {@Liverpool FC@} / {@Red Sox@} owner John Henry is pretty amazing . {{URL}}
    
    Answer:
    1. England | True | as it is a location (location)
    2. billionares | False | as it is not a named entity
    3. {@The Boston Globe@} | True | as it is a news organization it is a corporation (corporation)
    4. {@Liverpool FC@} | True | as it is a sports team it is a group not a corporation (group)
    5. {@Red Sox@} | True | as it is a sports team (group)
    6. John Henry | True | as it is a person (person)
    
    """

    cot_exemplars = [cot_exemplar_1, cot_exemplar_2, cot_exemplar_3]

    no_tf_exemplar_1 = """
            # ReformPoliceNG {@Nigeria Police Force@} not # EndSARS the entire {@Nigeria Police Force@} is the problem not only the SARS operatives,i remember sometime in 2017,it was the {@UEFA Champions League@} season,around 10.30-11.00pm,myself and two of my friends watched the game a street opposite my street,on our way back ,

        Answer:
        1. {@Nigeria Police Force@} | as a police force is a corporate organization (corporation)
        2. {@UEFA Champions League@} | it is a football tournament refered to as an event not a corporation (event)
        """
    no_tf_exemplar_2 = """
        Soulja Boy Tell'em - Crank That ( Soulja Boy ) ( Official Music Video ) {{URL}} via {@YouTube@} {@Soulja Boy Draco 🌹@}
        
        Answer:
        1. Soulja Boy | this is the name of a rapper (person)
        2. Tell'em - Crank That | this is the name of a song created by Soulja Boy (creative_work)
        3. Official Music Video | as it is a work of art created by a human (creative_work)
        4. {@YouTube@} | as YouTube is a product (product)
        5. {@Soulja Boy Draco 🌹@} | as it is the twitter handle of a person (person)

        """

    no_tf_exemplar_3 = """
    The story of how working class folks , diehard fans from rival football clubs across England , all joined together in solidarity to defeat the greed of billionaires like {@The Boston Globe@} / {@Liverpool FC@} / {@Red Sox@} owner John Henry is pretty amazing . {{URL}}
    
    Answer:
    1. England | as it is a location (location)
    2. {@The Boston Globe@} | as it is a news organization it is a corporation (corporation)
    3. {@Liverpool FC@} | as it is a sports team it is a group not a corporation (group)
    4. {@Red Sox@} | as it is a sports team (group)
    5. John Henry | as it is a person (person)
    """

    no_tf_exemplars = [no_tf_exemplar_1, no_tf_exemplar_2, no_tf_exemplar_3]

    tf_exemplar_1 = """
        # ReformPoliceNG {@Nigeria Police Force@} not # EndSARS the entire {@Nigeria Police Force@} is the problem not only the SARS operatives,i remember sometime in 2017,it was the {@UEFA Champions League@} season,around 10.30-11.00pm,myself and two of my friends watched the game a street opposite my street,on our way back ,

        Answer:
        1. {@Nigeria Police Force@} | True | (corporation)
        2. SARS operatives | False | None
        3. {@UEFA Champions League@} | True | (event)
        """
    tf_exemplar_2 = """
        Soulja Boy Tell'em - Crank That ( Soulja Boy ) ( Official Music Video ) {{URL}} via {@YouTube@} {@Soulja Boy Draco 🌹@}
        
        Answer:
        1. Soulja Boy | True | (person)
        2. Tell'em - Crank That | True | (creative_work)
        3. Official Music Video | True | (creative_work)
        4. {{URL}} | False | as urls are not entities
        5. {@YouTube@} | True | (product)
        6. {@Soulja Boy Draco 🌹@} | True | (person)

        """

    tf_exemplar_3 = """
        The story of how working class folks , diehard fans from rival football clubs across England , all joined together in solidarity to defeat the greed of billionaires like {@The Boston Globe@} / {@Liverpool FC@} / {@Red Sox@} owner John Henry is pretty amazing . {{URL}}
    
        Answer:
        1. England | True | (location)
        2. billionares | False | None
        3. {@The Boston Globe@} | True | (corporation)
        4. {@Liverpool FC@} | True | (group)
        5. {@Red Sox@} | True | (group)
        6. John Henry | True | (person)
    """
    tf_exemplars = [tf_exemplar_1, tf_exemplar_2, tf_exemplar_3]

    exemplar_1 = """
        # ReformPoliceNG {@Nigeria Police Force@} not # EndSARS the entire {@Nigeria Police Force@} is the problem not only the SARS operatives,i remember sometime in 2017,it was the {@UEFA Champions League@} season,around 10.30-11.00pm,myself and two of my friends watched the game a street opposite my street,on our way back ,

        Answer:
        1. {@Nigeria Police Force@} |  (corporation)
        2. {@UEFA Champions League@} | (event)

        """
    exemplar_2 = """
        Soulja Boy Tell'em - Crank That ( Soulja Boy ) ( Official Music Video ) {{URL}} via {@YouTube@} {@Soulja Boy Draco 🌹@}
        
        Answer:
        1. Soulja Boy | (person)
        2. Tell'em - Crank That | (creative_work)
        3. Official Music Video | (creative_work)
        4. {@YouTube@} | (product)
        5. {@Soulja Boy Draco 🌹@} | (person)
        """

    exemplar_3 = """
    The story of how working class folks , diehard fans from rival football clubs across England , all joined together in solidarity to defeat the greed of billionaires like {@The Boston Globe@} / {@Liverpool FC@} / {@Red Sox@} owner John Henry is pretty amazing . {{URL}}
    
    Answer:
    1. England | (location)
    2. {@The Boston Globe@} | (corporation)
    3. {@Liverpool FC@} | (group)
    4. {@Red Sox@} | (group)
    5. John Henry | (person)
    """
    exemplars = [exemplar_1, exemplar_2, exemplar_3]


class FabNERConfig(Config):
    defn = "An entity is a Material (MATE), Manufacturing Process (MANP), Application (APPL), Features (ENGF), Mechanical Property or Properties (MECHP), Characterization (PROC), Parameters (PROP), Machine/Equipment (MACEQ), Enabling Technology (ENAT), Concept/Principles (CONPRI), BioMedical entity (BIOP) or Manufacturing Standards (MANS)"

    cot_exemplar_1 = """
    However , parts fabricated using UAM often exhibit a reduction in strength levels when loaded normal to the welding interfaces ( Z-direction ) .
    
    Answer: 
    1. fabricated | True | as it is related to a concept of fabrication (CONPRI)
    2. UAM | True | it is used to fabricate something so it is a manufacturing process (MANP)
    3. exhibit | False | as it is not a concept or any other type of entity defined above
    4. reduction | True | as it is refers to a concept in manufacturing (CONPRI)
    5. strength | True | as it is a property of parts (PRO)
    6. normal | False | as it describes a direction and not an entity
    7. welding interfaces | True | as they are engineering features (ENGF)
    8. Z-direction | True | as it is a feature of engineering (ENGF)   
    """
    cot_exemplar_2 = """
    The 3D model is converted into a file format that is understood by AM machines .
    
    Answer:
    1. 3D model | True | as it is an application (APPL)
    2. converted | False | as it is not related to manufacturing 
    3. file | True | as it is standardized item that is understood by machines (MANS)
    4. AM machines | True | as it is a kind of machine (MACEQ)
    """

    cot_exemplar_3 = """
    Nowadays , these technologies have other names such as 3D printing or additive manufacturing , and so forth , but they all have the same origins from rapid prototyping 

    Answer: 
    1. technologies | True | as it is a general concept in manufacturing (CONPRI)
    2. 3D printing | True | as it is a process (MANP)
    3. additive manufacturing | True | as it is a process (MANP)
    4. rapid prototyping | True | as it is a technology that enables the development of processes (ENAT)
    """

    cot_exemplars = [cot_exemplar_1, cot_exemplar_2, cot_exemplar_3]

    no_tf_exemplar_1 = """
    However , parts fabricated using UAM often exhibit a reduction in strength levels when loaded normal to the welding interfaces ( Z-direction ) .
    
    Answer: 
    1. fabricated | as it is related to a concept of fabrication (CONPRI)
    2. UAM | it is used to fabricate something so it is a manufacturing process (MANP)
    3. reduction | as it is refers to a concept in manufacturing (CONPRI)
    4. strength | as it is a property of parts (PRO)
    5. welding interfaces | as they are engineering features (ENGF)
    6. Z-direction | as it is a feature of engineering (ENGF)   
        """
    no_tf_exemplar_2 = """
    The 3D model is converted into a file format that is understood by AM machines .
    
    Answer:
    1. 3D model | as it is an application (APPL)
    2. file | as it is standardized item that is understood by machines (MANS)
    3. AM machines | as it is a kind of machine (MACEQ)
    """

    no_tf_exemplar_3 = """
    Nowadays , these technologies have other names such as 3D printing or additive manufacturing , and so forth , but they all have the same origins from rapid prototyping 

    Answer: 
    1. technologies | as it is a general concept in manufacturing (CONPRI)
    2. 3D printing | as it is a process (MANP)
    3. additive manufacturing | as it is a process (MANP)
    4. rapid prototyping | as it is a technology that enables the development of processes (ENAT)
    """

    no_tf_exemplars = [no_tf_exemplar_1, no_tf_exemplar_2, no_tf_exemplar_3]

    tf_exemplar_1 = """
    However , parts fabricated using UAM often exhibit a reduction in strength levels when loaded normal to the welding interfaces ( Z-direction ) .
    
    Answer: 
    1. fabricated | True | (CONPRI)
    2. UAM | True | (MANP)
    3. exhibit | False | None
    4. reduction | True | (CONPRI)
    5. strength | True | (PRO)
    6. normal | False | None
    7. welding interfaces | True | (ENGF)
    8. Z-direction | True | (ENGF)   
        """
    tf_exemplar_2 = """
    The 3D model is converted into a file format that is understood by AM machines .
    
    Answer:
    1. 3D model | True | (APPL)
    2. converted | False | None
    3. file | True | (MANS)
    4. AM machines | True | (MACEQ)
        """

    tf_exemplar_3 = """
    Nowadays , these technologies have other names such as 3D printing or additive manufacturing , and so forth , but they all have the same origins from rapid prototyping 

    Answer: 
    1. technologies | True | (CONPRI)
    2. 3D printing | True | (MANP)
    3. additive manufacturing | True | (MANP)
    4. rapid prototyping | True | (ENAT)
    """

    tf_exemplars = [tf_exemplar_1, tf_exemplar_2, tf_exemplar_3]

    exemplar_1 = """
    However , parts fabricated using UAM often exhibit a reduction in strength levels when loaded normal to the welding interfaces ( Z-direction ) .
    
    Answer: 
    1. fabricated | (CONPRI)
    2. UAM | (MANP)
    3. reduction | (CONPRI)
    4. strength | (PRO)
    5. welding interfaces | (ENGF)
    6. Z-direction | (ENGF)   
        """
    exemplar_2 = """
    The 3D model is converted into a file format that is understood by AM machines .
    
    Answer:
    1. 3D model | (APPL)
    2. file | (MANS)
    3. AM machines | (MACEQ)
        """

    exemplar_3 = """
    Nowadays , these technologies have other names such as 3D printing or additive manufacturing , and so forth , but they all have the same origins from rapid prototyping 

    Answer: 
    1. technologies | (CONPRI)
    2. 3D printing | (MANP)
    3. additive manufacturing | (MANP)
    4. rapid prototyping | (ENAT)
    """
    exemplars = [exemplar_1, exemplar_2, exemplar_3]


class CrossNERPoliticsConfig(Config):
    defn = """
    An entity is a person or named character (person), organisation(organisation), politician(politician), political party (politicalparty), event(event), election(election), country(country), location that is not a country(location) or 
    other political entity (misc). Dates, times, abstract concepts, adjectives and verbs are not entities
    """

    cot_exemplar_1 = """
     Sitting as a Liberal Party of Canada Member of Parliament ( MP ) for Niagara Falls , she joined the Canadian Cabinet after the Liberals defeated the Progressive Conservative Party of Canada government of John Diefenbaker in the 1963 Canadian federal election .
    
    Answer:
    1. Liberal Party of Canada | True | as it is a political party (politicalparty)
    2. Parliament | True | as it is an organisation (organisation)
    3. Niagara Falls | True | as it is a location (misc)
    4. Canadian Cabinet | True | as it is a political entity (misc)
    5. Liberals | True | as it is a political group but not the party name (misc)
    6. Progressive Conservative Party of Canada | True | as it is a political party (politicalparty)
    7. government | False | as it is not actually an entity in this sentence
    8. John Diefenbaker | True | as it is a politician (politician)
    9. 1963 Canadian federal election | True | as it is an election (election)
    """

    cot_exemplar_2 = """
    The MRE took part to the consolidation of The Olive Tree as a joint electoral list both for the 2004 
    European Parliament election and the 2006 Italian general election , along with the Democrats of the Left and 
    Democracy is Freedom - The Daisy .

    Answer:
    1. MRE | True | as it is a political party (politicalparty)
    2. consolidation | False | as it is an action
    3. The Olive Tree | True | as it is a group or organization (organisation)
    4. 2004 European Parliament election | True | as it is an election (election)
    5. 2006 Italian general election | True | as it is an election (election)
    6. Democrats of the Left | True | as it is a political party (politicalparty)
    7. Democracy is Freedom - The Daisy | True | as it is a political party (politicalparty)
    """

    cot_exemplars = [cot_exemplar_1, cot_exemplar_2]

    no_tf_exemplar_1 = """
         Sitting as a Liberal Party of Canada Member of Parliament ( MP ) for Niagara Falls , she joined the Canadian Cabinet after the Liberals defeated the Progressive Conservative Party of Canada government of John Diefenbaker in the 1963 Canadian federal election .
        
        Answer:
        1. Liberal Party of Canada | as it is a political party (politicalparty)
        2. Parliament | as it is an organisation (organisation)
        3. Niagara Falls | as it is a location (misc)
        4. Canadian Cabinet | as it is a political entity (misc)
        5. Liberals | as it is a political group but not the party name (misc)
        6. Progressive Conservative Party of Canada | as it is a political party (politicalparty)
        7. John Diefenbaker | as it is a politician (politician)
        8. 1963 Canadian federal election | as it is an election (election)
        """

    no_tf_exemplar_2 = """
        The MRE took part to the consolidation of The Olive Tree as a joint electoral list both for the 2004 
        European Parliament election and the 2006 Italian general election , along with the Democrats of the Left and 
        Democracy is Freedom - The Daisy .

        Answer:
        1. MRE | as it is a political party (politicalparty)
        2. The Olive Tree | as it is a group or organization (organization)
        3. 2004 European Parliament election | as it is an election (election)
        4. 2006 Italian general election | as it is an election (election)
        5. Democrats of the Left | as it is a political party (politicalparty)
        6. Democracy is Freedom - The Daisy | as it is a political party (politicalparty)
        """

    no_tf_exemplars = [no_tf_exemplar_1, no_tf_exemplar_2]

    tf_exemplar_1 = """
        Sitting as a Liberal Party of Canada Member of Parliament ( MP ) for Niagara Falls , she joined the Canadian Cabinet after the Liberals defeated the Progressive Conservative Party of Canada government of John Diefenbaker in the 1963 Canadian federal election .
             
        1. Liberal Party of Canada | True | (politicalparty)
        2. Parliament | True | (organisation)
        3. Niagara Falls | True | (misc)
        4. Canadian Cabinet | True | (misc)
        5. Liberals | True | (misc)
        6. Progressive Conservative Party of Canada | True | (politicalparty)
        7. government | False | None
        8. John Diefenbaker | True | (politician)
        9. 1963 Canadian federal election | True | (election)
        """

    tf_exemplar_2 = """
        The MRE took part to the consolidation of The Olive Tree as a joint electoral list both for the 2004 
        European Parliament election and the 2006 Italian general election , along with the Democrats of the Left and 
        Democracy is Freedom - The Daisy .
    
        Answer:
        1. MRE | True | (politicalparty)
        2. consolidation | False | None
        3. The Olive Tree | True | (organisation)
        4. 2004 European Parliament election | True | (election)
        5. 2006 Italian general election | True | (election)
        6. Democrats of the Left | True | (politicalparty)
        7. Democracy is Freedom - The Daisy | True | (politicalparty)
        """

    tf_exemplars = [tf_exemplar_1, tf_exemplar_2]

    exemplar_1 = """
        Sitting as a Liberal Party of Canada Member of Parliament ( MP ) for Niagara Falls , she joined the Canadian Cabinet after the Liberals defeated the Progressive Conservative Party of Canada government of John Diefenbaker in the 1963 Canadian federal election .
             
        1. Liberal Party of Canada | (politicalparty)
        2. Parliament | (organisation)
        3. Niagara Falls | (misc)
        4. Canadian Cabinet | (misc)
        5. Liberals | (misc)
        6. Progressive Conservative Party of Canada | (politicalparty)
        7. John Diefenbaker | (politician)
        8. 1963 Canadian federal election | (election)
    """

    exemplar_2 = """
        The MRE took part to the consolidation of The Olive Tree as a joint electoral list both for the 2004 
        European Parliament election and the 2006 Italian general election , along with the Democrats of the Left and 
        Democracy is Freedom - The Daisy .

        Answer:
        1. MRE | (politicalparty)
        2. The Olive Tree | (organization)
        3. 2004 European Parliament election | (election)
        4. 2006 Italian general election | (election)
        5. Democrats of the Left | (politicalparty)
        6. Democracy is Freedom - The Daisy | (politicalparty)
    """

    exemplars = [exemplar_1, exemplar_2]


class CrossNERNaturalSciencesConfig(Config):
    defn = """
    An entity is a person or named character (person), university(university), scientist(scientist), organisation(organisation), country(country), location that is not a country(location), scientific discipline(discipline), enzyme(enzyme), 
    protein(protein), chemical compound(chemicalcompound), chemical element(chemicalelement), event(event), astronomical object(astronomicalobject), academic journal(academicjournal), award(award), or theory(theory). 
    Abstract scientific concepts can be entities if they have a name associated with them. If an entity does not fit the types above it is (misc)
    Dates, times, adjectives and verbs are not entities
    """

    cot_exemplar_1 = """
    He attended the U.S. Air Force Institute of Technology for a year , earning a bachelor 's degree in aeromechanics , and received his test pilot training at Edwards Air Force Base in California before his assignment as a test pilot at Wright-Patterson Air Force Base in Ohio .
    
    Answer:
    1. U.S. Air Force Institute of Technology | True | as he attended this institute is likely a university (university)
    2. bachelor 's degree | False | as it is not a university, award or any other entity type
    3. aeromechanics | True | as it is a scientific discipline (discipline)
    4. Edwards Air Force Base | True | as an Air Force Base is an organised unit (organisation)
    5. California | True | as in this case California refers to the state of California itself (location)
    6. Wright-Patterson Air Force Base | True | as an Air Force Base is an organisation (organisation)
    7. Ohio | True | as it is a state (location)
    """

    cot_exemplar_2 = """
    In addition , there would probably have been simple hydride s such as those now found in gas giants like Jupiter and Saturn , notably water vapor , methane , and ammonia .
    
    Answer:
    1. hydride | True | as it is a chemical (chemicalcompound)
    2. gas giants | True | as it is a category of astronomical object (misc)
    3. Jupiter | True | as it is a planet (astronomicalobject)
    4. water vapor | True | as it is a chemical (chemicalcompound)
    5. methane | True | as it is a chemical (chemicalcompound)
    6. ammonia | True | as it is a chemical (chemicalcompound)
    """
    cot_exemplars = [cot_exemplar_1, cot_exemplar_2]

    no_tf_exemplar_1 = """
    He attended the U.S. Air Force Institute of Technology for a year , earning a bachelor 's degree in aeromechanics , and received his test pilot training at Edwards Air Force Base in California before his assignment as a test pilot at Wright-Patterson Air Force Base in Ohio .
    
    Answer:
    1. U.S. Air Force Institute of Technology | as he attended this institute is likely a university (university)
    3. aeromechanics | as it is a scientific discipline (discipline)
    4. Edwards Air Force Base | as an Air Force Base is an organised unit (organisation)
    5. California | as in this case California refers to the state of California itself (location)
    6. Wright-Patterson Air Force Base | as an Air Force Base is an organisation (organisation)
    7. Ohio | as it is a state (location)
    """

    no_tf_exemplar_2 = """
    In addition , there would probably have been simple hydride s such as those now found in gas giants like Jupiter and Saturn , notably water vapor , methane , and ammonia .
    
    Answer:
    1. hydride | as it is a chemical (chemicalcompound)
    2. gas giants | as it is a category of astronomical object (misc)
    3. Jupiter | as it is a planet (astronomicalobject)
    4. water vapor | as it is a chemical (chemicalcompound)
    5. methane | as it is a chemical (chemicalcompound)
    6. ammonia | as it is a chemical (chemicalcompound)
    """

    no_tf_exemplars = [no_tf_exemplar_1, no_tf_exemplar_2]

    tf_exemplar_1 = """
    He attended the U.S. Air Force Institute of Technology for a year , earning a bachelor 's degree in aeromechanics , and received his test pilot training at Edwards Air Force Base in California before his assignment as a test pilot at Wright-Patterson Air Force Base in Ohio .
        
    1. U.S. Air Force Institute of Technology | True | (university)
    2. bachelor 's degree | False | None
    3. aeromechanics | True | (discipline)
    4. Edwards Air Force Base | True | (organisation)
    5. California | True | (location)
    6. Wright-Patterson Air Force Base | True | (organisation)
    7. Ohio | True | (location)
    """

    tf_exemplar_2 = """
    In addition , there would probably have been simple hydride s such as those now found in gas giants like Jupiter and Saturn , notably water vapor , methane , and ammonia .
    
    Answer:
    1. hydride | True | (chemicalcompound)
    2. gas giants | True | (misc)
    3. Jupiter | True | (astronomicalobject)
    4. water vapor | True | (chemicalcompound)
    5. methane | True | (chemicalcompound)
    6. ammonia | True | (chemicalcompound)
    """
    tf_exemplars = [tf_exemplar_1, tf_exemplar_2]

    exemplar_1 = """
    He attended the U.S. Air Force Institute of Technology for a year , earning a bachelor 's degree in aeromechanics , and received his test pilot training at Edwards Air Force Base in California before his assignment as a test pilot at Wright-Patterson Air Force Base in Ohio .
    
    Answer:
    1. U.S. Air Force Institute of Technology | (university)
    3. aeromechanics | (discipline)
    4. Edwards Air Force Base | (organisation)
    5. California | (location)
    6. Wright-Patterson Air Force Base | (organisation)
    7. Ohio | (location)
    """

    exemplar_2 = """
    In addition , there would probably have been simple hydride s such as those now found in gas giants like Jupiter and Saturn , notably water vapor , methane , and ammonia .
    
    Answer:
    1. hydride | (chemicalcompound)
    2. gas giants | (misc)
    3. Jupiter | (astronomicalobject)
    4. water vapor | (chemicalcompound)
    5. methane | (chemicalcompound)
    6. ammonia | (chemicalcompound)
    """
    exemplars = [exemplar_1, exemplar_2]


class CrossNERMusicConfig(Config):
    defn = """
    An entity is a person or named character (person), country(country), location that is not a country(location), organisation(organisation), 
    music genre(musicgenre), song(song), band(band), album(album), artist(musicalartist), 
    musical instrument(musicalinstrument), award(award), event(event) or musical entity (misc)
    Dates, times, adjectives and verbs are not entities. 
    """

    cot_exemplar_1 = """
    Artists from outside California who were associated with early alternative country included singer-songwriters 
    such as Lucinda Williams , Lyle Lovett and Steve Earle , the Nashville country rock band Jason and the Scorchers and the British post-punk band The Mekons .
    
    Answer:
    1. Artists | False | because it is not a specific artist, it is a common noun
    2. California | True | it is a state (location)
    3. alternative country | True | as it is a musical genre (musicgenre)
    4. Lucinda Williams | True | as this is an artist (musicalartist)
    5. Lyle Lovett | True | as this is an artist (musicalartist)
    6. Steve Earle | True | as this is an artist (musicalartist)
    7. Nashville country rock band | True | as it is an entity related to music (misc)
    8. Jason and the Scorchers | True | as it is the name of a band not a person (band)
    9. British | True | as it is a nationality (misc)
    10. post-punk | True | as it is a music genre (musicalgenre)
    11. The Mekons | True | as it is a band (band)
    """

    cot_exemplar_2 = """
    The film was nominated for the Academy Awards for Academy Award for Best Picture , as well as Academy Award for 
    Best Production Design ( Carroll Clark and Van Nest Polglase ) , Academy Award for 
    Best Original Song ( Irving Berlin for Cheek to Cheek ) , and Dance Direction ( Hermes Pan for Piccolino and Top Hat ) .
    
    Answer:
    1. Academy Awards | True | as it is an award (award)
    2. Academy Award for Best Picture | True | as it is the name of a specific award (award)
    3. Academy Award for Best Production Design | True | as it is the name of an award (award)
    4. Carroll Clark | True | it is a person but not a musician (person)
    5. Van Nest Polglase | True | person but not a musician (person)
    6. Academy Award for Best Original Song | True | an award (award)
    7. Irving Berlin | True | a person who recieved an award for a song is a musician or artist (musicalartist)
    8. Dance Direction | True | an award (award)
    9. Hermes Pan | True | a person who is not a musician (person)
    10. Piccolino | True | a dance performance name (misc)
    11. Top Hat | True | name of a dance (misc)
    
    """
    cot_exemplars = [cot_exemplar_1, cot_exemplar_2]

    no_tf_exemplar_1 = """
    Artists from outside California who were associated with early alternative country included singer-songwriters 
    such as Lucinda Williams , Lyle Lovett and Steve Earle , the Nashville country rock band Jason and the Scorchers and the British post-punk band The Mekons .
    
    Answer:
    1. California | it is a state (location)
    2. alternative country | as it is a musical genre (musicgenre)
    3. Lucinda Williams | as this is an artist (musicalartist)
    4. Lyle Lovett | as this is an artist (musicalartist)
    5. Steve Earle | as this is an artist (musicalartist)
    6. Nashville country rock band | as it is an entity related to music (misc)
    7. Jason and the Scorchers | as it is the name of a band not a person (band)
    8. British | as it is a nationality (misc)
    9. post-punk | as it is a music genre (musicalgenre)
    10. The Mekons | as it is a band (band)
    """

    no_tf_exemplar_2 = """
    The film was nominated for the Academy Awards for Academy Award for Best Picture , as well as Academy Award for 
    Best Production Design ( Carroll Clark and Van Nest Polglase ) , Academy Award for 
    Best Original Song ( Irving Berlin for Cheek to Cheek ) , and Dance Direction ( Hermes Pan for Piccolino and Top Hat ) .
    
    Answer:
    1. Academy Awards | as it is an award (award)
    2. Academy Award for Best Picture |  as it is the name of a specific award (award)
    3. Academy Award for Best Production Design | as it is the name of an award (award)
    4. Carroll Clark | it is a person but not a musician (person)
    5. Van Nest Polglase | person but not a musician (person)
    6. Academy Award for Best Original Song | an award (award)
    7. Irving Berlin | a person who recieved an award for a song is a musician or artist (musicalartist)
    8. Dance Direction | an award (award)
    9. Hermes Pan | a person who is not a musician (person)
    10. Piccolino | a dance performance name (misc)
    11. Top Hat | name of a dance (misc)
    """

    no_tf_exemplars = [no_tf_exemplar_1, no_tf_exemplar_2]

    tf_exemplar_1 = """
    Artists from outside California who were associated with early alternative country included singer-songwriters 
    such as Lucinda Williams , Lyle Lovett and Steve Earle , the Nashville country rock band Jason and the Scorchers and the British post-punk band The Mekons .
    
    Answer:
    1. Artists | False | None
    2. California | True | (location)
    3. alternative country | True | (musicgenre)
    4. Lucinda Williams | True | (musicalartist)
    5. Lyle Lovett | True | (musicalartist)
    6. Steve Earle | True | (musicalartist)
    7. Nashville country rock band | True | (misc)
    8. Jason and the Scorchers | True | (band)
    9. British | True | (misc)
    10. post-punk | True | (musicalgenre)
    11. The Mekons | True | (band)
    """

    tf_exemplar_2 = """
    The film was nominated for the Academy Awards for Academy Award for Best Picture , as well as Academy Award for 
    Best Production Design ( Carroll Clark and Van Nest Polglase ) , Academy Award for 
    Best Original Song ( Irving Berlin for Cheek to Cheek ) , and Dance Direction ( Hermes Pan for Piccolino and Top Hat ) .
    
    Answer:
    1. Academy Awards | True | (award)
    2. Academy Award for Best Picture | True | (award)
    3. Academy Award for Best Production Design | True | (award)
    4. Carroll Clark | True | (person)
    5. Van Nest Polglase | True | (person)
    6. Academy Award for Best Original Song | True | (award)
    7. Irving Berlin | True | (musicalartist)
    8. Dance Direction | True | (award)
    9. Hermes Pan | True | (person)
    10. Piccolino | True | (misc)
    11. Top Hat | True | (misc)

    """
    tf_exemplars = [tf_exemplar_1, tf_exemplar_2]

    exemplar_1 = """
    Artists from outside California who were associated with early alternative country included singer-songwriters 
    such as Lucinda Williams , Lyle Lovett and Steve Earle , the Nashville country rock band Jason and the Scorchers and the British post-punk band The Mekons .
    
    Answer:
    1. California | (location)
    2. alternative country | (musicgenre)
    3. Lucinda Williams | (musicalartist)
    4. Lyle Lovett | (musicalartist)
    5. Steve Earle | (musicalartist)
    6. Nashville country rock band | (misc)
    7. Jason and the Scorchers | (band)
    8. British | (misc)
    9. post-punk | (musicalgenre)
    10. The Mekons | (band)
    """

    exemplar_2 = """
    As a group , the Spice Girls have received a number of notable awards including five Brit Awards , 
    three American Music Awards , three MTV Europe Music Awards , one MTV Video Music Award and three World Music Awards.

    Answer:
    1. Academy Awards | (award)
    2. Academy Award for Best Picture | (award)
    3. Academy Award for Best Production Design | (award)
    4. Carroll Clark | (person)
    5. Van Nest Polglase | (person)
    6. Academy Award for Best Original Song | (award)
    7. Irving Berlin | (musicalartist)
    8. Dance Direction | (award)
    9. Hermes Pan | (person)
    10. Piccolino | (misc)
    11. Top Hat | (misc)
    """
    exemplars = [exemplar_1, exemplar_2]


class CrossNERLiteratureConfig(Config):
    defn = """
    An entity is a person or named character (person), country(country), location that is not a country(location), organisation(organisation), book(book), writer(writer), poem(poem), magazine(magazine), 
    award(award), event(event), country(country), literary genre (literarygenre), nationality(misc) or other enitity in literature (misc). 
    Dates, times, adjectives and verbs are not entities. 
    """

    cot_exemplar_1 = """
     The poor conditions of the hospital in Lambaréné were also famously criticized by Nigerian professor and 
     novelist Chinua Achebe in his essay on Joseph Conrad ' s novel Heart of Darkness : 
     In a comment which has often been quoted Schweitzer says : ' The African is indeed my brother but my junior brother .
    
    Answer:
    1. hospital | False | as it is a building type not a named location
    2. Lambaréné | True | as it is a location in which the hospital is located (location)
    3. Nigerian | True | as it is a nationality (misc)
    4. professor | False | as it is not an entity type as defined by the list
    5. Chinua Achebe | True | as this is a write who is a novelist (writer)
    6. Joseph Conrad | True | as this is a writer who wrote a novel called the Heart of Darkness (writer)
    7. novel | True | as this is a genre or type of literature (literarygenre)
    8. Heart of Darkness | True | as this is the name of a book (book)
    9. Schweitzer | True | as this is a person, not a writer (person)
    10. African | True | as this is like a nationality (misc)
    """

    cot_exemplar_2 = """
    During this period , he covered Timothy Leary and Richard Alpert ' s Millbrook , New York -based 
    Castalia Foundation at the instigation of Alan Watts in The Realist , cultivated important friendships with 
    William S. Burroughs and Allen Ginsberg , and lectured at the Free University of New York on ' Anarchist and Synergetic Politics ' in 1965 .
    
    Answer: 
    1. period | False | as this indicates a time period
    2. Timothy Leary | True | as this is a person who has not written a literary work (person)
    3. Richard Alpert | True | as this person hasn't written a literary work(person)
    4. Millbrook | True | as it is a location inside New York (location)
    5. New York | True | as it is a state (location)
    6. Castalia Foundation | True | as it is an organisation (organisation)
    7. instigation | False | as it is an action 
    8. Alan Watts | True | as it is a person who has written in a magazine (writer)
    9. The Realist | True | the name of a magazine (magazine)
    10. William S. Burroughs | True | the name of famous author  (writer)
    11. Allen Gibsberg | True | a person who has written literary works (writer)
    12. Free University of New York | True | a university is an organisation (organisation)
    13. Anarchist and Synergetic Politics | True | some formal academic work (misc)
    """

    cot_exemplars = [cot_exemplar_1, cot_exemplar_2]

    no_tf_exemplar_1 = """
     The poor conditions of the hospital in Lambaréné were also famously criticized by Nigerian professor and 
     novelist Chinua Achebe in his essay on Joseph Conrad ' s novel Heart of Darkness : 
     In a comment which has often been quoted Schweitzer says : ' The African is indeed my brother but my junior brother .
    
    Answer:
    1. Lambaréné | as it is a location in which the hospital is located (location)
    2. Nigerian | as it is a nationality (misc)
    3. Chinua Achebe | as this is a write who is a novelist (writer)
    4. Joseph Conrad | as this is a writer who wrote a novel called the Heart of Darkness (writer)
    5. novel | as this is a genre or type of literature (literarygenre)
    6. Heart of Darkness | as this is the name of a book (book)
    7. Schweitzer | as this is a person, not a writer (person)
    8. African | as this is like a nationality (misc)
    """

    no_tf_exemplar_2 = """
    During this period , he covered Timothy Leary and Richard Alpert ' s Millbrook , New York -based 
    Castalia Foundation at the instigation of Alan Watts in The Realist , cultivated important friendships with 
    William S. Burroughs and Allen Ginsberg , and lectured at the Free University of New York on ' Anarchist and Synergetic Politics ' in 1965 .
    
    Answer: 
    1. Timothy Leary | as this is a person who has not written a literary work (person)
    2. Richard Alpert | as this person hasn't written a literary work(person)
    3. Millbrook | as it is a location inside New York (location)
    4. New York | as it is a state (location)
    5. Castalia Foundation | as it is an organisation (organisation)
    6. Alan Watts | as it is a person who has written in a magazine (writer)
    7. The Realist | the name of a magazine (magazine)
    8. William S. Burroughs | the name of famous author  (writer)
    9. Allen Gibsberg | a person who has written literary works (writer)
    10. Free University of New York | a university is an organisation (organisation)
    11. Anarchist and Synergetic Politics | some formal academic work (misc)
    """

    no_tf_exemplars = [no_tf_exemplar_1, no_tf_exemplar_2]

    tf_exemplar_1 = """
     The poor conditions of the hospital in Lambaréné were also famously criticized by Nigerian professor and 
     novelist Chinua Achebe in his essay on Joseph Conrad ' s novel Heart of Darkness : 
     In a comment which has often been quoted Schweitzer says : ' The African is indeed my brother but my junior brother .
    
    Answer:
    1. hospital | False | None
    2. Lambaréné | True | (location)
    3. Nigerian | True | (misc)
    4. professor | False | None
    5. Chinua Achebe | True | (writer)
    6. Joseph Conrad | True | (writer)
    7. novel | True | (literarygenre)
    8. Heart of Darkness | True | (book)
    9. Schweitzer | True | (person)
    10. African | True | (misc)
    """

    tf_exemplar_2 = """
    During this period , he covered Timothy Leary and Richard Alpert ' s Millbrook , New York -based 
    Castalia Foundation at the instigation of Alan Watts in The Realist , cultivated important friendships with 
    William S. Burroughs and Allen Ginsberg , and lectured at the Free University of New York on ' Anarchist and Synergetic Politics ' in 1965 .
    
    Answer: 
    1. period | False | None
    2. Timothy Leary | True | (person)
    3. Richard Alpert | True | (person)
    4. Millbrook | True | (location)
    5. New York | True | (location)
    6. Castalia Foundation | True | (organisation)
    7. instigation | False | None
    8. Alan Watts | True | (writer)
    9. The Realist | True | (magazine)
    10. William S. Burroughs | True | (writer)
    11. Allen Gibsberg | True | (writer)
    12. Free University of New York | True | (organisation)
    13. Anarchist and Synergetic Politics | True | (misc)
    """

    tf_exemplars = [tf_exemplar_1, tf_exemplar_2]

    exemplar_1 = """
     The poor conditions of the hospital in Lambaréné were also famously criticized by Nigerian professor and 
     novelist Chinua Achebe in his essay on Joseph Conrad ' s novel Heart of Darkness : 
     In a comment which has often been quoted Schweitzer says : ' The African is indeed my brother but my junior brother .
    
    Answer:
    1. Lambaréné | (location)
    2. Nigerian | (misc)
    3. Chinua Achebe | (writer)
    4. Joseph Conrad | (writer)
    5. novel | (literarygenre)
    6. Heart of Darkness | (book)
    7. Schweitzer | (person)
    8. African | (misc)
    """

    exemplar_2 = """
    During this period , he covered Timothy Leary and Richard Alpert ' s Millbrook , New York -based 
    Castalia Foundation at the instigation of Alan Watts in The Realist , cultivated important friendships with 
    William S. Burroughs and Allen Ginsberg , and lectured at the Free University of New York on ' Anarchist and Synergetic Politics ' in 1965 .
    
    Answer: 
    1. Timothy Leary |  (person)
    2. Richard Alpert | (person)
    3. Millbrook | (location)
    4. New York | (location)
    5. Castalia Foundation | (organisation)
    6. Alan Watts | (writer)
    7. The Realist | (magazine)
    8. William S. Burroughs | (writer)
    9. Allen Gibsberg | (writer)
    10. Free University of New York | (organisation)
    11. Anarchist and Synergetic Politics | (misc)
    """

    exemplars = [exemplar_1, exemplar_2]


class CrossNERAIConfig(Config):
    defn = """
    An entity is a person or named character (person), country(country), location that is not a country(location), organisation(organisation), field of Artificial Intelligence, Computer Science or Engineering (field), 
    task in artificial intelligence(task), product(product), algorithm(algorithm), 
    metric in artificial intelligence(metrics), university or academic institution(university), 
    researcher(researcher), AI conference (conference), programming language (programlang) 
    or other entity related to AI research (misc). If an entity is an AI research organization that accepts publishing of papers e.g. Association for the Advancement of Artificial Intelligence it is a conference (conference). Academic degrees like PhD or diplomas are entities (misc)
    Dates, times, adjectives and verbs are not entities. 
    """

    cot_exemplar_1 = """
    Since the Google acquisition , the company has notched up a number of significant achievements , 
    perhaps the most notable being the creation of AlphaGo , a program that defeated world champion Lee Sedol at the complex game of Go 
    
    Answer:
    1. Google | True | as it is a company or organisation (organisation)
    2. creation | False | as it is an action
    3. AlphaGo | True | as it is a program or product using AI (product)
    4. Lee Sedol | True | as this is a person but not a researcher (person)
    5. Go | True | as this is a game that the AI played and is an entitty (misc)
    """

    cot_exemplar_2 = """
    In machine learning , support-vector machines ( SVMs , also support-vector networks ) are 
    supervised learning models with learning algorithm s that analyze data used for classification and regression analysis .
    
    Answer:
    1. machine learning | True | as it is a field of AI (field)
    2. support-vector machines | True | an algorithm in AI (algorithm)
    3. SVMs | True | the abbreviation of support-vector machines which is an algorithm (algorithm)
    4. supervised learning | True | a subfield of AI (field)
    5. learning algorithms | False | as it is not a specific algorithm or task
    6. classification | True | as it is a specific task in machine learning or AI (task)
    7. regression analysis | True | as it is a specific task in machine learning or AI (task)

    """

    cot_exemplar_3 ="""
    In Cryo Electron Tomography , where the limited number of projections are acquired due to the hardware limitations 
    and to avoid the biological specimen damage , it can be used along with compressive sensing techniques or 
    regularization functions ( e.g. Huber loss ) to improve the reconstruction for better interpretation 
    
    Answer: 
    1. Cryo Electron Tomography | True | as it is a field or method related to AI research but is not an AI algorithm itself (misc)
    2. hardware limitations | False | as it is an abstract concept, not an entity
    3. biological specimen damage | False | as it's an abstract concept, not an entity
    4. compressive sensing techniques | True | as they are techniques related to AI research (algorithm)
    5. regularization functions | False | as they are a specific components of ML algorithms (algorithm)
    6. Huber loss | True | as it is a specific loss function in AI used to benchmark performance (metrics)    
    """


    cot_exemplars = [cot_exemplar_1, cot_exemplar_2]

    no_tf_exemplar_1 = """
    Since the Google acquisition , the company has notched up a number of significant achievements , 
    perhaps the most notable being the creation of AlphaGo , a program that defeated world champion Lee Sedol at the complex game of Go 
    
    Answer:
    1. Google | as it is a company or organisation (organisation)
    2. AlphaGo | as it is a program or product using AI (product)
    3. Lee Sedol | as this is a person but not a researcher (person)
    4. Go | as this is a game that the AI played and is an entitty (misc)

    """

    no_tf_exemplar_2 = """
    In machine learning , support-vector machines ( SVMs , also support-vector networks ) are 
    supervised learning models with learning algorithm s that analyze data used for classification and regression analysis .
    
    Answer:
    1. machine learning | as it is a field of AI (field)
    2. support-vector machines | an algorithm in AI (algorithm)
    3. SVMs | the abbreviation of support-vector machines which is an algorithm (algorithm)
    4. supervised learning | a subfield of AI (field)
    5. classification | as it is a specific task in machine learning or AI (task)
    6. regression analysis | as it is a specific task in machine learning or AI (task)
    """

    no_tf_exemplars = [no_tf_exemplar_1, no_tf_exemplar_2]

    tf_exemplar_1 = """
    Since the Google acquisition , the company has notched up a number of significant achievements , 
    perhaps the most notable being the creation of AlphaGo , a program that defeated world champion Lee Sedol at the complex game of Go 
    
    Answer:
    1. Google | True | (organisation)
    2. creation | False | None
    3. AlphaGo | True | (product)
    4. Lee Sedol | True | (person)
    5. Go | True | (misc)
        """

    tf_exemplar_2 = """
    In machine learning , support-vector machines ( SVMs , also support-vector networks ) are 
    supervised learning models with learning algorithm s that analyze data used for classification and regression analysis .
    
    Answer:
    1. machine learning | True | (field)
    2. support-vector machines | True | (algorithm)
    3. SVMs | True | (algorithm)
    4. supervised learning | True |(field)
    5. learning algorithms | False | None
    6. classification | True | (task)
    7. regression analysis | True | (task)
        """
    tf_exemplars = [tf_exemplar_1, tf_exemplar_2]

    exemplar_1 = """
    Since the Google acquisition , the company has notched up a number of significant achievements , 
    perhaps the most notable being the creation of AlphaGo , a program that defeated world champion Lee Sedol at the complex game of Go 
    
    Answer:
    1. Google | (organisation)
    2. AlphaGo | (product)
    3. Lee Sedol | (person)
    4. Go | (misc)

    """

    exemplar_2 = """
    In machine learning , support-vector machines ( SVMs , also support-vector networks ) are 
    supervised learning models with learning algorithm s that analyze data used for classification and regression analysis .
    
    Answer:
    1. machine learning | (field)
    2. support-vector machines | (algorithm)
    3. SVMs | (algorithm)
    4. supervised learning | (field)
    5. classification | (task)
    6. regression analysis | (task)
    """
    exemplars = [exemplar_1, exemplar_2]


class FewNERDConfig(Config):
    person = "person"
    art = "piece of art"
    miscellaneous = "product, language, living thing, currency, god or scientific concept in astronomy, biology etc. "
    locations = "locations with the following types of locations - Country(location-GPE), Province/State(location-GPE), " \
                "City(location-GPE) or District(location-GPE) or nationality (location-GPE), rivers, lakes, oceans, bays, locations of Harbours or other bodies of water " \
                "(location-bodiesofwater) Islands (location-island), Mountains (location-mountain), " \
                "park (location-park), road, railway, tunnel or highway (location-road/railway/highway/transit) or other location (location-other)"
    organizations = "organizations with the following types of organizations - Companies (organization-company), university or institute of education (organization-education), government related (organization-government), media (organization-media), political party (organization-politicalparty), religion (organization-religion), sports team (organization-sportsteam), sporting league (organization-sportsleague), performance group (organization-showorg) or other organization (organization-other) e.g. libraries are (organization-other)"
    buildings = "the names of buildings"
    events = "events"
    clearly_not = "Dates, times, abstract concepts and adjectives"
    train_group = f"{person}, {art}, {miscellaneous}."
    dev_group = f"{buildings} and {events}"
    test_group = f"{locations} and {organizations}"


class FewNERDINTRATestConfig(FewNERDConfig):
    defn = f"""
        Entities are  {FewNERDConfig.test_group}. Entities are not a {FewNERDConfig.train_group}, 
        {FewNERDConfig.dev_group} are also not entities. {FewNERDConfig.clearly_not} are not entities. Military organizations like a Regiment are (organization-other). If a country is mentioned as a competitor in a sport it is a (organization-sportsteam), if the government of the country is mentioned it is (GPE) e.g. in U.S government the entity is U.S (location-GPE)
        """

    cot_exemplar_1 = """
        Paragraph: The 1873 Invercargill by-election was a by-election during the 5th New Zealand Parliament in the Southland electorate of Invercargill .
         
        Answer:
        1. 1873 Invercargill by-election | False | as it is an event
        2. New Zealand | True | as it is a country (location-GPE)
        3. Southland | True | as it is an electorate in New Zealand (location-GPE)
        4. Invercargill | True | as it is a city (location-GPE)
         """

    cot_exemplar_2 = """
        Paragraph: In 1921 he was appointed Ophthalmic Surgeon at Cardiff Royal Infirmary and United Cardiff Hospitals and served there for 37 years 
    
        Answer:
        1. 1921 | False | as it is a date
        2. Ophthalmic Surgeon | False | as it is a job title, not an organization or location
        3. Cardiff Royal Infirmary and United Cardiff Hospitals | True | as it is a single organization (organization-other)
        4. 37 years | False | as it is a duration, not an organization or location
         """

    cot_exemplar_3 = """
            Paragraph: In 1921 he was appointed Ophthalmic Surgeon at Cardiff Royal Infirmary and United Cardiff Hospitals and served there for 37 years 

            Answer:
            1. 1921 | False | as it is a date
            2. Ophthalmic Surgeon | False | as it is a job title, not an organization or location
            3. Cardiff Royal Infirmary and United Cardiff Hospitals | True | as it is a single organization (organization-other)
            4. 37 years | False | as it is a duration, not an organization or location
             """

    no_tf_exemplar_1 = """
          Answer:
          1. Association of Artificial Intelligence | as this is an organisation (organisation-education)
         """

    no_tf_exemplar_2 = """
         Answer:
         1. England | as it is a location (location-GPE)
         """

    tf_exemplar_1 = """
          Answer:
          1. Albert Einstein | False | None
          2. USD | False | (currency)
          3. purchase | False | None
          4. Eiffel tower | False | (building)
          5. Association of Artificial Intelligence | True | (organisation-education)
                     """

    tf_exemplar_2 =  """
                     Answer:
                     1. England | True | (location-GPE)
                     2. festival | False | None
                     3. Grand Jubilee | False | (event)
                     4. 1982 | False | None
                     5. Attila the Hun | False | (person)
                     6. The Starry Night | False | (art)
                     """

    exemplar_1 = """
         Answer:
         1. Association of Artificial Intelligence | (organisation-education)
         """

    exemplar_2 = """
         Answer: 
         1. England | (location-GPE)
         """
    cot_exemplars = [cot_exemplar_1, cot_exemplar_2, cot_exemplar_3]
    no_tf_exemplars = [no_tf_exemplar_1, no_tf_exemplar_2]
    exemplars = [exemplar_1, exemplar_2]
    tf_exemplars = [tf_exemplar_1, tf_exemplar_2]
