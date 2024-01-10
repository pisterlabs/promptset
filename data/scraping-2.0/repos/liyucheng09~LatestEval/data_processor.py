from datasets import load_dataset
import re
from nltk.tokenize import sent_tokenize
import os
import openai
import time
from collections import Counter
import markdown
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
from datetime import datetime

from tqdm import tqdm
from dataclasses import dataclass

from typing import Union, List

import torch
from transformers import LlamaForCausalLM, LlamaTokenizerFast, AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GenerationConfig

from glob import glob
import argparse

from concurrent.futures import ThreadPoolExecutor

@dataclass
class Doc:
    text: str
    source: str
    entry_id: str
    
    original_passage: str = None
    passage_to_input: str = None
    
    original_sentences: list = None

    queries: list[str] = None
    answers: list[str] = None
    answers_sentences: list[(str, str)] = None
    masked_answers_sentences: list[(str, str, int)] = None
    meta_data: dict = None

    sections: list[str] = None

    def __repr__(self):
        return f"Doc({self.source}, {self.entry_id})"

class BaseProcessor:
    """
    Base class for LatestEval processors.

    What is a processor?
    1) It take a Doc object as input, and output a list of QA pairs.
    2) It can be dataset specific, or agnostic

    Steps:
    1) It goes through the given document, and identify the answer sentences with answer types.
    2) It generate the answer and query from the answer sentence.
    3) It prepare the masked passage for the answer and query.
    4) It generate the QA pairs.
    """

    def __init__(self, source, time_stamp):
        # the source of the data
        self.source = source
        # the time stamp of the data
        self.time_stamp = time_stamp

        self.tokenizer_for_token_counting = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
    
    def _num_tokens(self, text):
        return len(self.tokenizer_for_token_counting(text)['input_ids'])

    def _annotate_sentences(self, sentences):
        """
        Check sentences to see if there is any answer in it.
        If there is, annotate the sentence with the answer type.

        answer_types: List[Union['terminology', 'summary', 'purpose', 'example', 'future', 'none']]
        """
        raise NotImplementedError
    
    def queries(self, answer_sent, answer_type):
        """
        Generate queries from the answer.
        """
        raise NotImplementedError

    def answers(self):
        raise NotImplementedError
    
    def qa_pairs(self, doc: Doc):
        """
        Generate QA pairs from the query, answer and context.
        """
        raise NotImplementedError
    
    def mask_passage(self):
        """
        Mask the passage with the answer and query.
        """
        raise NotImplementedError
    
    def gpt(self, prompt, num_retry = 5, model = 'gpt-4-1106-preview'):
        assert model in ['gpt-3.5-turbo', 'gpt-4-1106-preview'], "model should be either gpt-3.5-turbo or gpt-4-1106-preview"
        openai_key = os.environ.get("OPENAI_API_KEY")
        assert openai_key is not None, "Please set OPENAI_API_KEY in your environment variables."
        for _ in range(num_retry):
            try:
                r = openai.ChatCompletion.create(
                    model = model,
                    messages = [
                        {"role": "user", "content": prompt},
                    ],
                    # response_format = { "type": "json_object" },
                )
                break
            except Exception as e:
                print(e)
                time.sleep(1)
        
        return r.choices[0]['message']['content']
    
    def _load_hf_models(self, prompt, model_name):
        # prepare models
        if 'GPTQ' in model_name:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            # only llama-30b use gptq
            model = AutoGPTQForCausalLM.from_quantized(model_name, device = 'cuda:0', use_safetensors = True, disable_exllama=True if '30b' in model_name else False, trust_remote_code=True)
        elif 'llama' in model_name.lower():
            model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto')
        elif 'opt' in model_name.lower():
            model = OPTForCausalLM.from_pretrained(model_name, device_map='auto')
        elif 'gpt2' == model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        self.models = {model_name: (model, tokenizer)}
        
        # prepare generation config
        self.generation_config = GenerationConfig(
            temperature=1.0,
            top_k=50,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def _hf_lm_generate(self, prompt, model_name):
        model, tokenizer = self.models[model_name]
        # generate answer sequentially
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            outputs = model.generate(input_ids, generation_config=self.generation_config, return_dict_in_generate=True, max_new_tokens=500)
        s = outputs.sequences[0]
        prompt_len = input_ids.shape[1]
        output = tokenizer.decode(s[prompt_len:])
        return output
    
    def _is_openai_models(self, model_name):
        return model_name in ['gpt-3.5-turbo', 'gpt-4-1106-preview']
    
    def llm(self, *args, **kwargs):
        """
        Wrapper for LLM.

        The current default model is gpt-3.5-turbo.
        """
        model = kwargs.get('model', None)
        if model is None:
            model = 'gpt-3.5-turbo'
        
        if self._is_openai_models(model):
            return self.gpt(*args, **kwargs)
        else:
            # assume it is a huggingface model
            if model not in self.models:
                self._load_hf_models(*args, **kwargs)
            return self._hf_lm_generate(*args, **kwargs)

class KeyWordsProcessor(BaseProcessor):
    """
    Key words-based LatestEval processors.

    1) Using key words matching to identify answer types.
    2) Given the answer type, generate the answer and query.
    3) Given the answer and query, prepare the masked passage.
    """

    def __init__(self, source, time_stamp):
        super().__init__(source, time_stamp)

        self.answer_keywords = {
            'summary': [
                'in summary',
                'to summarize',
                'main contribution',
                'in conclusion',
                'major contribution',
                'in short',
                'td;dr',
            ],
            'purpose': [
                'because',
                'in order to',
                'allow us to',
                'lead to',
                'so that',
                'enable',
                'allow',
            ],
            'example': [
                'for example',
                'e.g.,',
                'for instance',
            ],
            'future': [
                'future',
                'upcoming feature',
                'forecast',
            ],
        }

        self.query_templates = {
            "terminology": [
                "What is {term}?",
                "Can you explain what {term} means?", 
                "What does the author mean by {term}?",
                "How does {term1} differ from {term2}?"
            ],
            
            "summary": [
                "Can you summarize this {section}?",
                "What are the key points made in this {section}?",
                "What is the main finding in this {section}?" 
            ],
            
            "purpose": [
                "What is the purpose of the {object}?",
                "Why did the author propose the {object}?",
                "What problem does the {object} aim to solve?"
            ],

            "example": [
                "Can you provide examples of {object}?",
                "What are some examples of {concept} in the passage?",
                "Could you illustrate {concept} with examples?"
            ],

            "future": [
                "What future work does the author suggest?",
                "How might {concept} evolve in the future according to the passage?", 
                "What predictions does the author make about {concept}?"
            ]
        }
    
    def answers(self):
        """
        Identify key points from fragments via copying the original text.

        Args:
            fragment: the answer fragment
        """
        for doc in self.docs:
            if doc.answers_sentences is None:
                doc.answers_sentences = []
            for sent_index, sent in enumerate(doc.original_sentences):
                for categoty, templates in self.answer_keywords.items():
                    for template in templates:
                        if template in sent.lower():
                            doc.answers_sentences.append((sent, categoty, sent_index))
                            break
        
        all_types = []
        for doc in self.docs:
            for sent, category in doc.answers_sentences:
                all_types.append(category)
        print(Counter(all_types))
        
        # use ChatGPT/GPT-4 to extract the exact answer from the answer sentence
        with open(f'answer_{self.source}_{self.time_stamp}.txt', 'w') as file:
            for doc in tqdm(self.docs, desc='processing docs'):
                for sent, category in doc.answers_sentences:
                    prompt = self.prompt_generator('answer', sent, category)
                    answer = self.llm(sent)
                    file.write(answer+'\n')
                    doc.answers.append(answer)

        print('Answers generation done.')
    
    def queries(self):
        with open(f'query_{self.source}_{self.time_stamp}.txt', 'w') as file:
            for doc in self.docs:
                if doc.queries is None:
                    doc.queries = []
                for index, (sent, category) in enumerate(doc.answers_sentences):
                    the_answer = doc.answers[index]
                    prompt = self.prompt_generator('query', sent, category)
                    query = self.llm(prompt)
                    doc.queries.append(query)
                    file.write(query+'\n')
        
        print('Queries generation done.')
    
    def mask_passage(self):
        with open(f'masked_passage_{self.source}_{self.time_stamp}.txt', 'w') as file:
            for doc in self.docs:
                for index, (sent, category) in enumerate(doc.answers_sentences):
                    the_answer = doc.answers[index]
                    the_query = doc.queries[index]
                    prompt = self.prompt_generator('masking', sent, category)
                    masked_passage = self.llm(prompt)
                    file.write(masked_passage+'\n')
                    doc.masked_answers_sentences.append(masked_passage)
        
        print('Masked passage generation done.')
    
    def qa_pairs(self):
        pairs = []
        for doc in self.docs:    
            for answer, query, sent, masked_sent in zip(doc.answers, doc.queries, doc.answers_sentences, doc.masked_answers_sentences):
                sent_index = sent[2]
                sents = doc.original_sentences.copy()
                sents[sent_index] = masked_sent
                doc.passage_to_input = '\n'.join(sents)
                pairs.append(
                    {
                        'query': query,
                        'answer': answer,
                        'passage': doc.passage_to_input,
                    }
                )
        with open(f'qa_pairs_{self.source}_{self.time_stamp}.txt', 'w') as file:
            json.dump(pairs, file, indent=2, ensure_ascii=False)
        return pairs
    
    def prompt_generator(self, request_type, answers_sentence, category):
        # generate prompt for gpt
        if request_type == 'answer':
            leading = {
                'summary': 'The following sentence has a summative statement. Please extract and copy the summative content.',
                'purpose': 'The following sentence has a purpose statement. Please extract and copy the purpose content.',
                'example': 'The following sentence has an example. Please extract and copy the example content.',
                'future': 'The following sentence has a future statement. Please extract and copy the future content.',
                'terminology': 'The following sentence has a terminology. Please extract and copy the terminology content.',
            }
            prompt = f"{leading[category]}\n----\n{answers_sentence}"
        elif request_type == 'query':
            leading = {
                'summary': f'The following sentence has a summary. Please generate a question targets the summative content. Here are some examples: {self.query_templates[category]}',
                'purpose': f'The following sentence has a purpose. Please generate a question targets the purpose content. Here are some examples: {self.query_templates[category]}',
                'example': f'The following sentence has an example. Please generate a question targets the example content. Here are some examples: {self.query_templates[category]}',
                'future': f'The following sentence has a future. Please generate a question targets the future content. Here are some examples: {self.query_templates[category]}',
                'terminology': f'The following sentence has a terminology. Please generate a question targets the terminology content. Here are some examples: {self.query_templates[category]}',
            }
            prompt = f"{leading[category]}\n----\n{answers_sentence}"
        elif request_type == 'masking':
            prompt = f"The following sentence has an answer of the categpry {category}. Please turn this sentence into a placeholder indicating only its subject. For example, 'CNN is a neural network proposed by ...' -> 'the definition of CNN is explained here'.\n----\n{answers_sentence}"

        return prompt
    
class PureLLMProcessor(BaseProcessor):
    """
    Pure LLM-based LatestEval processors.
    As the release of the new GPT-4 turbo, the context window is long enough to process the whole passage.
    This enables us to use this pure LLM-based processor to conduct the test samples.

    In this processor, we send the document to LLM, expecting LLM to render a json dict containing:
    1) the answer
    2) the answer type
    3) the query
    4) the masked passage

    Therefore, we process a document at a time instead of perform answer, query and masked passage generation separately.
    """

    def prompt_generator(self, doc: Doc):

        sents = doc.original_sentences
        sents = [f'{sent_index}. {sent.strip()}' for sent_index, sent in enumerate(sents)]
        original_passage = '\n'.join(sents)

        instruction = f"""You are a excellent professor in a top university. You are going to give a lecture about the passage above.
Your job is to identify key informations from the passage and write a corresponding question about it. We mainly focus on 5 types of key information:
1) terminology: the definition of a terminology, a concept. e.g., a new technique proposed in the passage, an institute that mentioned in the passage, or a project explained in the passage. The definition should be explicitly stated in the passage. Locations, Persons, or well known terms like WHO, BBC are not good examples.
2) summary: main finding of the passage, a one-sentence summary for a news article, or a general introduction for a python package. Usually indicated by phrases "in summary", "our contributions are mainly".
3) purpose: the purpose of some actions. Like the purpose of proposing a new technique, using a specific method, the reason of a conclusion drawn in the passage.
4) example: examples to illustrate something. This is usually indicated by phrases "for example" or "for instance".
5) future: predictions about the future. e.g., potential future works of a research, future trend of an event. Usually indicated by phrases "future work", "forecast"

Beware that all key points should be explicitly stated in the passage to make sure the questions are answerable based on the given passage. So key points from the reference are not of interests.
        
For each key point, return a json dict:
{{
    'sentence_index': <the index of the sentence in the passage>,
    'answer_type': '<which type of key information do you think it is>',
    'key_information': <the key information extracted from the sentence. This will be used as the reference answer.>,
    'query': '<write a query that targets the key information>',
    'place_holder': '<we replace the answer sentence with this place_holder to keep students from knowing the answer directly when we hand the passage to them.>',
}}

Here is an example:
{{
    'sentence_index': <the index of sentence, for example: 3>,
    'answer_type': 'terminology',
    'key_information': 'CNN refers to convolutional neural networks proposed by ...',
    'query': 'What is CNN?', or 'What does CNN mean?', or 'Can you explain what CNN means?',
    'place_holder': 'The definition of CNN is explained here.',
}}

The query should have diverse forms. But always target the key information. The place_holder is used to replace the answer sentence from the passage and should follow the format of the example.

Do the task sentence by sentence. And for each sentence, you should only return one json dict, which means each sentence should only have one key information extracted.
And for each passage, return at most 10 json dicts - 10 key informations and their corresponding queries and masked passages. return less if you cannot find more.
Return only the json file (dicts enclosed in square brackets) and nothing else."""

        prompt = f"{original_passage}\n--------------\n{instruction}"
        num_tokens = self._num_tokens(prompt)
        # print(prompt)

        return prompt

    def qa_pairs(self, doc: Union[Doc, List[Doc]] ):
        # the basic idea is to go through the passage sentence by sentence, and generate a json dict containing the answer, answer type, query and masked passage.
        # we start by checking the overall length of the passage. The length of the passage plus the max length of the answer generation should be less than max context window of LLM.
        # the number of max context window, and max output tokens:
        # gpt-3.5-turbo: 16385, 4096
        # gpt-4-1106-preview: 128000, 4096

        # Here we use gpt-4-1106-preview as the default model. So that the context window wont be a problem.
        # However, we still need to consider the max output tokens.
        # We should avoid very long output (the json) to be generated. So we should limit the number of qa pairs generated from a single document.

        prompt = self.prompt_generator(doc)
        response = self.llm(prompt, model='gpt-4-1106-preview')

        prompt_len = self._num_tokens(prompt)
        response_len = self._num_tokens(response)

        response = self._deal_with_json_from_llm(response)

        return {
            'id': f'{doc.source}-{doc.entry_id}',
            'prompt_len': prompt_len,
            'response_len': response_len,
            'response': response,
        }

    def _deal_with_json_from_llm(self, response):
        response = response.strip('`json\n')

        try:
            response = json.loads(response)
        except Exception as e:
            try:
                response = self._fix_broken_json_from_llm(response)
                response = json.loads(response)
            except Exception as e:
                pass

        return response
    
    def _fix_broken_json_from_llm(self, json_like_string):
        lines = json_like_string.strip().split('\n')
        
        fixed_lines = []
        for line in lines:
            single_quote_indices = [i for i, char in enumerate(line) if char == "'"]
            
            line_list = list(line)
            for i in single_quote_indices[:3] + single_quote_indices[-1:]:
                line_list[i] = '"'
            
            fixed_line = ''.join(line_list)
            fixed_lines.append(fixed_line)

        final_lines = []
        for line in fixed_lines:
            double_quote_indices = [i for i, char in enumerate(line) if char == '"']

            line_list = list(line)
            for i in double_quote_indices[3:-1]:
                line_list[i] = "'"

            fixed_line = ''.join(line_list)
            final_lines.append(fixed_line)
        
        fixed_json = '\n'.join(final_lines)
        return fixed_json

class BaseEval:
    
    def __init__(self, source):
        self.source = source
        this_week_str = datetime.now().strftime("%Y-%W")
        self.time_stamp = this_week_str

        self.today = datetime.today().strftime("%Y%m%d")

        if not os.path.exists(f'{source}/'):
            os.makedirs(f'{source}/')
    
    def prepare_docs(self):
        raise NotImplementedError
    
    def make_test_set(self, processor: BaseProcessor):

        # cache_path = f'qa_pairs_{self.source}_{self.time_stamp}.json'
        # if os.path.exists(cache_path):
        #     with open(cache_path, 'r') as file:
        #         all_results = json.load(file)
        #     return all_results
        
        all_results = []

        for doc in tqdm(self.docs, desc=f'making test set for {self.source}'):
            if self.check_whether_alread_exist(doc.entry_id):
                all_results.append(self.load_from_file(doc.entry_id))
                continue
            result = processor.qa_pairs(doc)
            result['created_at'] = self.today

            self.save_to_file(result, doc.entry_id)
            all_results.append(result)

        # save benchmarks to two places
        # 1) benchmarks/latest; and 2) benchmarks/{year}-{week}
        if not os.path.exists(f'benchmarks/latest/'):
            os.makedirs(f'benchmarks/latest/')
        if not os.path.exists(f'benchmarks/{self.time_stamp}/'):
            os.makedirs(f'benchmarks/{self.time_stamp}/')

        with open(f'benchmarks/latest/qa_pairs_{self.source}_{self.time_stamp}.json', 'w') as file:
            json.dump(all_results, file, indent=2, ensure_ascii=False)
        with open(f'benchmarks/{self.time_stamp}/qa_pairs_{self.source}_{self.time_stamp}.json', 'w') as file:
            json.dump(all_results, file, indent=2, ensure_ascii=False)
    
    def save_to_file(self, to_save, id):
        # to_save should be a list of dicts, or a string

        save_file_path = f'{self.source}/{id}.json'
        with open(save_file_path, 'w') as file:
            json.dump(to_save, file, indent=2, ensure_ascii=False)
        
    def check_whether_alread_exist(self, id):
        save_file_path = f'{self.source}/{id}.json'
        return os.path.exists(save_file_path)

    def load_from_file(self, id):
        save_file_path = f'{self.source}/{id}.json'
        with open(save_file_path, 'r') as file:
            data = json.load(file)
        return data

class ArxivEval(BaseEval):
    """
    deal with arXiv dataset.
    """

    def __init__(self, dataset_name = 'RealTimeData/arxiv_latest', num_docs = 500):
        """
        Args:
            args: the arguments
        """
        super().__init__(source='arxiv')

        self.dataset_name = dataset_name
        self.prepare_docs(num_docs = num_docs)

    def prepare_docs(self, num_docs):
        if num_docs == 'all':
            self.ds = load_dataset(self.dataset_name, split='train')
        else:
            self.ds = load_dataset(self.dataset_name, split=f'train[:{num_docs}]')

        self.docs = []
        docs = []
        for instance in self.ds:
            arxiv_id = instance['entry_id'].split('/')[-1]
            entry_id = arxiv_id
            text = instance['text']
            source = self.source

            doc = Doc(text=text, source=source, entry_id=entry_id)
            docs.append(doc)
                
        def parse_arxiv(doc):
            text = doc.text
            # remove anything before introduction
            # text = re.sub(r"^.*?(§)", r"\1", text, flags=re.DOTALL)

            # split article into sections
            sections = re.split(r"(?<!§\.)§\s", text)
            sections = [self.beautify_context(section) for section in sections if section.strip()]
            if len(sections) < 2:
                return None, sections
            
            passages = {}
            for i, section in enumerate(sections):
                if i ==0: 
                    passages['summary'] = f'Abstract\n{section}'
                if i ==1: 
                    passages['intro'] = section
                if section.lower().startswith('conclusion'):
                    passages['conclusion'] = section
                    continue
                
            return passages, sections

        for index, doc in enumerate(docs):
            fragments, sections = parse_arxiv(doc)
            if fragments is None: 
                continue
            doc.meta_data = fragments
            doc.original_passage = fragments['intro']
            doc.original_sentences = sent_tokenize(doc.original_passage)
            doc.sections = sections
            self.docs.append(doc)
        
        print(f"Loaded {len(self.docs)} documents from {self.source} on {self.dataset_name} time slice.")

    def beautify_context(self, context: str) -> str:
        context = context.replace("<cit.>", '').replace('<ref>', '')
        context = re.sub(r"\s+", " ", context)
        context = re.sub(r"\n+", " ", context)

        return context.strip()

class BBCNewsEval(BaseEval):

    def __init__(self, dataset_name, num_docs = 500):
        """
        Args:
            args: the arguments
        """
        super().__init__(source='bbc')
        self.dataset_name = dataset_name
        self.prepare_docs(num_docs=num_docs)

    def prepare_docs(self, num_docs):
        if num_docs == 'all':
            self.ds = load_dataset(self.dataset_name, split='train')
        else:
            self.ds = load_dataset(self.dataset_name, split=f'train[:{num_docs}]')

        self.docs = []
        docs = []
        for instance in self.ds:
            bbc_id = instance['link'].split('/')[-1]
            entry_id = bbc_id
            text = instance['content']
            source = self.source

            title = instance['title']
            description = instance['description']

            doc = Doc(text=text, source=source, entry_id=entry_id, meta_data={'title': title, 'description': description})
            docs.append(doc)
        
        for index, doc in enumerate(docs):
            if 'live' in doc.entry_id: 
                continue
            doc.original_passage = doc.text
            doc.original_sentences = sent_tokenize(doc.original_passage)
            self.docs.append(doc)
        
        print(f"Loaded {len(self.docs)} documents from {self.source} on {self.dataset_name} time slice.")

class GithubEval(BaseEval):

    def __init__(self, dataset_name, num_docs = 500):
        """
        Args:
            args: the arguments
        """
        super().__init__(source='github')
        self.dataset_name = dataset_name
        self.prepare_docs(num_docs=num_docs)

    def prepare_docs(self, num_docs):
        if num_docs == 'all':
            self.ds = load_dataset(self.dataset_name, split='train')
        else:
            self.ds = load_dataset(self.dataset_name, split=f'train[:{num_docs}]')

        self.docs = []
        for instance in self.ds:
            entry_id = instance['full_name'].replace('/', ' ')
            original_sentences = self.markdown_to_plain_text(instance['readme'])
            text = '\n'.join(original_sentences)
            source = self.source

            description = instance['description']

            doc = Doc(text=text, source=source, entry_id=entry_id, meta_data={'description': description}, original_sentences=original_sentences, original_passage=text)
            self.docs.append(doc)
        
        print(f"Loaded {len(self.docs)} documents from {self.source} on {self.dataset_name} time slice.")

    def markdown_to_plain_text(self, md_text):
        # Add newline before list items that are denoted by " - "
        md_text = md_text.replace(" - ", "\n- ")
        
        # Convert markdown to HTML
        html_text = markdown.markdown(md_text)
        text = ''.join(BeautifulSoup(html_text, features="html.parser").stripped_strings)

        # Replace markdown headers with a newline
        text = re.sub(r'#+\s', '\n', text)
        
        # Remove emoji-like patterns
        text = re.sub(r':\w+:', '', text)
        
        # Add newlines around numbers likely to be part of a list
        text = re.sub(r'(\d+\.)', r'\n\1', text)

        return [line.strip() for line in text.strip().split('\n') if line.strip()]

class CustomizedEval(BaseEval):

    def __init__(self, file_path, source = 'customized', num_docs = 500):
        """
        Args:
            args: the arguments
        """
        super().__init__(source=source)
        self.file_path = file_path
        self.prepare_docs(num_docs=num_docs)

    def prepare_docs(self, num_docs):
        files = glob(f'{self.file_path}/*.txt')

        self.docs = []
        docs = []
        for instance in files[:num_docs]:
            with open(instance, 'r') as file:
                text = file.read()

            # filename as entry_id
            entry_id = instance.split('/')[-1].split('.')[0]

            doc = Doc(text=text, source=self.source, entry_id=entry_id, meta_data=None)
            docs.append(doc)
        
        print(f"Loaded {len(self.docs)} documents from {self.source} with path {self.file_path}.")

        for index, doc in enumerate(docs):
            doc.original_passage = doc.text
            doc.original_sentences = sent_tokenize(doc.original_passage)
            self.docs.append(doc)

def process_source(source, file_path = None, num_docs = 300):
    if source == 'customized':
        assert file_path is not None, "Please provide the file path if the source is customized."
        benchmark = CustomizedEval(file_path, num_docs=num_docs)
    elif source == 'arxiv':
        benchmark = ArxivEval('RealTimeData/arxiv_latest', num_docs=num_docs)
    elif source == 'bbc':
        benchmark = BBCNewsEval('RealTimeData/bbc_latest', num_docs=num_docs)
    elif source == 'github':
        benchmark = GithubEval('RealTimeData/github_latest', num_docs=num_docs)
    
    benchmark.make_test_set(PureLLMProcessor(source, benchmark.time_stamp))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LatestEval data processor')
    
    # Define the --source argument
    parser.add_argument('--source', type=str, choices=['arxiv', 'bbc', 'github', 'customized', 'all'], required=True, 
                        help='The source of the data. Can be "arxiv", "bbc", "github", "all", or "customized".')
    
    # Define the --file_path argument
    parser.add_argument('--file_path', type=str,
                        help='The file path, required if the source is "customized".')
    
    # num_docs
    parser.add_argument('--num_docs', type=int, default=500,
                        help='The number of documents to process.')

    # Parse the arguments
    args = parser.parse_args()

    # check folder existance
    if not os.path.exists('benchmarks/'):
        os.makedirs('benchmarks/')

    # empty existing benchmarks under benchmarks/latest folder
    if len(glob('benchmarks/latest/*.json')) > 0:
        os.system('rm benchmarks/latest/*.json')

    # use multi-threading to accelerate if source == 'all'
    if args.source != 'all':
        process_source(args.source, file_path=args.file_path, num_docs=args.num_docs)
    else:
        with ThreadPoolExecutor(max_workers=3) as executor:
            for source in ['arxiv', 'bbc', 'github']:
                executor.submit(process_source, source, num_docs=args.num_docs)