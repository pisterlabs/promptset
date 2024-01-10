from swissnyf.retriever.base_retriever import BaseRetriever
from langchain.prompts import PromptTemplate
from abc import abstractclassmethod
from typing import Optional, Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM
from sentence_transformers import SentenceTransformer, util
from torchtyping import TensorType
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.tools.tool_spec.load_and_search.base import LoadAndSearchToolSpec
from llama_index.llms import AzureOpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index.embeddings import HuggingFaceEmbedding
import re, torch
import numpy as np
import time

class BaseAPI:
    def __init__(
        self,
        name: str, # the name of the API call
        description: str, # the natural language description of the API call
        prompt_template: PromptTemplate, # API usage guide
        pattern: str=None # output pattern
    ):
        self.name = name
        self.description = description
        self.prompt_template = prompt_template
        self.pattern = pattern
        self.mocked_result = None

    @abstractclassmethod
    def execute(self, *args, **kwards):
        pass
    
    def __call__(self, *args: str, **kargs: str) -> Optional[str]:
        if "mock" in kargs and kargs["mock"] and self.mocked_result is not None:
            return self.mocked_result

        kargs.pop("mock", None)

        try:
            output = self.execute(*args, **kargs)
        except Exception as e:
            print(f"API {self.name} failed with error: {e}, args: {args}, kwargs: {kargs}")
            return None

        return str(output) if output is not None else None

NUMBER_PATTERN = "n"
ENGLISH_TOKEN_PATTERN = "e"
SYMBOL_PATTERN = "s"
NON_ENGLISH_PATTERN = "o"
SLEEP_PATTERN = "S"
MOVE_PATTERN = "m"
TIME_PATTERN = "t"
WEATHER_PATTERN = "w"


english_token_pattern = re.compile(r'[A-Za-z]+')
number_pattern = re.compile(r'[-+]?\d*\.\d+|[-+]?\d+')
symbol_pattern = re.compile(r'[^\w\s]+')
non_english_pattern = re.compile(r'[^\x00-\x7F]')
sleep_pattern = re.compile(r"sleep for [0-9]*\.?[0-9]+ seconds")
robotmove_pattern = re.compile(r"(?i)(?=.*\brobot\b)(?=.*\bmoving\b).*")
time_pattern = re.compile(r'\b(\d{4}-\d{2}-\d{2}|\d{2}:\d{2}(?::\d{2})?|\d{2}/\d{2}/\d{4})\b')


PATTERN_PROB = {ENGLISH_TOKEN_PATTERN: 0.75, NON_ENGLISH_PATTERN: 0.15, NUMBER_PATTERN: 0.02, SYMBOL_PATTERN: 0.02, SLEEP_PATTERN: 0.02, MOVE_PATTERN: 0.02, TIME_PATTERN: 0.02}

PATTERN = {SYMBOL_PATTERN: symbol_pattern, NUMBER_PATTERN: number_pattern, NON_ENGLISH_PATTERN: non_english_pattern, ENGLISH_TOKEN_PATTERN: english_token_pattern, SLEEP_PATTERN: sleep_pattern, MOVE_PATTERN: robotmove_pattern, TIME_PATTERN: time_pattern}

class GearRet(BaseRetriever):
    def __init__(self, 
                filter_llm = "asfa",
                experiment = 'gptj',
                openai_model = 'gpt3',
                tool = None,
                prompt = 'mt',
                slm1 = 'EleutherAI/gpt-neo-1.3B',
                slm2 = "sentence-transformers/all-mpnet-base-v2",
                max_tokens = 512,
                top_k = 5,
                llm = "EleutherAI/gpt-j-6B",
                early_stop = 0,
                check_point = None,
                dataset = None,
                fdevice = "cuda:0",
                ALPHA = 0.75,
                output = None, 
                **kwargs):
        
        super().__init__(**kwargs)
        self.apis = [] #might need to change later
        self.slm1 = slm1
        self.slm2 = slm2
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.llm = llm
        self.early_stop = early_stop
        self.check_point = check_point
        self.dataset = dataset
        self.output = output
        self.experiment = experiment
        self.openai_model = openai_model
        self.tool = tool
        self.prompt = prompt
        self.fdevice = fdevice
        self.ALPHA = ALPHA
        self.filter_llm = filter_llm
        # order the apis by if the prompt_templete is None
        self.apis.sort(key=lambda x: x.prompt_template is None)
        self.model = SentenceTransformer(self.slm2)
        self.tokenizer = AutoTokenizer.from_pretrained(self.slm1, padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm_model = AutoModelForCausalLM.from_pretrained(self.slm1, pad_token_id = self.tokenizer.eos_token_id).to(self.device)
        self.max_tokens = self.max_tokens
        self.verbose = self.verbose
        self.top_k = self.top_k
        
    def create_base_api(self):
        self.apis = []
        for api in self.all_tools: #can be vectorized
            # for api in [*tool:
            input_ids = self.tokenizer(self.generate_prompt_template(api[0], api[1]), return_tensors="pt").input_ids.to(self.device)
            outputs = self.lm_model.generate(input_ids, do_sample=True, max_length=1500)
            usage = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0] 
            usage = usage.replace('{', '').replace('}', '') + "\nInput:{input}" + "\nOutput:"
            # print("\n\n\n", api, usage, "\n\n\n")
            api = BaseAPI(api[0], api[1], usage)
            self.apis.append(api)
        

    def _encode_patterns(self, input: str, prior_pattern: str=None) -> Dict[str, int]:
        encoded_pattern = {}
        if input is None:
            for pattern_type in PATTERN.keys():
                encoded_pattern[pattern_type] = 0
        else:
            if prior_pattern is None:
                for pattern_type, pattern_regex in PATTERN.items():
                    encoded_pattern[pattern_type] = len(pattern_regex.findall(input))
            else:
                # API has specifc output pattern
                for pattern_type, pattern_regex in PATTERN.items():
                    encoded_pattern[pattern_type] = len(pattern_regex.findall(input)) if pattern_type == prior_pattern else 0

        return encoded_pattern

    def generate_call(self, input: str) -> List[str]:
        """
        Put input question and api prompt into language model and return the generated API calls
        Args:
            input (str): input question
        Returns:
            List[str]: Generated API calls
        """
        if "chatgpt" not in self.slm1:

            eos_token_id = self.tokenizer("\n")["input_ids"][0] # eos_token_id list for EleutherAI/gpt-neo-1.3B

            # batch generation for all APIs
                    
            prompts_batch = [api.prompt_template.format(input=input) for api in self.apis if api.prompt_template is not None]
            inputs_batch = self.tokenizer(prompts_batch, padding=True, return_tensors="pt")
            PROMPT_LENGTH = len(inputs_batch['input_ids'][0])

            mtokens = self.max_tokens
            while mtokens > 0:
                try:
                    outputs_ids = self.lm_model.generate(inputs_batch['input_ids'].to(self.device),
                                                         attention_mask=inputs_batch['attention_mask'].to(self.device),
                                                         do_sample=True,
                                                         top_p=0.9,
                                                         eos_token_id = eos_token_id,
                                                         max_new_tokens = mtokens)
                    outputs_ids = outputs_ids[:, PROMPT_LENGTH:]
                    outputs = self.tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
                    break
                except:
                    outputs = [""] * len(prompts_batch)
                    mtokens = mtokens // 2
                    # print(f"New Max tokens: {mtokens}")
                    continue
            
        else:
            outputs = []
            for api in self.apis:
                if api.prompt_template is not None:
                    outputs.append(self.lm_model.execute(input, api.prompt_template))
    
        # for apis with no prompt_template, use the input question as the API call
        if len(outputs) < len(self.apis):
            outputs.extend([None] * (len(self.apis) - len(outputs)))

        if self.verbose:
            for i in range(len(self.apis)):
                print(f"{self.apis[i].name} API Call: {outputs[i]}")

        return outputs
        
    def obtain_api_response(self, input: str, outputs: List[str]) -> List[Optional[str]]:
        """
        Obtain the response from generated outputs
        """
        api_responses = []

        for api, output in zip(self.apis, outputs):
            if api.prompt_template is None: # APIs do not need API calls from LM
                request_args = None
                if api.name != "MultilingualQA":
                    api_response = api(input)  
                else: 
                    trying = 0
                    while trying < 3:
                        try:
                            api_response=api(input,
                                         mtdest = 'en', 
                                         mtsrc = api.apis['MT'].translator.detect(re.search('question: (.*)context:', input).group(1)).lang if re.search('question: (.*)context:', input) is not None else 'en')
                            break
                        except:
                            api_response = None
                            trying += 1
                            time.sleep(60)
            else:                           # APIs need generated API calls
                request_args = self._extract_api_request_content(output, api.name)
                if request_args is None:
                    api_response = None
                else:
                    api_response = api(*request_args) if api.name != "MultilingualQA" else None


            if self.verbose:
                print(f"{api.name} request content: {request_args}")
                print(f"{api.name} response: {api_response}")
            api_responses.append(api_response)

        return api_responses

    def semantic_similarity_score(self, input: str) -> TensorType["num_apis"]:
        """
        Compute the semantic score between the input and the API description. 
        Args:
            input (str): the input string to be matched with the API description.
        Returns:
            List[float]: the semantic similarity score between the input and the API description.
        """
        input_batch = [input] * len(self.apis)
        description_batch = [api.description for api in self.apis]

        input_emb = self.model.encode(input_batch, convert_to_tensor=True)
        description_emb = self.model.encode(description_batch, convert_to_tensor=True)

        #Compute cosine-similarities
        cosine_scores = util.cos_sim(input_emb, description_emb)
        #Compute cosine-similarities
        cosine_scores = util.cos_sim(input_emb, description_emb)
        
        if self.verbose:
            for i in range(len(input_batch)):
                print("{} \t\t Semantic Score: {:.4f}".format(self.apis[i].name, cosine_scores[i][i]))
        
        return cosine_scores[0].cpu().numpy()

    def encode_answers(self, api_responses: List[Optional[str]], candidate: Optional[str]) -> Tuple[List[Optional[Dict[str, int]]], Optional[Dict[str, int]]]:
        """
        Encode the api responses and candidate answers based on token patterns
        """ 
        encoded_response_patterns = []

        for api_index, api_response in enumerate(api_responses):
            # encode all tokens to the same pattern for the special API
            if (self.apis[api_index].pattern is not None) and (api_response is not None):
                encoded_pattern = self._encode_patterns(api_response, self.apis[api_index].pattern)
            elif (self.apis[api_index].pattern is None) and (api_response is not None):
                # unmeaningful response
                if api_response.lower().strip() in ["", "unknown", "none", "no answer"]: 
                    encoded_pattern = None
                else:
                    encoded_pattern = self._encode_patterns(api_response)
            else:
                encoded_pattern = None
            
            if self.verbose:
                print(f"{self.apis[api_index].name}: (Orignal Response: {api_response}, Encoded Response: {encoded_pattern})")

            encoded_response_patterns.append(encoded_pattern)
        
        # encode the candidate answer
        encoded_candidate_response = self._encode_patterns(candidate)
        if self.verbose:
            print(f"Potential Answer: (Orignal candidate: {candidate} Encoded candidate: {encoded_candidate_response})")

        return encoded_response_patterns, encoded_candidate_response

    def pattern_similarity_score(self, response_patterns: List[Optional[Dict[str, int]]], candidate_pattern: Optional[Dict[str, int]]) -> List[float]:

        candidate_length = sum(candidate_pattern.values())
        pattern_similarity_scores = []

        for api_index in range(len(response_patterns)):
            if response_patterns[api_index] is None:
                pattern_similarity_scores.append(0) # assign the minimum pattern score if the response is empty
                continue
            else:
                pattern_similarity_score = 0
                # find the number of each pattern in the response
                pattern_count = response_patterns[api_index]
                response_length = sum(response_patterns[api_index].values())
                # compute the pattern similarity score
                for pattern in PATTERN_PROB.keys():
                    # add-1 smoothing
                    pattern_similarity_score += pattern_count[pattern] * (candidate_pattern[pattern]+1) / ((candidate_length+len(PATTERN_PROB)) * response_length) * np.log(1 / PATTERN_PROB[pattern])
            
                pattern_similarity_scores.append(pattern_similarity_score)
        
        if self.verbose:
            for i in range(len(self.apis)):
                print(f"{self.apis[i].name} pattern similarity score: {pattern_similarity_scores[i]}")
        
        return pattern_similarity_scores

    def filter(self, input: str) -> Tuple[List[BaseAPI], Dict[str, List[float]]]:
        """
        Filter the APIs by two similarity scores
        Args:
            input (str): input question
        Returns:
            List[BaseAPI]: list of APIs that pass the filtering
        """
        
        if "chatgpt" not in self.slm1 :
            # generate potential answers from LM1 without prompt. Open-domain QA
            
            input_ids = self.tokenizer.encode(input, return_tensors="pt").to(self.device)
            eos_token_id = self.tokenizer(".\n")["input_ids"][0] 

            with torch.no_grad():
                output_ids_without_prompt = self.lm_model.generate(input_ids=input_ids,
                                                                   do_sample = False,
                                                                   eos_token_id=eos_token_id,
                                                                   max_new_tokens =self.max_tokens)

            output_ids_without_prompt = output_ids_without_prompt[:, input_ids.shape[1]:]
            output_without_prompt = self.tokenizer.decode(output_ids_without_prompt[0], skip_special_tokens=True).strip()
            output_without_prompt = self._process_potential_answers(output_without_prompt)
        
        
        else:
            output_without_prompt = self.lm_model.execute(input)
        
        if self.verbose:
            print("***********************Start Filtering***********************")
            print(f"Input query: {input}")
            print(f"Potential answers: {output_without_prompt}")
        
        generated_outputs = self.generate_call(input)

        api_responses = self.obtain_api_response(input, generated_outputs)
        assert len(api_responses) == len(self.apis), "api_responses and apis list should have the same length"
        
        # encode the api responses and candidate answers
        encoded_response_patterns, encoded_candidate_pattern = self.encode_answers(api_responses, output_without_prompt)

        # compute the semantic similarity score between the input question and the API description
        semantic_similarity_scores = self.semantic_similarity_score(input)
        # compute the pattern similarity score between the api responses and the candidate answer
        pattern_similarity_scores = self.pattern_similarity_score(encoded_response_patterns, encoded_candidate_pattern)

        final_similarity_scores = [semantic_similarity_scores[i]*self.ALPHA + pattern_similarity_scores[i]*(1-self.ALPHA) for i in range(len(self.apis))]

        if self.verbose:
            for i in range(len(self.apis)):
                print(f"{self.apis[i].name} final similarity score: {final_similarity_scores[i]}")

        filtered_apis = [self.apis[i] for i in np.argsort(final_similarity_scores)[-self.top_k:]]

        filtered_apis_with_scores = {self.apis[i].name: [float(semantic_similarity_scores[i]), float(pattern_similarity_scores[i])] for i in np.argsort(final_similarity_scores)[-self.top_k:]}
        if self.verbose:
            print(filtered_apis_with_scores)
            print("***********************End filtering***********************")
        return filtered_apis_with_scores

    def _text2int(self, text: str, numwords={}) -> str:
        if not numwords:
          units = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen",
          ]
          tens = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
          scales = ["hundred", "thousand", "million", "billion", "trillion"]
        #   numwords["and"] = (1, 0)
          for idx, word in enumerate(units):    numwords[word] = (1, idx)
          for idx, word in enumerate(tens):     numwords[word] = (1, (idx+2) * 10)
          for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)
        current = result = 0
        output = ""
        tokens = re.split("\s|(?<!\d)[,.:](?!\d)", text)
        for idx, word in enumerate(tokens):
            if word not in numwords:
                num = result + current
                if num != 0:
                    output = output + str(num) + " "
                result = current = 0
                output = output + word + " "
            elif idx != len(tokens) - 1:
                scale, increment = numwords[word]
                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
            else: 
                scale, increment = numwords[word]
                current = current * scale + increment
                result += current
                output = output + str(result)
        return output

    def _process_potential_answers(self, potential_answer: str) -> str:
        """
        Process the potential answer by removing the \n and stripping. Converting numeric words to numbers
        Args:
            potential_answer (str): the potential answer
        Returns:
            str: the processed potential answer
        """
        potential_answer = potential_answer.replace("\n", " ")
        potential_answer = potential_answer.strip().lower()
        potential_answer = self._text2int(potential_answer) # convert numeric words to numbers
        return potential_answer
    
    def _extract_api_request_content(self, text: str, api_name:str) -> list:
        """Extract the content of an API request from a given text."""
        try:
            left_bracket = text.split(f"{api_name}(")[1]
            right_bracket_ind = left_bracket.rfind(")")
            inside = left_bracket[:right_bracket_ind]
        except Exception as e:
            return None
    
        request_args = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', inside)
        request_args = [arg.strip() for arg in request_args]
        request_args = [arg.replace('"', '') for arg in request_args]
        return request_args

    def generate_prompt_template(self, name, description):
        
        prompt_template = f"""
        You are a tool user. Your task is to create tool usage examples based on an API name and its description.
        Here are few examples of this task:
     ====================================
    Input 1:
    "name": "bing_news_search",
    "description": "bing_news_search tool connects to a Bing account and enables an Agent to perform various news   search tasks using the Bing News Search service."
    
    Output 1:
    "name": "bing_news_search",
    "description": "bing_news_search tool connects to a Bing account and enables an Agent to perform various news search tasks using the Bing News Search service.",
    "examples": 
    Input: Find the latest news articles about climate change.
    Output: <Q> Search for the latest news articles about climate change. <API> [bing_news_search(climate change)]
    
    Input: Locate news reports on the recent advancements in artificial intelligence.
    Output: <Q> Find news reports on the recent advancements in artificial intelligence. <API> [bing_news_search(recent advancements in artificial intelligence)]
    
    Input: I need news articles on the global economic impact of the COVID-19 pandemic.
    Output: <Q> Retrieve news articles on the global economic impact of the COVID-19 pandemic. <API> [bing_news_search(global economic impact COVID-19 pandemic)]
    
    Input: Identify recent news stories about space exploration.
    Output: <Q> Find recent news stories about space exploration. <API> [bing_news_search(recent news space exploration)]
    
    Input: Search for news updates on the stock market.
    Output: <Q> Explore news updates on the stock market. <API> [bing_news_search(stock market news updates)]
    
    ================================================
    Input 2:
    "name" = "qa"
    "description" = "Question Answering API helps you get additional information required to answer the question."
    Output 2:
    "name" = "qa"
    "description" = "Question Answering API helps you get additional information required to answer the question."
    qa_prompt = Question Answering API helps you get additional information required to answer the question. You task is to rephrase the question prepended by the special token <Q> and generate QA API call prepended by <API> for solving that question. Here are some examples of API calls:
    You can call the API by writing "[QA(question)]" where "question" is the question you want to ask. Here are some examples of QA API calls:
    
    Input: Where was Joe Biden Born?
    Output: <Q> Where was Joe Biden Born? <API> [QA("Where was Joe Biden Born?")].
    
    Input: What other name is Coca-Cola known by?
    Output: <Q> What other name is Coca-Cola known by? <API> [QA("What other name is Coca-Cola known by?")].
    
    Input: What is the capital of France?
    Output: <Q> What is the capital of France? <API> [QA("What if the capital of France?")].

    
    
    Now lets start:
    "name" = {name}
    "description" = {description}
    Output:
    """
    
        return prompt_template

    def set_tool_def(self, tool_def) -> None:
        super().set_tool_def(tool_def)
        self.create_base_api()
        
    def add_tool_def(self, new_tool_def) -> None:
        super().add_tool_def(new_tool_def)
        self.create_base_api()


