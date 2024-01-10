import argparse

from src.build_context_simple import build_context_simple
from src.build_context_iterative import build_context_iterative
import requests
from qwikidata.entity import WikidataItem
from qwikidata.linked_data_interface import get_entity_dict_from_api
import json
from tqdm import tqdm
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models.openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import re
import pinecone

class ExperimentPipeline:
    """
    This class defines an experiment pipeline for processing datasets and generating answers to questions.

    Attributes:
        dataset_file (str): Path to the dataset file.
        llm (ChatOpenAI): An instance of a language model.
        embedding_model (OpenAIEmbeddings): An embedding model for context generation.
        contex_strategy (function): Strategy function for context generation.
        params (dict): Parameters for the context generation strategy.
        question_type (str): Type of questions in the dataset, determined by the dataset file's name.
    """

    def __init__(self, dataset_file, llm, embedding_model, contex_strategy, params):
        self.dataset_file = dataset_file
        self.llm = llm
        self.embedding_model = embedding_model
        self.contex_strategy = contex_strategy
        self.params = params

        # Determine the question type based on the dataset file name
        if "strategyqa" in dataset_file:
            self.question_type = 'yesno'
        elif "2wikimultihop" in dataset_file:
            self.question_type = 'short'

    def answer_question(self, question, info_context):
        """
        Answers a question using the language model and context.

        Args:
            question (str): The question to be answered.
            info_context (str): The context information to use for answering.

        Returns:
            tuple: Depending on the question type, returns a boolean or string answer, along with the response.
        """

        llm = self.llm
        question_type = self.question_type

        if question_type == "yesno":

            template = "To answer question: \"{query}\", use information: \"{info_context}\". Explain reasoning and COMPULSORILY give your best guess for answer as \"(YES)\" or \"(NO)\""
            prompt = PromptTemplate(template=template, input_variables=["query", "info_context"])
            llm_chain_yesno_answer_pipeline = LLMChain(prompt=prompt, llm=llm)

            response = llm_chain_yesno_answer_pipeline.run(query=question, info_context=info_context)
            print(response)
            if "(NO)" in response or "(No)" in response:
                return False, response
            if "(YES)" in response or "(Yes)" in response:
                return True, response
            return "Unsure", response
        
        elif question_type == "short":
            template = "To answer question: \"{query}\", use information: \"{info_context}\". Explain reasoning and make sure to give your final answer in () parentheses"
            prompt = PromptTemplate(template=template, input_variables=["query", "info_context"])
            llm_chain_short_answer_pipeline = LLMChain(prompt=prompt, llm=llm)

            response = llm_chain_short_answer_pipeline.run(query=question, info_context=info_context)
            print(response)
            potential_answers = re.findall(r'\((.*?)\)', response)
            if len(potential_answers) == 0:
                return "Unsure", info_context
            else:
                # return last thing in parentheses
                return potential_answers[-1], response
            
        else:
            raise ValueError("Invalid question_type: " + question_type)

    def run(self, question):
        """
        Generates context for a given question and produces an answer.

        Args:
            question (str): The question to be answered.

        Returns:
            tuple: Context, answer, and response for the given question.
        """

        if self.contex_strategy is None:
            return self.run_baseline(question)

        # Step 1: Generate context (using the specified strategy)
        context, graph = self.contex_strategy(self.llm, self.embedding_model, question, **self.params)
        print(context)

        # Step 2: Answer the question using the context built in Step 1
        answer, reponse = self.answer_question(question, context)

        # Step 3: Return the context, answer, and response for evaluation
        return context, answer, reponse
    
    def run_baseline(self, question):
        """
        Produces an answer directly using the LLM without any retrieval

        Args:
            question (str): The question to be answered.

        Returns:
            tuple: Context, answer, and response for the given question.
        """

        question_type = self.question_type

        if question_type == "yesno":

            template = "Answer the question: \"{query}\". Explain reasoning and COMPULSORILY give your best guess for answer as \"(YES)\" or \"(NO)\""
            prompt = PromptTemplate(template=template, input_variables=["query"])
            llm_chain_yesno_answer_pipeline = LLMChain(prompt=prompt, llm=llm)

            response = llm_chain_yesno_answer_pipeline.run(query=question)
            print(response)
            if "(NO)" in response or "(No)" in response:
                return "", False, response
            if "(YES)" in response or "(Yes)" in response:
                return "", True, response
            return "", "Unsure", response
        
        elif question_type == "short":
            template = "Answer the question: \"{query}\". Explain reasoning and make sure to give your final answer in () parentheses"
            prompt = PromptTemplate(template=template, input_variables=["query"])
            llm_chain_short_answer_pipeline = LLMChain(prompt=prompt, llm=llm)

            response = llm_chain_short_answer_pipeline.run(query=question)
            print(response)
            potential_answers = re.findall(r'\((.*?)\)', response)
            if len(potential_answers) == 0:
                return "", "Unsure", response
            else:
                # return last thing in parentheses
                return "", potential_answers[-1], response
            
        else:
            raise ValueError("Invalid question_type: " + question_type)
        
    
    def wikimultihop_eval(self, out_file, num_todo=2):
        """
        Evaluates the performance of the pipeline on a wikimultihop dataset.

        Args:
            out_file (str): File to write the evaluation results to.
            num_todo (int): Number of items from the dataset to process.

        Returns:
            tuple: Count of correct and incorrect answers, and the modified dataset.
        """
        counts = {}
        counts[True] = 0
        counts[False] = 0
        # read jsonl file
        dataset = []
        with open(self.dataset_file, "r") as f:
            json_list = list(f)
            for json_str in json_list:
                dataset.append(json.loads(json_str))
        try:
            for i, task in tqdm(enumerate(dataset[:num_todo])):
                question = task['text']
                answer = task['metadata']['answer']
                metadata = task['metadata']
                metadata['our_context'], metadata['our_answer'],  metadata['our_response'] = self.run(question)
                if metadata['our_answer'].lower() == answer.lower():
                    counts[True] += 1
                else:
                    counts[False] += 1
        finally:
            dataset.append(counts)
            with open(out_file, "w+") as f:
                json.dump(dataset, f)
        print(counts)
        return counts, dataset

    def strategyqa_eval(self, out_file, num_todo=2):
        """
        Evaluates the performance of the pipeline on a StrategyQA dataset.

        Args:
            out_file (str): File to write the evaluation results to.
            num_todo (int): Number of items from the dataset to process.

        Returns:
            tuple: Count of various answer categories, and the modified dataset.
        """
        counts = {}
        counts[True] = {True: 0, False: 0, "Unsure": 0}
        counts[False] = {True: 0, False: 0, "Unsure": 0}
        try:
            with open(self.dataset_file, "r") as f:
                dataset = json.load(f)
            for i, task in tqdm(enumerate(dataset[:num_todo])):
                question = task['question']
                answer = task['answer']
                dataset[i]['our_context'], dataset[i]['our_answer'], dataset[i]['our_response'] = self.run(question)
                counts[answer][dataset[i]['our_answer']] += 1
        finally:
            with open(out_file, "w+") as f:
                dataset.append(counts)
                json.dump(dataset, f)
        return counts, dataset


search_strategies = {
    'simple': build_context_simple,
    'iterative': build_context_iterative,
    'none': None,
}

dataset_files = {
    'strategyqa': 'datasets/strategyqa/questions.json',
    'wikimultihop': 'datasets/2wikimultihop/queries.jsonl',
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment Pipeline for Datasets')
    parser.add_argument('--k_values', type=int, nargs='+', help='List of values for k')
    parser.add_argument('--choose_types', nargs='+', help='List of choose types')
    parser.add_argument('--search_strategies', nargs='+', help='List of search strategies')
    parser.add_argument('--datasets', nargs='+', help='List of dataset files')
    parser.add_argument('--num_todo', type=int, help='Number of items to process')

    args = parser.parse_args()

    if args.datasets is None:
        args.datasets = ['strategyqa', 'wikimultihop']

    if args.k_values is None or args.choose_types is None or args.search_strategies is None or args.datasets is None:
        args.k_values = [0]
        args.choose_types = [None]
        args.search_strategies = ['none']

    for dataset in args.datasets:
        for k in args.k_values:
            for choose_type in args.choose_types:
                for search_strategy in args.search_strategies:

                    print("Running experiment with:")
                    print("dataset: " + dataset)
                    print("k: " + str(k))
                    print("choose_type: " + str(choose_type))
                    print("search_strategy: " + str(search_strategy))

                    params = {
                        'choose_count': k,
                        'choose_type': choose_type,
                    }

                    if search_strategy == 'iterative':
                        params['max_depth'] = 2
                        params['max_branching'] = 2

                    # Create the language model and embedding model here
                    llm = ChatOpenAI(openai_api_key="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", temperature=0.0, model_name='gpt-3.5-turbo')
                    embeddings_model = OpenAIEmbeddings(openai_api_key="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")

                    # Get the dataset file here
                    dataset_file = dataset_files[dataset]

                    # Create and run the pipeline here
                    pipeline = ExperimentPipeline(
                        dataset_file=dataset_file,
                        llm=llm,
                        embedding_model=embeddings_model,
                        contex_strategy=search_strategies[search_strategy],
                        params=params,
                    )
                    if 'wikimultihop' in dataset_file:
                        out_file = f'results/wikimultihop_{search_strategy}_{choose_type}_{k}.json'
                        if search_strategy == 'none':
                            out_file = f'results/wikimultihop_{search_strategy}.json'
                        pipeline.wikimultihop_eval(out_file=out_file, num_todo=args.num_todo)
                    elif 'strategyqa' in dataset_file:
                        out_file = f'results/strategyqa_{search_strategy}_{choose_type}_{k}.json'
                        if search_strategy == 'none':
                            out_file = f'results/strategyqa_{search_strategy}.json'
                        pipeline.strategyqa_eval(out_file=out_file, num_todo=args.num_todo)
                    else:
                        raise ValueError("Invalid dataset file: " + dataset_file)


    # for choose_type in ['classic', 'nearest_neighbor']:
    #     for choose_count in [3, 5]:
    #         params = {
    #             'choose_type': choose_type,
    #             'choose_count': choose_count,
    #         }
    #         llm = ChatOpenAI(openai_api_key="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", temperature=0.0, model_name='gpt-3.5-turbo')
    #         embeddings_model = OpenAIEmbeddings(openai_api_key="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")
    #         pipeline = ExperimentPipeline(
    #             dataset_file='datasets/strategyqa/questions.json',
    #             # dataset_file='datasets/2wikimultihop/queries.jsonl',
    #             llm=llm,
    #             embedding_model=embeddings_model,
    #             contex_strategy=None,
    #             params=params,
    #         )

    #         pipeline.strategyqa_eval(out_file=f'results/strategyqa_none_{choose_type}_{choose_count}.json', num_todo=50)
    #         # pipeline.wikimultihop_eval(out_file=f'results/wikimultihop_none_{choose_type}_{choose_count}.json', num_todo=50)