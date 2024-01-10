import argparse
import logging
import os
from dataclasses import asdict, dataclass
from typing import Dict

import yaml
from langchain import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DeepLake


@dataclass
class SimilaritySearchData:
    query: str
    current_combined_docs: str
    previous_combined_docs: str


@dataclass
class SimilaritySearchDataWithJudgeDecision(SimilaritySearchData):
    judge_decision: str


def response_data_representer(dumper, data):
    return dumper.represent_dict(asdict(data))


yaml.add_representer(SimilaritySearchDataWithJudgeDecision, response_data_representer)


@dataclass
class ResponseData:
    previous_response: str
    current_response: str
    context: str


@dataclass
class ResponseDataWithJudgeDecision(ResponseData):
    judge_decision: str


def response_data_representer(dumper, data):
    return dumper.represent_dict(asdict(data))


yaml.add_representer(ResponseData, response_data_representer)


def load_questions_answers():
    with open("tests/test_data/qa_pairs.yaml") as file:
        all_qa_pairs = yaml.safe_load(file)
    questions = [qa_pair["question"] for qa_pair in all_qa_pairs]
    answers = [qa_pair["answer"] for qa_pair in all_qa_pairs]
    return questions, answers


def save_responses(
    response_dict: Dict[str, SimilaritySearchDataWithJudgeDecision]
) -> None:
    response_list = [(k, asdict(v)) for k, v in response_dict.items()]
    with open("manual_similarity_search_evaluation.yaml", "w") as file:
        yaml.safe_dump(response_list, file)


def save_prompt_responses(
    response_dict: Dict[str, ResponseDataWithJudgeDecision]
) -> None:
    response_list = [(k, asdict(v)) for k, v in response_dict.items()]

    with open("manual_response_evaluation.yaml", "w") as file:
        yaml.safe_dump(response_list, file)


def create_context_eval_prompt(query, current_combined_docs, previous_combined_docs):
    prompt = f"""
        Given the following question and the results of performing a similarity search on the current and previous vectorstores, which vectorstore returned information more useful to answering the question? Vectorstores should be penalized for returning information that is not needed to answer the question.
            Question: {query}
            C1: {current_combined_docs}
            C2: {previous_combined_docs}
        Only answer with "C1" or "C2"
    """
    return prompt


def create_eval_prompt(
    question: str, context: str, previous_answer: str, current_answer: str
) -> str:
    prompt = f"""
        Given the following question and answers from two different models, which is better?
            Question: {question}
            Context: {context}
            A1: {previous_answer}
            A2: {current_answer}
        Only answer with "M1" or "M2". If both are equally good/bad answer "M0". Don't explain your answer.
    """
    return prompt


class IngestStrategyFactory:
    @classmethod
    def create(cls, type, chunk_size, chunk_overlap, args):
        return type(
            args,
            RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            ),
        )


class ChunkEvaluator:
    def __init__(self, args):
        self.args = args
        self.embedding_model = OpenAIEmbeddings()
        self.model_runner = OpenAiModelRunner(
            api_key=os.environ["OPENAI_API_KEY"],
            project_name="llm-question_answering",
            in_token_limit=2000,
        )
        self.loader = DirectoryLoader(
            os.path.join(self.args.data_dir, "articles"),
            glob="**/*.txt",
            loader_cls=TextLoader,
        )
        self.current_ingest_strategy: ArticleIngest = IngestStrategyFactory.create(
            type=ArticleIngest,
            args=self.args,
            chunk_size=self.args.current_chunk_size,
            chunk_overlap=self.args.current_chunk_overlap,
        )
        self.previous_ingest_strategy: ArticleIngest = IngestStrategyFactory.create(
            type=ArticleIngest,
            args=self.args,
            chunk_size=self.args.previous_chunk_size,
            chunk_overlap=self.args.previous_chunk_overlap,
        )
        self.current_deeplake_dir = os.path.join(self.args.save_dir, "current_articles")
        self.previous_deeplake_dir = os.path.join(
            self.args.save_dir, "previous_articles"
        )

        self.questions, self.answers = load_questions_answers()
        self.judge_mapping = {"C1": "Previous", "C2": "Current"}
        self.prompt_judge_mapping = {"M1": "Previous", "M2": "Current"}
        self.better_model_response_counts = {"Previous": 0, "Current": 0, "Neither": 0}
        self.better_prompt_response_counts = {"Previous": 0, "Current": 0, "Neither": 0}
        self.current_chain = load_qa_with_sources_chain(
            llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo"), prompt=current_prompt
        )
        self.previous_chain = load_qa_with_sources_chain(
            llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo"),
            prompt=previous_prompt,
        )

    def build_vectorstores(self):
        chunked_documents = {
            "current_docs": self.current_ingest_strategy.load_docs(),
            "previous_docs": self.previous_ingest_strategy.load_docs(),
        }
        self.current_vectorstore = DeepLake.from_documents(
            chunked_documents["current_docs"],
            self.embedding_model,
            dataset_path=self.current_deeplake_dir,
            overwrite=True,
        )
        self.previous_vectorstore = DeepLake.from_documents(
            chunked_documents["previous_docs"],
            self.embedding_model,
            dataset_path=self.previous_deeplake_dir,
            overwrite=True,
        )
        logger.info(
            f"Persisting previous article vectorstore to {self.previous_deeplake_dir}"
        )
        logger.info(
            f"Persisting current article vectorstore to {self.current_deeplake_dir}"
        )

    def evaluate_chunking_methods(self, queries):
        responses_with_judge_decision = {}
        model_responses_with_judge_decision = {}
        for query in queries:
            current_similar_docs = self.current_vectorstore.similarity_search(
                query, k=3
            )
            previous_similar_docs = self.previous_vectorstore.similarity_search(
                query, k=3
            )
            current_combined_docs = " ".join(
                [c.page_content for c in current_similar_docs]
            )
            previous_combined_docs = " ".join(
                [c.page_content for c in previous_similar_docs]
            )

            eval_prompt: str = create_context_eval_prompt(
                query=query,
                current_combined_docs=current_combined_docs,
                previous_combined_docs=previous_combined_docs,
            )
            judge_response: str = self.model_runner.chat_prompt(
                eval_prompt, n_generations=1, temperature=0, model="gpt-4"
            )[0]

            if judge_response in self.judge_mapping:
                self.better_model_response_counts[
                    self.judge_mapping[judge_response]
                ] += 1
                judge_decision = self.judge_mapping[judge_response]
            else:
                self.better_model_response_counts["Neither"] += 1
                judge_decision = "Neither"

            responses_with_judge_decision[
                query
            ] = SimilaritySearchDataWithJudgeDecision(
                previous_combined_docs=previous_combined_docs,
                current_combined_docs=current_combined_docs,
                query=query,
                judge_decision=judge_decision,
            )
            # logger.info(f"Updated chunking response counts: {self.better_model_response_counts}")

            current_answer = self.current_chain(
                {"question": query, "input_documents": current_similar_docs}
            )
            previous_answer = self.previous_chain(
                {"question": query, "input_documents": current_similar_docs}
            )
            model_eval_prompt: str = create_eval_prompt(
                question=query,
                context=current_combined_docs,
                previous_answer=previous_answer,
                current_answer=current_answer,
            )
            model_judge_response = self.model_runner.chat_prompt(
                model_eval_prompt, n_generations=1, temperature=0, model="gpt-4"
            )[0]
            if model_judge_response in self.prompt_judge_mapping:
                self.better_prompt_response_counts[
                    self.prompt_judge_mapping[model_judge_response]
                ] += 1
                model_judge_decision = self.prompt_judge_mapping[model_judge_response]
            else:
                self.better_prompt_response_counts["Neither"] += 1
                model_judge_decision = "Neither"
            model_responses_with_judge_decision[query] = ResponseDataWithJudgeDecision(
                previous_response=previous_answer["output_text"],
                current_response=current_answer["output_text"],
                context=current_combined_docs,
                judge_decision=model_judge_decision,
            )
            logger.info(
                f"Updated better prompt response counts: {self.better_model_response_counts}"
            )

        return responses_with_judge_decision, model_responses_with_judge_decision

    def run(self, queries):
        (
            responses_with_judge_decision,
            model_responses_with_judge_decision,
        ) = self.evaluate_chunking_methods(queries=queries)
        save_responses(responses_with_judge_decision)
        save_prompt_responses(model_responses_with_judge_decision)
        print(
            f"Better chunking method response counts: {self.better_model_response_counts}"
        )
        print(f"Better prompt response counts: {self.better_prompt_response_counts}")


args = argparse.Namespace(
    save_dir="/home/vscode/projects/llm/hackday_life_raft/models/evaluation",
    data_dir="/home/vscode/projects/llm/hackday_life_raft/datasets/2023_06_25__19_55_UTC/",
    current_chunk_size=350,
    current_chunk_overlap=0,
    previous_chunk_size=350,
    previous_chunk_overlap=0,
)
queries = [
    "what is a guest card?",
    "how can i transfer a security deposit?",
    "Why dont i have the option to refund my tenant?",
    "What are the core components of the owner statement?",
    "What texting  options does Appfolio support?",
    "Do eChecks have a fee?",
    "Can I pay a bill with cash?",
    "Does Appfolio support sending leases through blue moon?",
]
evaluator = ChunkEvaluator(args)
evaluator.build_vectorstores()
evaluator.run(queries=queries)
