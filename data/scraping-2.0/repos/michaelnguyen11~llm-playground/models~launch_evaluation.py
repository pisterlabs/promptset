import os
import pandas as pd
import argparse
import openai
from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import OpenAI

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

from utils import load_documents

openai.api_key = os.environ["OPENAI_API_KEY"]

def evaluation_with_ragas(
        service_context: ServiceContext,
        documents: SimpleDirectoryReader,
        eval_questions_file: str):
    """
    Evaluate the model with Ragas
    """
    questions = []
    with open(eval_questions_file, "r") as f:
        for line in f:
            questions.append(line.strip())

    index = VectorStoreIndex.from_documents(
        documents=documents,
        service_context=service_context
    )

    query_engine = index.as_query_engine(similarity_top_k=2)

    contexts = []
    answers = []

    for question in questions:
        response = query_engine.query(question)
        contexts.append([x.node.get_content() for x in response.source_nodes])
        answers.append(str(response))

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
    )

    result = evaluate(ds, [answer_relevancy, faithfulness])

    return result


def evaluate_gpt_model(
        documents: SimpleDirectoryReader,
        model_name: str,
        eval_questions_file: str):
    """
    Evaluate the GPT models with Ragas
    """
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(model=model_name, temperature=0.3),
        context_window=2048, # limit the context window to 2048 tokens so that refine is used
    )

    result = evaluation_with_ragas(service_context, documents, eval_questions_file)

    return result


def get_question(questions_file: str, question_number: str):
    """
    Get question from questions file
    """
    questions = []
    with open(questions_file, "r") as f:
        for line in f:
            questions.append(line.strip())

    question = questions[question_number]

    return question


def get_response(
        documents: SimpleDirectoryReader,
        model_name: str,
        question: int):
    """
    Get response from model from a specific question
    """
    index = VectorStoreIndex.from_documents(documents)
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(model=model_name, temperature=0.3),
        context_window=2048,  # limit the context window artifically to test refine process
    )
    query_engine = index.as_query_engine(service_context=service_context)
    answer = query_engine.query(question)

    return answer


def parge_args():
    parser = argparse.ArgumentParser(description="Dataset preparation for fine-tuning")
    parser.add_argument("-eb", "--eval_baseline", action="store_true",
                        help="Evaluate baseline for GPT-3.5-turbo model")
    parser.add_argument("-ef", "--eval_finetuned", action="store_true",
                        help="Evaluate GPT-3.5-turbo fine tuned model")
    parser.add_argument("-e4", "--eval_gpt4", action="store_true",
                        help="Evaluate GPT-4 model")
    parser.add_argument("-ea", "--eval_all", action="store_true",
                        help="Evaluate all models : based GPT-3.5-turbo, finetuned GPT-3.5-turbo and GPT-4 model")
    parser.add_argument("-cr", "--compare_response", action="store_true",
                        help="Compare different responses between GPT-3.5-turbo baseline, finetuned and GPT-4 models")
    parser.add_argument("-vp", "--val_path", type=str, default="datasets/eval_questions_gpt4_generate.txt",
                        help="Path to save val questions in .txt format")
    parser.add_argument("-rf", "--response_file", type=str, default="compare_responses.csv",
                        help="Save compare responses between models in .csv format")
    args = parser.parse_args()

    return args

def main():
    documents = load_documents(["docs/Generative_Agents_Interactive_Simulacra_of_Human_Behavior.pdf"])

    gpt_35_baseline = 'gpt-3.5-turbo-1106'
    # The fine-tuned model trained with train questions generated from GPT-3.5-turbo-1106
    # gpt_35_tuned = 'ft:gpt-3.5-turbo-1106:aitomatic-inc:hiep:8clVbyfK'
    # The fine-tuned model trained with train questions generated from GPT-4
    gpt_35_tuned = 'ft:gpt-3.5-turbo-1106:aitomatic-inc:hiep:8cuu5f77'
    # gpt_35_tuned = 'ft:gpt-3.5-turbo-1106:personal::8eY9TPw4'
    gpt_4_baseline = 'gpt-4-1106-preview'

    args = parge_args()

    if args.eval_baseline:
        gpt_35_baseline_result = evaluate_gpt_model(documents=documents,
                                                    model_name=gpt_35_baseline,
                                                    eval_questions_file=args.val_path)
        print('Evaluation model {} with Ragas results : {}'.format(gpt_35_baseline, gpt_35_baseline_result))
    elif args.eval_finetuned:
        gpt_35_tuned_result = evaluate_gpt_model(documents=documents,
                                                 model_name=gpt_35_tuned,
                                                 eval_questions_file=args.val_path)
        print('Evaluation model {} with Ragas results : {}'.format(gpt_35_tuned, gpt_35_tuned_result))
    elif args.eval_gpt4:
        # Not neccessary to evaluate GPT-4 model because it is not fine-tuned
        gpt_4_baseline_result = evaluate_gpt_model(documents=documents,
                                                   model_name=gpt_4_baseline,
                                                   eval_questions_file=args.val_path)
        print('Evaluation model {} with Ragas results : {}'.format(gpt_4_baseline, gpt_4_baseline_result))
    elif args.eval_all:
        gpt_35_baseline_result = evaluate_gpt_model(documents=documents,
                                                    model_name=gpt_35_baseline,
                                                    eval_questions_file=args.val_path)
        print('Evaluation model {} with Ragas results : {}'.format(gpt_35_baseline, gpt_35_baseline_result))
        gpt_35_tuned_result = evaluate_gpt_model(documents=documents,
                                                 model_name=gpt_35_tuned,
                                                 eval_questions_file=args.val_path)
        print('Evaluation model {} with Ragas results : {}'.format(gpt_35_tuned, gpt_35_tuned_result))
        gpt_4_baseline_result = evaluate_gpt_model(documents=documents,
                                                   model_name=gpt_4_baseline,
                                                   eval_questions_file=args.val_path)
        print('Evaluation model {} with Ragas results : {}'.format(gpt_4_baseline, gpt_4_baseline_result))

        # Save to DF
        eval_df = pd.DataFrame(
            {
                "Model Name": [gpt_35_baseline, gpt_35_tuned, gpt_4_baseline],
                "Ragas Answer Selevancy Score": [gpt_35_baseline_result['answer_relevancy'],
                                                 gpt_35_tuned_result['answer_relevancy'],
                                                 gpt_4_baseline_result['answer_relevancy']],
                "Ragas Faithfulness Score": [gpt_35_baseline_result['faithfulness'],
                                             gpt_35_tuned_result['faithfulness'],
                                             gpt_4_baseline_result['faithfulness']],
            },
        )
        eval_df.to_csv('evaluate_all_models.csv', index=False)
        print(eval_df)
    elif args.compare_response:
        question = get_question(questions_file=args.val_path, question_number=12)
        gpt_35_answer = get_response(documents=documents,
                                    model_name=gpt_35_baseline,
                                    question=question)
        gpt_35_tuned_answer = get_response(documents=documents,
                                        model_name=gpt_35_tuned,
                                        question=question)
        gpt_4_baseline_answer = get_response(documents=documents,
                                            model_name=gpt_4_baseline,
                                            question=question)

        # Let's quickly compare the differences in responses,
        # to demonstrate that fine tuning did indeed change something.
        eval_df = pd.DataFrame(
            {
                "Question": question,
                "Model Name": [gpt_35_baseline, gpt_35_tuned, gpt_4_baseline],
                "Answer": [gpt_35_answer, gpt_35_tuned_answer, gpt_4_baseline_answer],
            },
        )

        eval_df.to_csv(args.response_file, index=False)
        print(eval_df)


if __name__ == '__main__':
    main()
