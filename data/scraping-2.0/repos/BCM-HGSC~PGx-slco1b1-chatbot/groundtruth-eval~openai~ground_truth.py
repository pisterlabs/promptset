# -*- coding:utf-8 -*-
# Created by liwenw at 10/12/23

from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import os
import pandas as pd
import ast

from chromadb.config import Settings
from omegaconf import OmegaConf
import re
import argparse


def create_parser():
    parser = argparse.ArgumentParser(description='demo how to use ai embeddings to chat.')
    parser.add_argument("-y", "--yaml", dest="yamlfile",
                        help="Yaml file for project", metavar="YAML")
    return parser

def evaluate_conditions(conditions,answer):
    '''
    Ex - condition = [[str1,str2],[str3,str4]]
    evaluates to (str1 and str2) or (str3 and str4)
    '''
    boolean_list = []
    for condition in conditions:
        if len(condition) > 1:
            if all(re.search(re.escape(c), answer, re.IGNORECASE) for c in condition):
                boolean_list.append(True)
            else:
                boolean_list.append(False)
        else:
            if re.search(re.escape(condition[0] ), answer, re.IGNORECASE):
                boolean_list.append(True)
            else:
                boolean_list.append(False)
    result = any(boolean_list)
    return result

def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.yamlfile is None:
        parser.print_help()
        exit()

    yamlfile = args.yamlfile
    config = OmegaConf.load(yamlfile)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        openai_api_key = config.openai.api_key

    question_file = config.validation_file
    test_questions = pd.read_csv(question_file)
    df_final_summary = test_questions.groupby('category').agg({'question': 'count'}).reset_index()
    df_final_individual_summary = test_questions

    embeddings = OpenAIEmbeddings()
    collection_name = config.chromadb.collection_name
    persist_directory = config.chromadb.persist_directory
    chroma_db_impl = config.chromadb.chroma_db_impl
    vector_store = Chroma(collection_name=collection_name,
                          embedding_function=embeddings,
                          client_settings=Settings(
                              chroma_db_impl=chroma_db_impl,
                              persist_directory=persist_directory
                          ),
                          )

    for k in [5, 10, 20]:
        for chat_search_type in ['mmr', 'similarity']:
            print(f"chat_search_type: {chat_search_type}")
            print(f"chat_search_k: {k}")
            retriever = vector_store.as_retriever(search_type=chat_search_type, search_kwargs={"k": k})

            output_summary = f"output_summary_{chat_search_type}_{k}.csv"
            output_pagecontent = f"output_pagecontent_{chat_search_type}_{k}.txt"
            final_summary = "final_summary.csv"
            output_pagecontent_handler = open(output_pagecontent, "w")

            evaluation_dict_final = {}

            for q, a in zip(test_questions['question'], test_questions['answer']):
                output_pagecontent_handler.write(f"question: {q}\n")
                # set up  data structure
                evaluation_per_question = []

                # retrieve docs from vector store
                retrieved_docs = {}
                docs = retriever.get_relevant_documents(q)
                output_pagecontent_handler.write(f"retrieved_docs: {docs}\n\n")

                for idx, doc in enumerate(docs):
                    retrieved_docs[idx + 1] = [doc.page_content]

                # evaluate conditions on retrieved docs
                conditions = ast.literal_eval(a)
                for key, item in retrieved_docs.items():
                    evaluate_response = evaluate_conditions(conditions, item[0])
                    evaluation_per_question.append(evaluate_response)
                evaluation_dict_final[q] = evaluation_per_question

            # create dataframe for evaluation results and write to csv
            name_columns = ['question', 'category']
            num_columns = [i for i in range(0, k)]
            columns = name_columns + num_columns

            df_evaluation = pd.DataFrame(evaluation_dict_final)
            df_evaluation = df_evaluation.T
            df_evaluation = df_evaluation.reset_index().merge(test_questions[['question', 'category']],
                                                              left_on="index", right_on="question")[columns]
            df_evaluation['total_matches'] = df_evaluation[num_columns].astype(int).sum(axis=1)
            df_evaluation['success'] = (df_evaluation['total_matches'] >= 1).astype(int)
            df_final_individual_summary[f"success_{chat_search_type}_{k}"] = df_evaluation['success']

            df_evaluation.to_csv(output_summary)

            output_pagecontent_handler.close()
            df_interm_summary = df_evaluation.groupby('category').agg({'success': 'mean', 'question': 'count'}).reset_index()
            df_final_summary[f"{chat_search_type}_{k}"] = df_interm_summary['success']

    # write summary final results
    df_final_individual_summary.to_csv("final_individual_summary.csv", index=False)
    df_final_summary.to_csv(final_summary)
    print(df_final_summary)


if __name__ == "__main__":
    main()



