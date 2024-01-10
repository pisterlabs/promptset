import os
from datasets import get_dataset
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from utils.timer import Timer
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from config import api_key, load_config
import wandb
import langchain
langchain.verbose = True

# If you don't want your script to sync to the cloud
os.environ["WANDB_MODE"] = "offline"

CLASSIFIY_PROMPT = """
You are a text-to-SQL expert able to identify poorly formulated questions in natural language.
The dataset used is consisting of questions and their corresponding golden SQL queries. You will be given the database schema of the database corresponding to the question.
Furthermore, you will also be given a hint that provides additional information that is needed to correctly convert the question and interpret the database schema.  
However, some of the questions in the data are poorly formulated or contain errors. 

Below is a classification scheme for the questions that are to be converted into SQL queries. 

0 = Correct question. May still contain minor errors in language or minor ambiguities that do not affect the interpretation and generation of the SQL query
1 = Is unclear, ambiguous, unspecific or contain grammatical errors that surely is going to affect the interpretation and generation of the SQL query. The question 
is unspecific in which columns that are to be returned. The question is not asking for a specific column, but asks generally about a table in the database.
2 = The question contains minor errors in language or minor ambiguities that might affect the interpretation and generation of the SQL query.
3 = The question is wrongly formulated when considering the structure of the database schema. The information that the question is asking for is not possible to accurately retrieve from the database.

Here are some examples of questions that would be classified with 1 and an explanation of why:

Example 1: List the customer who made the transaction id 3682978
Explanation: The question is unspecific in which columns that are to be returned. It asks to list the customers, but does not specify which columns that are to be returned from the client table. 

Example 2: Which district has the largest amount of female clients?
Explanation: The question is unspecific in which columns that are to be returned. It asks "which district", but does not specify which columns that are to be returned from the district table. 

Example 3: What is the average amount of transactions done in the year of 1998 ?
Explanation: Is unclear, ambiguous, unspecific or contain grammatical errors that surely is going to affect the interpretation and generation of the SQL query.

Here is an example of a question that would be classified with 2 and an explanation of why:

Example 1: What are the top 5 loans by region names for the month of Mars 1997?
Explanation: The statement 'top 5' could be ambiguous. It could mean the top 5 loans by amount or the top 5 loans by number of loans.

Here are some examples of questions that would be classified with 3 and an explanation of why:

Example 1: What is the disposition id of the oldest client in the Prague region?
Explanation: The question is wrongly formulated when considering the structure of the database schema. There can be multiple disposition ids for a client, 
since a client can have multiple accounts. The question is not asking for a specific disposition id, but asks generally about a client.


Here are some examples of questions that would be classified with 0 and an explanation of why:

Example 1: List the id of the customer who made the transaction id : 3682978
Explanation: Clear and correct question.

Example 2: What is the name of the district that has the largest amount of female clients?
Explanation: Specific and  correct question.

Example 3: What is the disposition id(s) of the oldest client in the Prague region?
Explanation: The question is open for disposition ids which is correct when considering the sql-schema.

Example 4: What was the average number of withdrawal transactions conducted by female clients from the Prague region during the year 1998?
Explanation: Clear and correct question.

Database schema: 
{database_schema}

Hint:
{evidence}

Below you will be provided with the correct SQL-query that represents what the questions is trying to ask for.

Gold query: 
{gold_query}

Please classify the question below according to the classification scheme above, the examples, the hint and the SQL gold query provided.
Also please assume that all dates, values, names and numbers in the questions are correct. 

Question: 
{question}

In your answer DO NOT return anything else than the mark as a sole number. Do not return any corresponding text or explanations. 
"""

#1 = Gray area, minor errors that may or may not affect the interpretation and generation of the SQL query.


class Classifier():
    total_tokens = 0
    prompt_tokens = 0 
    total_cost = 0
    completion_tokens = 0
    last_call_execution_time = 0
    total_call_execution_time = 0

    def __init__(self, llm):        
        self.llm = llm

        self.prompt_template = CLASSIFIY_PROMPT
        prompt = PromptTemplate(    
            # input_variables=["question", "database_schema","evidence"],
            input_variables=["question", "database_schema", "evidence", 'gold_query'],
            template=CLASSIFIY_PROMPT,
        )

        self.chain = LLMChain(llm=llm, prompt=prompt)


    def classify_question(self, question, schema, evidence, gold_query):
        with get_openai_callback() as cb:
            with Timer() as t:
                response = self.chain.run({
                    'question': question,
                    'database_schema': schema,
                    'evidence': evidence,
                    'gold_query': gold_query
                })

            logging.info(f"OpenAI API execution time: {t.elapsed_time:.2f}")
            
            self.last_call_execution_time = t.elapsed_time
            self.total_call_execution_time += t.elapsed_time
            self.total_tokens += cb.total_tokens
            self.prompt_tokens += cb.prompt_tokens
            self.total_cost += cb.total_cost
            self.completion_tokens += cb.completion_tokens

            return response


accepted_faults = [1, 3]

def main():
    config = load_config("classifier_config.yaml")

    wandb.init(
        project=config.project,
        config=config,
        name=config.current_experiment,
        entity=config.entity
    )

    artifact = wandb.Artifact('experiment_results', type='dataset')
    table = wandb.Table(columns=["Question", "Classified_quality", "Difficulty"]) ## Är det något mer vi vill ha med här?
    wandb_cm = wandb.Table(columns=['0', '1', '2', '3'])
    metrics_table = wandb.Table(columns=["Class", "Precision", "Recall", "F1 Score", "Accuracy"])
    weighted_avg_table = wandb.Table(columns=["Metric", "Weighted Average"])
    # "Weighted Averages", weighted_averages['precision'], weighted_averages['recall'], weighted_averages['f1'], weighted_averages['accuracy']


    llm = ChatOpenAI(
        openai_api_key=api_key, 
        model_name=config.llm_settings.model,
        temperature=config.llm_settings.temperature,
        request_timeout=config.llm_settings.request_timeout
    )

    dataset = get_dataset("BIRDCorrectedFinancialGoldAnnotated")
    classifier = Classifier(llm)

    wandb.config['prompt'] = classifier.prompt_template

    no_data_points = dataset.get_number_of_data_points()

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    confusion_matrix = np.zeros((4,4))
    annotation_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for i in range(no_data_points):
        data_point = dataset.get_data_point(i)
        evidence = data_point['evidence']
        db_id = data_point['db_id']            
        question = data_point['question']
        gold_query = data_point['SQL']
        difficulty = data_point['difficulty'] if 'difficulty' in data_point else ""
        annotated_question_quality = data_point["annotation"]
        
        sql_schema = dataset.get_schema_and_sample_data(db_id)

        classified_quality = classifier.classify_question(question, sql_schema, evidence, gold_query)
        classified_quality = int(classified_quality) if classified_quality.isdigit() else None

        print('classified_quality: ',classified_quality)

        if classified_quality is not None:
            for annotated_quality in annotated_question_quality:  
                annotation_counts[annotated_quality] +=1
                confusion_matrix[annotated_quality][classified_quality] += 1
                


    print('confusion matrix:')
    print(confusion_matrix)
    # Converting to integer
    confusion_matrix = np.array(confusion_matrix).astype(int)
    
    print('annotation counts: ',annotation_counts)
    labels = [0, 1, 2, 3] 
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="YlOrRd", xticklabels=labels, yticklabels=labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.savefig(f'{config.current_experiment}_heatmap.png')

    wandb.log({"confusion_matrix_heatmap": wandb.Image(f'{config.current_experiment}_heatmap.png')})

    metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0}
    weighted_sums = {'precision': 0, 'recall': 0, 'f1': 0, 'accuracy': 0}
    total_instances = np.sum(confusion_matrix)

    for i in range(4):
        row_data = confusion_matrix[i].tolist()
        print('row_data: ', row_data)
        wandb_cm.add_data(*row_data)
        tp = confusion_matrix[i][i]
        fp = sum(confusion_matrix[:, i]) - tp
        fn = sum(confusion_matrix[i, :]) - tp
        tn = np.sum(confusion_matrix) - (tp + fp + fn)

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        metrics[i] = {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}
        metrics_table.add_data(i, metrics[i]['precision'], metrics[i]['recall'], metrics[i]['f1'], metrics[i]['accuracy'])

        class_weight = sum(confusion_matrix[i, :])
        weighted_sums['precision'] += precision * class_weight
        weighted_sums['recall'] += recall * class_weight
        weighted_sums['f1'] += f1 * class_weight
        weighted_sums['accuracy'] += accuracy * class_weight

        print('metrics for class ', i, ': ', metrics[i])




        # metrics now contains the precision, recall, and F1-score for each category
            # annotated_question_qualities = set(annotated_question_quality)
            # if classified_quality.isdigit() and int(classified_quality) == 1:            
            #     if any(element in annotated_question_qualities for element in accepted_faults):
            #         tp += 1
            #     else:
            #         fp += 1
            # elif classified_quality.isdigit() and int(classified_quality) == 0:
            #     if any(element in annotated_question_qualities for element in accepted_faults):
            #         fn += 1
            #     else:
            #         tn += 1
            
            # precision = tp / (tp + fp)
            # recall = tp / (tp + fn)
            # f1 = 2 * ((precision * recall) / (precision + recall))
            # accuracy = (tp + tn) / (tp + tn + fp + fn)

        table.add_data(question, classified_quality, difficulty)
        wandb.log({                      
            "total_tokens": classifier.total_tokens,
            "prompt_tokens": classifier.prompt_tokens,
            "completion_tokens": classifier.completion_tokens,
            "total_cost": classifier.total_cost,
            "openAPI_call_execution_time": classifier.last_call_execution_time,
        }, step=i+1)
    
        print("Predicted quality: ", classified_quality, " Annotated quality: ", " ".join(map(str, annotated_question_quality)))
        
    weighted_averages = {metric: total / total_instances for metric, total in weighted_sums.items()}

    print("Weighted Averages:", weighted_averages)
    
    # weighted_avg_table.add_data("Weighted Averages", weighted_averages['precision'], weighted_averages['recall'], weighted_averages['f1'], weighted_averages['accuracy'])
    weighted_avg_table.add_data("Precision", weighted_averages['precision'])
    weighted_avg_table.add_data("Recall", weighted_averages['recall'])
    weighted_avg_table.add_data("F1 Score", weighted_averages['f1'])
    weighted_avg_table.add_data("Accuracy", weighted_averages['accuracy'])



    wandb.run.summary["total_tokens"]                       = classifier.total_tokens
    wandb.run.summary["prompt_tokens"]                      = classifier.prompt_tokens
    wandb.run.summary["completion_tokens"]                  = classifier.completion_tokens
    wandb.run.summary["total_cost"]                         = classifier.total_cost
    wandb.run.summary['total_predicted_execution_time']     = dataset.total_predicted_execution_time
    wandb.run.summary['total_openAPI_execution_time']       = classifier.total_call_execution_time

    
    artifact.add(wandb_cm, "ConfusionMatrix_predictions")
    artifact.add(table, "query_results")
    artifact.add(metrics_table, "metrics")
    artifact.add(weighted_avg_table, "weighted_averages_metric_table")
    wandb.log_artifact(artifact)

    artifact_code = wandb.Artifact('code', type='code')
    artifact_code.add_file("src/run_classifier.py")
    wandb.log_artifact(artifact_code)

    wandb.finish()



if __name__ == "__main__":
    main()