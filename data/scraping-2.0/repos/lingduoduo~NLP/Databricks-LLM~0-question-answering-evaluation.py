# Databricks notebook source
# MAGIC %md
# MAGIC # LLM Evaluation with MLflow example
# MAGIC
# MAGIC This notebook demonstrates how to evaluate various LLMs and RAG systems with MLflow, leveraging simple metrics such as perplexity and toxicity, as well as LLM-judged metrics such as relevance, and even custom LLM-judged metrics such as professionalism.
# MAGIC
# MAGIC For details about how to use `mlflow.evaluate()`, refer to Evaluate LLMs with MLflow ([AWS](https://docs.databricks.com/en/mlflow/llm-evaluate.html)|[Azure](https://learn.microsoft.com/azure/databricks/mlflow/llm-evaluate)).
# MAGIC
# MAGIC ## Requirements
# MAGIC  
# MAGIC To use the MLflow LLM evaluation feature, you must use MLflow flavor 2.8.0 or above.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup Databricks Runtime
# MAGIC - If you are using a cluster running Databricks Runtime, you must install the mlflow library from PyPI. 
# MAGIC - If you are using a cluster running Databricks Runtime ML, the mlflow library is already installed.

# COMMAND ----------

# MAGIC %md
# MAGIC Install the mlflow library. This is required for Databricks Runtime clusters only. If you are using a cluster running Databricks Runtime ML, skip to Set OpenAI Key step.

# COMMAND ----------

# If you are running Databricks Runtime version 7.1 or above, uncomment this line and run this cell:
%pip install mlflow
# If you are running Databricks Runtime version 6.4 to 7.0, uncomment this line and run this cell:
#dbutils.library.installPyPI("mlflow")

# COMMAND ----------

import mlflow

# COMMAND ----------

# MAGIC %pip install --upgrade typing_extensions
# MAGIC %pip install openai 

# COMMAND ----------

import openai

# COMMAND ----------

# MAGIC %md
# MAGIC Import the required libraries.

# COMMAND ----------

import os
import pandas as pd

## Check yout MLflow version
mlflow.__version__

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set OpenAI Key

# COMMAND ----------

# os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope="your-scope", key="your-secret-key")
os.environ["OPENAI_API_KEY"] = '-'
# Uncomment below, if using Azure OpenAI
# os.environ["OPENAI_API_TYPE"] = "azure"
# os.environ["OPENAI_API_VERSION"] = "2023-05-15"
# os.environ["OPENAI_API_BASE"] = "https://<>.<>.<>.com/"
# os.environ["OPENAI_DEPLOYMENT_NAME"] = "deployment-name"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Question-Answering Evaluation

# COMMAND ----------

# MAGIC %md
# MAGIC Create a test case of `inputs` that is passed into the model and `ground_truth` which is used to compare against the generated output from the model.

# COMMAND ----------

eval_df = pd.DataFrame(
    {
        "inputs": [
            "How does useEffect() work?",
            "What does the static keyword in a function mean?",
            "What does the 'finally' block in Python do?",
            "What is the difference between multiprocessing and multithreading?",
        ],
        "ground_truth": [
            "The useEffect() hook tells React that your component needs to do something after render. React will remember the function you passed (we’ll refer to it as our “effect”), and call it later after performing the DOM updates.",
            "Static members belongs to the class, rather than a specific instance. This means that only one instance of a static member exists, even if you create multiple objects of the class, or if you don't create any. It will be shared by all objects.",
            "'Finally' defines a block of code to run when the try... except...else block is final. The finally block will be executed no matter if the try block raises an error or not.",
            "Multithreading refers to the ability of a processor to execute multiple threads concurrently, where each thread runs a process. Whereas multiprocessing refers to the ability of a system to run multiple processors in parallel, where each processor can run one or more threads.",
        ],
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC Create a simple OpenAI model that asks gpt-3.5 to answer the question in two sentences. Call `mlflow.evaluate()` with the model and evaluation dataframe. 

# COMMAND ----------

with mlflow.start_run() as run:
    system_prompt = "Answer the following question in two sentences"
    basic_qa_model = mlflow.openai.log_model(
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        artifact_path="model",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "{question}"},
        ],
    )
    results = mlflow.evaluate(
        basic_qa_model.model_uri,
        eval_df,
        targets="ground_truth",  # specify which column corresponds to the expected output
        model_type="question-answering",  # model type indicates which metrics are relevant for this task
        evaluators="default",
    )
results.metrics

# COMMAND ----------

# MAGIC %md
# MAGIC Inspect the evaluation results table as a dataframe to see row-by-row metrics to further assess model performance

# COMMAND ----------

results.tables["eval_results_table"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLM-judged correctness with OpenAI GPT-4

# COMMAND ----------

# MAGIC %md
# MAGIC Construct an answer similarity metric using the `answer_similarity()` metric factory function.

# COMMAND ----------

from mlflow.metrics.genai import EvaluationExample, answer_similarity

# Create an example to describe what answer_similarity means like for this problem.
example = EvaluationExample(
    input="What is MLflow?",
    output="MLflow is an open-source platform for managing machine "
    "learning workflows, including experiment tracking, model packaging, "
    "versioning, and deployment, simplifying the ML lifecycle.",
    score=4,
    justification="The definition effectively explains what MLflow is "
    "its purpose, and its developer. It could be more concise for a 5-score.",
    grading_context={
        "targets": "MLflow is an open-source platform for managing "
        "the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, "
        "a company that specializes in big data and machine learning solutions. MLflow is "
        "designed to address the challenges that data scientists and machine learning "
        "engineers face when developing, training, and deploying machine learning models."
    },
)

# Construct the metric using OpenAI GPT-4 as the judge
answer_similarity_metric = answer_similarity(model="openai:/gpt-4", examples=[example])

print(answer_similarity_metric)

# COMMAND ----------

# MAGIC %md
# MAGIC Call `mlflow.evaluate()` again but with your new `answer_similarity_metric`

# COMMAND ----------

with mlflow.start_run() as run:
    results = mlflow.evaluate(
        basic_qa_model.model_uri,
        eval_df,
        targets="ground_truth",
        model_type="question-answering",
        evaluators="default",
        extra_metrics=[answer_similarity_metric],  # use the answer similarity metric created above
    )
results.metrics

# COMMAND ----------

# MAGIC %md
# MAGIC See the row-by-row LLM-judged answer similarity score and justifications

# COMMAND ----------

results.tables["eval_results_table"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Custom LLM-judged metric for professionalism

# COMMAND ----------

# MAGIC %md
# MAGIC Create a custom metric that is used to determine professionalism of the model outputs. Use `make_genai_metric` with a metric definition, grading prompt, grading example, and judge model configuration

# COMMAND ----------

from mlflow.metrics.genai import EvaluationExample, make_genai_metric

professionalism_metric = make_genai_metric(
    name="professionalism",
    definition=(
        "Professionalism refers to the use of a formal, respectful, and appropriate style of communication that is tailored to the context and audience. It often involves avoiding overly casual language, slang, or colloquialisms, and instead using clear, concise, and respectful language"
    ),
    grading_prompt=(
        "Professionalism: If the answer is written using a professional tone, below "
        "are the details for different scores: "
        "- Score 1: Language is extremely casual, informal, and may include slang or colloquialisms. Not suitable for professional contexts."
        "- Score 2: Language is casual but generally respectful and avoids strong informality or slang. Acceptable in some informal professional settings."
        "- Score 3: Language is balanced and avoids extreme informality or formality. Suitable for most professional contexts. "
        "- Score 4: Language is noticeably formal, respectful, and avoids casual elements. Appropriate for business or academic settings. "
        "- Score 5: Language is excessively formal, respectful, and avoids casual elements. Appropriate for the most formal settings such as textbooks. "
    ),
    examples=[
        EvaluationExample(
            input="What is MLflow?",
            output=(
                "MLflow is like your friendly neighborhood toolkit for managing your machine learning projects. It helps you track experiments, package your code and models, and collaborate with your team, making the whole ML workflow smoother. It's like your Swiss Army knife for machine learning!"
            ),
            score=2,
            justification=(
                "The response is written in a casual tone. It uses contractions, filler words such as 'like', and exclamation points, which make it sound less professional. "
            ),
        )
    ],
    version="v1",
    model="openai:/gpt-4",
    parameters={"temperature": 0.0},
    grading_context_columns=[],
    aggregations=["mean", "variance", "p90"],
    greater_is_better=True,
)

print(professionalism_metric)

# COMMAND ----------

# MAGIC %md
# MAGIC Call `mlflow.evaluate` with your new professionalism metric. 

# COMMAND ----------

with mlflow.start_run() as run:
    results = mlflow.evaluate(
        basic_qa_model.model_uri,
        eval_df,
        model_type="question-answering",
        evaluators="default",
        extra_metrics=[professionalism_metric],  # use the professionalism metric we created above
    )
print(results.metrics)

# COMMAND ----------

results.tables["eval_results_table"]

# COMMAND ----------

# MAGIC %md
# MAGIC Lets see if we can improve `basic_qa_model` by creating a new model that could perform better by changing the system prompt.

# COMMAND ----------

# MAGIC %md
# MAGIC Call `mlflow.evaluate()` using the new model. Observe that the professionalism score has increased!

# COMMAND ----------

with mlflow.start_run() as run:
    system_prompt = "Answer the following question using extreme formality."
    professional_qa_model = mlflow.openai.log_model(
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        artifact_path="model",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "{question}"},
        ],
    )
    results = mlflow.evaluate(
        professional_qa_model.model_uri,
        eval_df,
        model_type="question-answering",
        evaluators="default",
        extra_metrics=[professionalism_metric],
    )
print(results.metrics)

# COMMAND ----------

results.tables["eval_results_table"]

# COMMAND ----------


