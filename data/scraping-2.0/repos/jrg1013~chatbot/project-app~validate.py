# validate.py

import utils
import cfg
import langchain
from datetime import datetime

# import pandas
import pandas as pd

# Documents loaders
from langchain import document_loaders

# Prompts
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Evaluation
from langchain.evaluation.qa import QAGenerateChain, QAEvalChain
from langchain.llms import HuggingFaceHub
from langchain.evaluation import load_evaluator


# Get llm
llm = utils.get_llm()

# Import vectordb
vectordb = utils.import_vectordb()

# Get our customized prompt
prompt = utils.get_prompt()

# Generate a question and asnwer based on our RAG
qa_chain = utils.get_qa_chain(prompt, vectordb)

# Import examples
# read csv
examples = pd.read_csv(
    './documents/Preguntas-Respuestas - ONLINE.csv', delimiter=";", header=0, names=["intent", "query", "answer"])
# Convert the DataFrame to a Dictionary
examples.drop('intent', inplace=True, axis=1)
examples = examples.iloc[::5]
examples = examples.to_dict(orient='records')

# Manual evaluation
langchain.debug = True
qa_chain.run(examples[0]["query"])
# Turn off the debug mode
langchain.debug = False

# Automatic QA
predictions = []

for e in examples:
    qa_chain = utils.get_qa_chain(prompt, vectordb)
    print(e["query"])
    e["result"] = qa_chain.run(e["query"])
    predictions.append(e)

eval_chain = QAEvalChain.from_llm(llm)
graded_outputs = eval_chain.evaluate(examples, predictions)

print(graded_outputs)

for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]["query"] + "\n")
    print("Real Answer: " + predictions[i]["answer"] + "\n")
    print("Predicted Answer: " + predictions[i]["result"] + "\n")
    print("Predicted Grade: " + graded_outputs[i]["results"] + "\n")
    print()

counter = 0

for i in range(len(graded_outputs)):
    if str(graded_outputs[i]["results"]).__contains__("CORRECT"):
        counter += 1

with open("./temp/"+str(counter)+"_Validation_"+str(datetime.now()), "x") as f:
    f.write("Number of correct answers:" + str(counter) + "\n\n\n")

    # Print cfg values
    f.write("################" + "\n")
    f.write("#    cfg.py    #" + "\n")
    f.write("################" + "\n\n")
    f.write("model_name : " + str(cfg.model_name) + "\n")
    f.write("temperature : " + str(cfg.temperature) + "\n")
    f.write("top_p = " + str(cfg.top_p) + "\n")
    f.write("repetition_penalty = " + str(cfg.repetition_penalty) + "\n")
    f.write("do_sample = " + str(cfg.do_sample) + "\n")
    f.write("max_new_tokens = " + str(cfg.max_new_tokens) + "\n")
    f.write("num_return_sequences = " + str(cfg.num_return_sequences) + "\n")
    f.write("split_chunk_size : " + str(cfg.split_chunk_size) + "\n")
    f.write("split_overlap : " + str(cfg.split_overlap) + "\n")
    f.write("embeddings_model_repo : " + str(cfg.embeddings_model_repo) + "\n")
    f.write("template : " + str(cfg.template) + "\n")

    # Print results of the QA Tests
    f.write("################" + "\n")
    f.write("#   Examples   #" + "\n")
    f.write("################" + "\n\n")
    for i, eg in enumerate(examples):
        f.write(f"Example {i}:")
        f.write("Question: " + predictions[i]["query"] + "\n")
        f.write("Real Answer: " + predictions[i]["answer"] + "\n")
        f.write("Predicted Answer: " + predictions[i]["result"] + "\n")
        f.write("Predicted Grade: " + graded_outputs[i]["results"] + "\n\n")

    # Close txt file
    f.close
