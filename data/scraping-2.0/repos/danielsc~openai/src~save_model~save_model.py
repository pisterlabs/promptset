from pandas.core.frame import DataFrame
import evaluate, os, sys

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code")
# Add the module directory to the Python path
sys.path.append(code_dir)

from openai_model import OpenAIModel
import shutil, mlflow
import mlflow.models 
import argparse
import pandas as pd
import yaml, os
import numpy as np
import tempfile


def load_yaml(filename):
   with open(filename, encoding='utf-8') as fh:
      file_dict = yaml.load(fh, Loader=yaml.FullLoader)
   return file_dict

def save_yaml(content, filename):
   with open(filename, encoding='utf-8', mode='w') as fh:
      yaml.dump(content, fh)

def empty_folder(path):
  import os
  import shutil

  for root, dirs, files in os.walk(path):
      for f in files:
          os.unlink(os.path.join(root, f))
      for d in dirs:
          shutil.rmtree(os.path.join(root, d))
         
def save_pyfunc_model(deployment: str, path: str, model_input: pd.DataFrame, model_output: np.array, scoring_parameters: dict):
  basefolder = tempfile.mkdtemp()
  artifact_path = basefolder + "/MLArtifact.yaml"
  deployment_dict = load_yaml(deployment)
  deployment_dict["scoring_parameters"] = scoring_parameters
  save_yaml(deployment_dict, artifact_path)

  artifacts = {
    'deployment': artifact_path,
  }
  model = OpenAIModel()

  signature = mlflow.models.infer_signature(model_input=model_input, model_output=model_output)

  # need to empty the folder first
  empty_folder(path)

  mlflow.pyfunc.save_model(path, python_model=model, artifacts=artifacts,
    code_path=model.source_paths(), conda_env=model.conda_path(), signature=signature)

def calculate_metrics(references, predictions):
  print("Calculating metrics")
  f1 = evaluate.load("f1").compute(references=references, predictions=predictions, average='weighted')
  accuracy = evaluate.load("accuracy").compute(references=references, predictions=predictions)
  precision = evaluate.load("precision").compute(references=references, predictions=predictions, average='weighted')
  recall = evaluate.load("recall").compute(references=references, predictions=predictions, average='weighted')
  # merge the dicts
  metrics = {**f1, **accuracy, **precision, **recall}
  metrics = {"test_" + k: v for k, v in metrics.items()}
  return metrics

def save_and_score(deployment, model_path, test_data, prompt_column, completion_column, scoring_parameters):
  
  df = pd.read_csv(test_data)
  prompt = df[[prompt_column]]
  completion = np.array(df[completion_column])

  save_pyfunc_model(deployment, model_path, prompt, completion, scoring_parameters)
  
  model = mlflow.pyfunc.load_model(model_path)
  predicted = model.predict(prompt)
  deployment_data = load_yaml(deployment)
  deployment_id = deployment_data['id']
  model_id = deployment_data['model']
  metrics = calculate_metrics(completion, predicted)
  metrics = {**metrics, 'deployment_id': deployment_id, 'model_id': model_id}
  return metrics


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--deployment", default="data/5deployment")
  parser.add_argument("--model", default="data/6model/")
  parser.add_argument("--test_data", default="data/1raw/yelp_mini.csv")
  parser.add_argument("--prompt_column", default="text")
  parser.add_argument("--completion_column", default="stars")
  parser.add_argument("--metrics", default="data/6stats/metrics.yaml")

  parser.add_argument("--max_tokens", type=int, default=1)
  parser.add_argument("--temperature", type=float, default=0)
  parser.add_argument("--top_p", type=float, default=0)
  parser.add_argument("--frequency_penalty", type=float, default=0)

  args = parser.parse_args()
  test_data = pd.DataFrame(pd.read_csv(args.test_data))

  metrics = save_and_score(deployment=args.deployment + "/MLArtifact.yaml", 
                            model_path=args.model, 
                            test_data=args.test_data, 
                            prompt_column=args.prompt_column,
                            completion_column=args.completion_column, 
                            scoring_parameters={"max_tokens": args.max_tokens,
                                                "temperature": args.temperature,
                                                "top_p": args.top_p,
                                                "frequency_penalty": args.frequency_penalty})

  metrics = {**metrics, 'max_tokens': args.max_tokens, 'temperature': args.temperature, 'top_p': args.top_p, 'frequency_penalty': args.frequency_penalty}
  print(metrics)
  save_yaml(metrics, args.metrics)
  mlflow.log_params({ "deployment": metrics["deployment_id"],
                      "model": metrics["model_id"],
                      "max_tokens": metrics["max_tokens"],
                      "temperature": metrics["temperature"],
                      "top_p": metrics["top_p"],
                      "frequency_penalty": metrics["frequency_penalty"]})
  mlflow.log_metrics({"test_f1": metrics['test_f1'],
                      "test_accuracy": metrics['test_accuracy'],
                      "test_precision": metrics['test_precision'],
                      "test_recall": metrics['test_recall']})

