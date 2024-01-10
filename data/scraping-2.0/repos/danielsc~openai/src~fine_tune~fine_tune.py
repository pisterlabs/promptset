from time import sleep
import requests, io
import os, argparse
import openai, yaml
import mlflow, time
import pandas as pd

def load_yaml(filename):
   with open(filename, encoding='utf-8') as fh:
      file_dict = yaml.load(fh, Loader=yaml.FullLoader)
   return file_dict

def save_yaml(content, filename):
   with open(filename, encoding='utf-8', mode='w') as fh:
      yaml.dump(content, fh)

def load_api_key(keyname = "openai-key"):
  from azureml.core.run import _OfflineRun
  from azureml.core import Run
  run = Run.get_context()
  if run.__class__ == _OfflineRun:
    print(f"loading key {keyname} from environment variables")
    secret_value = os.getenv("OPENAI_API_KEY")
  else:
    print(f"loading key {keyname} from keyvault")
    secret_value = run.get_secret(name=keyname)

  if not secret_value is None:
    return secret_value
  else:
    raise Exception("No OPENAI_API_KEY found in environment variables or keyvault")


def submit_fine_tune(url, api_key, training_file, validation_file, args):
  headers={
    "api-key": api_key,
    "Content-Type": "application/json"
  }
  payload = {
    "training_file": training_file,
    "validation_file": validation_file,
    "model": args.model,
    "hyperparams": {
      "n_epochs": args.n_epochs,
      "batch_size": args.batch_size,
      "learning_rate_multiplier": args.learning_rate_multiplier,
      "compute_classification_metrics": args.compute_classification_metrics,
      "classification_n_classes": args.classification_n_classes,
    },
  }
  
  params = {
    "api-version": args.api_version
  }

  print("Submitting fine tune")
  print("url: ", url)
  print("headers: ", {k:v for (k,v) in headers.items() if k!='api-key'})
  print("params: ", params)
  print("payload: ", payload)

  r = requests.post(url,
    headers = headers,
    params = params,
    json = payload
  )

  data = r.json()
  print("response:")
  print(data)
  save_yaml(data, args.fine_tune_metadata + "/MLArtifact.yaml")
  return data['id']

def get_fine_tune(url, api_key, fine_tune_id, args):
  url = url + "/" + fine_tune_id
  headers={
    "api-key": api_key,
    "Content-Type": "application/json"
  }
  params = {
    "api-version": args.api_version
  }

  r = requests.get(url,
    headers = headers,
    params = params
  )

  data = r.json()
  return data

def get_fine_tune_and_log(url, api_key, fine_tune_id, args):
  data = get_fine_tune(url, api_key, fine_tune_id, args)
  print(data)
  save_yaml(data, args.fine_tune_metadata + "/MLArtifact.yaml")
  return data

def log_metrics(fine_tune, metric_logged):
  if "result_files" in fine_tune:
    result_file = fine_tune["result_files"][0]
    response = openai.File.download(id=result_file['id'])
    f = io.BytesIO(response)
    df = pd.read_csv(f)

    # step,elapsed_tokens,elapsed_examples,training_loss,training_sequence_accuracy,training_token_accuracy,validation_loss,validation_sequence_accuracy,validation_token_accuracy
    for col in df.columns[1:]: 
      values = df[['step',col]].dropna() 
      # drop all rows with -1 as a value
      values = values[values[col] != -1]
      for i, row in values.iterrows():
        if i >= metric_logged:
          mlflow.log_metric(key=col, value=row[col], step=int(row.step)) 
          print(dict(key=col, value=row[col], step=int(row.step)))
    metric_logged = len(df)
  return metric_logged

def wait_for_job_completion(url, api_key, fine_tune_id, args):
  print("waiting for job completion")

  metric_logged = 0
  event_printed = 0
  while True:
    fine_tune = openai.FineTune.retrieve(id=fine_tune_id)

    for event in fine_tune['events'][event_printed:]:
      print()
      print(event)
      event_printed += 1
    
    metric_logged = log_metrics(fine_tune, metric_logged)

    print(".", end="", flush=True)

    if fine_tune["status"] in ["succeeded", "failed", "canceled"]:
      break

    time.sleep(10)

  return fine_tune

def stream_fine_tune_status(url, api_key, fine_tune_id, args):
  print("streaming fine tune status")
  url = url + "/" + fine_tune_id + "/events"
  headers={
    "api-key": api_key,
    "Content-Type": "application/json"
  }
  params = {
    "api-version": args.api_version,
    "stream": True
  }

  print("url: ", url)
  print("fine_tune_id: ", fine_tune_id)
  print("headers: ", headers)
  print("params: ", params)

  r = requests.get(url,
    headers = headers,
    params = params,
    stream=True
  )

  for data in r.iter_lines():
    print(data)

  return 

def log_result_file(url, api_key, fine_tune, args):
  headers={
    "api-key": api_key,
    "Content-Type": "application/json"
  }
  params = {
    "api-version": args.api_version
  }

  if 'fine_tuned_model' in fine_tune:
    mlflow.log_param("fine_tuned_model", fine_tune['fine_tuned_model'])

  if not 'result_files' in fine_tune:
    raise Exception("no result_files in fine_tune: " + fine_tune)

  result_files = fine_tune['result_files']
  fine_tune_results = next((file for file in result_files if file['purpose']=='fine-tune-results'), None)
  
  if fine_tune_results is None:
    raise Exception("no result_files with purpose fine-tune-results in fine_tune: " + fine_tune)
  
  print("getting fine_tune_results")
  r = requests.get(url + "/" + fine_tune_results['id'] + "/content",
    headers=headers,
    params=params
  )
  with open(fine_tune_results['filename'], 'wb') as file:
    file.write(r.content)

  mlflow.log_artifact(fine_tune_results['filename'])

  return



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--training_data", default="./data/3train_uploaded")
  parser.add_argument("--validation_data", default="./data/3validation_uploaded")
  parser.add_argument("--model", default="ada")
  parser.add_argument("--n_epochs", default=1, type=int)
  parser.add_argument("--batch_size", default=4, type=int)
  parser.add_argument("--learning_rate_multiplier", default=0.1, type=float)
  parser.add_argument("--classification_n_classes", default=5, type=int)
  parser.add_argument("--compute_classification_metrics", default=True, type=bool)
  parser.add_argument("--aoai_endpoint", default="https://aoai.openai.azure.com/")
  parser.add_argument("--api_version", default="2022-06-01-preview")
  parser.add_argument("--fine_tune_metadata", default="./data/4fine_tune")
  args = parser.parse_args()

  mlflow.log_params({
    "model": args.model,
    "n_epochs": args.n_epochs,
    "batch_size": args.batch_size,
    "learning_rate_multiplier": args.learning_rate_multiplier,
    "classification_n_classes": args.classification_n_classes,
    "compute_classification_metrics": args.compute_classification_metrics,
    "aoai_endpoint": args.aoai_endpoint,
    "api_version": args.api_version,
  })

  api_key = load_api_key()
  openai.api_version = args.api_version
  openai.api_key = api_key
  openai.api_base = args.aoai_endpoint
  openai.api_type = "azure"
  training_file_metadata = load_yaml(f"{args.training_data}/MLArtifact.yaml")
  validation_file_metadata = load_yaml(f"{args.validation_data}/MLArtifact.yaml")
  url = f"{args.aoai_endpoint}openai/fine-tunes"

  # kick off the fine tune job
  fine_tune_id = submit_fine_tune(url, api_key, training_file_metadata['id'], validation_file_metadata['id'], args)
  mlflow.log_param("fine_tune_id", fine_tune_id)

  # stream the events while the job is running
  # stream_fine_tune_status(url, api_key, fine_tune_id, args)

  # wait for the job to complete
  # fine_tune = wait_for_job_completion(url, api_key, fine_tune_id, args)
  # print("fine_tune ended with status:", fine_tune['status'])
  wait_for_job_completion(url, api_key, fine_tune_id, args)

  # write final data of the fine_tune job to the output 
  fine_tune = get_fine_tune_and_log(url, api_key, fine_tune_id, args)
  print("fine_tune ended with status:", fine_tune['status'])

  # end with error if the job failed
  if fine_tune['status'] in ['failed', 'canceled']:
    print(fine_tune)
    exit(1)

  # log the metrics to mlflow
  file_url = f"{args.aoai_endpoint}openai/files"
  log_result_file(file_url, api_key, fine_tune, args)

  print("done...")

