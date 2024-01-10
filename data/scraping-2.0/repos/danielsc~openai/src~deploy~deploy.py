import time 
import openai
import argparse
import os
import yaml

AOAI_DEPLOYMENT_CANCEL_STATUS = "canceled"
AOAI_DEPLOYMENT_FAILED_STATUS = "failed"
AOAI_DEPLOYMENT_TERMINAL_STATUS = ["succeeded", AOAI_DEPLOYMENT_FAILED_STATUS, AOAI_DEPLOYMENT_CANCEL_STATUS]

def AOAI_wait_until_done(status_retrieve_func, status_retrieve_func_args, terminal_status, check_interval=10, time_out=None, patience=20):
    cumulative_waiting_time, status, error_cnt = 0, None, 0
    while True:
        try:
            status = status_retrieve_func(**status_retrieve_func_args)
        except Exception as e:
            error_cnt += 1
            print(f"hitting exception: {e} while retrieving status, will silently ignore and restart retrieving, current error_cnt: {error_cnt}, total error_cnt before raising error: {patience}")
            if error_cnt > patience:
                raise Exception(f"Exceeding maximum patience ({patience}), please check previous error reports")
        if status in terminal_status:
            print(f'job: {status_retrieve_func_args} hitting terminal status, current status: {status}, current cumulative_waiting_time: {cumulative_waiting_time}')
            break
        print(f'job: {status_retrieve_func_args} is not hitting terminal status, current status: {status}, will sleep for {check_interval} seconds, current cumulative_waiting_time: {cumulative_waiting_time}')
        time.sleep(check_interval)
        cumulative_waiting_time += check_interval
        if (time_out is not None) and (cumulative_waiting_time >= time_out):
            raise TimeoutError(f'job: {status_retrieve_func_args} status: {status}, track hit time_out ({time_out}) threshold')

    print(f"job: {status_retrieve_func_args} hitting terminal status: {status}")
    return status

    
class AOAIDeployer:
    def __init__(self, finetune_job_id):
        self.finetune_job_id = finetune_job_id
        self.deployment = None

    def deploy_job(self):
        result = openai.FineTune.retrieve(id=self.finetune_job_id)
        assert result["status"] == 'succeeded', f"Trying to deploy a unsuccessful job: {self.finetune_job_id}, current job status: {result}"
        model = result["fine_tuned_model"]
        self.deployment = openai.Deployment.create(model=model, scale_settings={"scale_type": "standard"})
        print(f"Creating a new deployment with model: {model}, with deployment id: {self.deployment['id']}")
        return self.deployment

    def check_deployment_status(self, check_interval=10, time_out=None):
        def status_retrieve_func(deployment_id):
            return openai.Deployment.retrieve(id=deployment_id)["status"]

        return AOAI_wait_until_done(status_retrieve_func=status_retrieve_func,
                                    status_retrieve_func_args={"deployment_id": self.deployment['id']},
                                    terminal_status=AOAI_DEPLOYMENT_TERMINAL_STATUS,
                                    check_interval=check_interval,
                                    time_out=time_out)

def str2bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        if val.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif val.lower() in ("no", "false", "f", "n", "0"):
            return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

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

def load_yaml(filename):
   with open(filename, encoding='utf-8') as fh:
      file_dict = yaml.load(fh, Loader=yaml.FullLoader)
   return file_dict


def get_finetune_id(path :str):
    finetune = load_yaml(path + "/MLArtifact.yaml")
    return finetune["id"]

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--aoai_endpoint", default="https://aoai.openai.azure.com/")
    parser.add_argument("--api_version", default="2022-06-01-preview")
    parser.add_argument("--fine_tune_metadata", default="./data/4fine_tune")
    parser.add_argument("--deployment_metadata", default="./data/5deployment")

    args = parser.parse_args()

    openai.api_key = load_api_key()
    openai.api_base = args.aoai_endpoint
    openai.api_type = 'azure'  # hard coding for now
    openai.api_version = args.api_version

    finetune_job_id = get_finetune_id(args.fine_tune_metadata)

    print(f"entering AOAI Deployer")
    AOAI_deployer = AOAIDeployer(finetune_job_id=finetune_job_id)
    deployment = AOAI_deployer.deploy_job().to_dict_recursive()
    deployment['endpoint'] = openai.api_base 
    AOAI_deployment_cur_status = AOAI_deployer.check_deployment_status()
    print(f"retrieved AOAI_deployment_cur_status: {AOAI_deployment_cur_status}")
    if AOAI_deployment_cur_status in [AOAI_DEPLOYMENT_CANCEL_STATUS, AOAI_DEPLOYMENT_FAILED_STATUS]:
        raise Exception(f"the deployment is {AOAI_deployment_cur_status} in server side, job id: {deployment['id']}, terminal status:\n{openai.Deployment.retrieve(id=deployment['id'])}")
    curr_deployment_id = deployment['id']
    print(f"curr_deployment_id: {curr_deployment_id}")
    with open(args.deployment_metadata + "/MLArtifact.yaml", "w") as f:
        yaml.safe_dump(deployment, f)

    print(deployment)


if __name__ == "__main__":
    main()
