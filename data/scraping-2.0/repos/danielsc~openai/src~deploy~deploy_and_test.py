import openai
import argparse
import yaml, os
import pandas as pd
from mlflow.tracking import MlflowClient
from deploy import load_api_key, AOAI_DEPLOYMENT_TERMINAL_STATUS, AOAI_DEPLOYMENT_CANCEL_STATUS, AOAI_DEPLOYMENT_FAILED_STATUS, AOAI_wait_until_done
from save_model import save_and_score
import tempfile

class AOAIDeployer:
    def __init__(self, fine_tuned_model):
        self.fine_tuned_model = fine_tuned_model
        self.deployment = None

    def deploy_job(self):
        self.deployment = openai.Deployment.create(model=self.fine_tuned_model, scale_settings={"scale_type": "standard"}).to_dict_recursive()
        print(f"Creating a new deployment with model: {self.fine_tuned_model}, with deployment id: {self.deployment['id']}")
        return self.deployment

    def retrieve_deployment(self):
        return openai.Deployment.retrieve(id=self.deployment['id']).to_dict_recursive()

    def check_deployment_status(self, check_interval=10, time_out=None):
        def status_retrieve_func(deployment_id):
            return self.retrieve_deployment()["status"]

        return AOAI_wait_until_done(status_retrieve_func=status_retrieve_func,
                                    status_retrieve_func_args={"deployment_id": self.deployment['id']},
                                    terminal_status=AOAI_DEPLOYMENT_TERMINAL_STATUS,
                                    check_interval=check_interval,
                                    time_out=time_out)
    def delete_deployment(self):
        print(f"deleting deployement: {self.deployment['id']}")

        openai.Deployment.delete(sid=self.deployment['id'])

def get_finetunes_from_sweep(sweep_run_id):

    client = MlflowClient()
    child_runs = []
    sweep = client.get_run(sweep_run_id)
    for k, _ in sweep.data.tags.items():
        if k.startswith(sweep_run_id):
            print(k)
            run = client.get_run(k)
            child_runs.append({**run.data.metrics, **run.data.params, "name":k})
    return child_runs

def deploy(fine_tuned_model):
    print(f"entering AOAI Deployer")
    AOAI_deployer = AOAIDeployer(fine_tuned_model=fine_tuned_model)
    deployment = AOAI_deployer.deploy_job()
    deployment['endpoint'] = openai.api_base 
    AOAI_deployment_cur_status = AOAI_deployer.check_deployment_status()
    print(f"retrieved AOAI_deployment_cur_status: {AOAI_deployment_cur_status}")
    if AOAI_deployment_cur_status in [AOAI_DEPLOYMENT_CANCEL_STATUS, AOAI_DEPLOYMENT_FAILED_STATUS]:
        raise Exception(f"the deployment is {AOAI_deployment_cur_status} in server side, job id: {deployment['id']}, terminal status:\n{openai.Deployment.retrieve(id=deployment['id'])}")
    curr_deployment_id = deployment['id']
    print(f"curr_deployment_id: {curr_deployment_id}")

    return AOAI_deployer

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--aoai_endpoint", default="https://aoai.openai.azure.com/")
    parser.add_argument("--api_version", default="2022-06-01-preview")
    parser.add_argument("--hyperdrive_run", default="HD_94013985-1f06-47cf-b41e-0c5bf5e5f6e7")
    parser.add_argument("--test_data", default="./data/1raw/yelp_test.csv")
    parser.add_argument("--prompt_column", default="text")
    parser.add_argument("--completion_column", default="stars")
    parser.add_argument("--stats", default="./data/6stats/model_stats.csv")

    args = parser.parse_args()

    openai.api_key = load_api_key()
    openai.api_base = args.aoai_endpoint
    openai.api_type = 'azure'  # hard coding for now
    openai.api_version = args.api_version
    
    fine_tune_jobs = get_finetunes_from_sweep(args.hyperdrive_run)
    # get tmp folder
    basefolder = tempfile.mkdtemp()
    artifact_path = basefolder + "/MLArtifact.yaml"
    model_path = basefolder + "/model"
    os.makedirs(model_path, exist_ok=True)   
    metics_total = []

    for job in fine_tune_jobs:
        print("working on job: ", job)
        deployer = None
        # check if the model is already deployed
        fine_tuned_model = job["fine_tuned_model"]
        deployments = openai.Deployment.list().data
        deployment = None
        for deployment_checked in deployments:
            if deployment_checked["model"] == fine_tuned_model:
                print(f"model {fine_tuned_model} is already deployed with deployment id: {deployment_checked['id']}")
                deployment = deployment_checked.to_dict_recursive()
        
        # if not deployed, deploy
        if deployment is None:    
            deployer = deploy(job["fine_tuned_model"])
            deployment = deployer.retrieve_deployment()

        print(deployment)

        # make sure endpoint is set
        deployment['endpoint'] = openai.api_base
        # save deployment
        with open(artifact_path, "w") as f:
            f.write(yaml.dump(deployment))

        print(f"scoring {args.test_data}")
        metrics = save_and_score(deployment=artifact_path, 
                        model_path=model_path, 
                        test_data=args.test_data, 
                        prompt_column=args.prompt_column,
                        completion_column=args.completion_column)
        print({**metrics, **job})
        metics_total.append({**metrics, **job})
        # delete deployment
        if deployer is not None:
            print(f"deleting deployment: {deployment['id']}")
            deployer.delete_deployment()

    df=pd.DataFrame(metics_total)
    df.to_csv(args.stats, index=False)


if __name__ == "__main__":
    main()
