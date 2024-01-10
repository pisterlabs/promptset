# set up openai api
import openai, os
import pandas as pd
from groundedness import groundedness
from patch import log_json_artifact
import mlflow, json, tempfile

openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.environ["OPENAI_API_BASE"]
openai.api_key = os.environ["OPENAI_API_KEY"]

from mlflow.tracking import MlflowClient
mlflow_client = MlflowClient()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversation_contexts", type=str, default="./data/amldocs/scores/validation_scores-10.json")
    parser.add_argument("--meta_prompt", type=str, default="./data/amldocs/prompts/groundedness_default.md")
    parser.add_argument("--no-log", action='store_true')
    parser.add_argument("--scores", type=str, default="./data/amldocs/scores/groundedness_scores-10.json")
    args = parser.parse_args()

    verbose = not args.no_log

    if verbose:
        # mlflow.start_run()
        mlflow.log_param("conversation_contexts", args.conversation_contexts)
        mlflow.log_param("meta_prompt", args.meta_prompt)
        mlflow.log_param("scores", args.scores)

    # load conversation_contexts json
    with open(args.conversation_contexts, "r") as f:
        conversation_contexts = json.load(f)
    
    scores = []

    with open(args.meta_prompt, "r") as f:
        meta_prompt = f.read()
    
    for i, context in enumerate(conversation_contexts):
        # Start a child run
        with mlflow.start_run(run_name=f"Groundedness Run #{i}", nested=True) as child_run:
            # Log some metrics and parameters for the child run
            log_json_artifact(context, "conversation_context.json")
            mlflow.log_param("meta_prompt", args.meta_prompt)

            result = groundedness(conversation_context=context, meta_prompt=meta_prompt, verbose=verbose)

            if verbose:
                log_json_artifact(result, "result.json")
            
            mlflow.log_metric("rating_out_of_10", result["rating_out_of_10"])
            
            print(json.dumps(result, indent=4))


            scores.append(result)
    
            # save scores to --scores output json file
            with open(args.scores, "w") as f:
                json.dump(scores, f, indent=4)
    
    ratings = [score["rating_out_of_10"] for score in scores]
    mlflow.log_metric("average_rating_out_of_10", sum(ratings)/len(ratings))
    print(f"Average rating: {sum(ratings)/len(ratings)}")
        


