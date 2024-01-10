
import os
# from datasets import get_dataset
from langchain.chat_models import ChatOpenAI

import sys
sys.path.append('/Users/fredrik/code/project/Text-to-SQL-Generation/src')
from config import load_config
import wandb
import langchain 
langchain.verbose = False


# If you don't want your script to sync to the cloud
os.environ["WANDB_MODE"] = "offline"



def main():
    config = load_config("/Users/fredrik/code/project/Text-to-SQL-Generation/config/c3_config.yaml")

## Generic Agent Code ####
    wandb.init(
        project=config.project,
        config=config,
        name= config.current_experiment,
        entity=config.entity,
        id="1338"
    )



if __name__ == "__main__":
    main()



    # artifact = wandb.Artifact('query_results', type='dataset')
    # table = wandb.Table(columns=["Question", "Gold Query", "Predicted Query", "Success"])    

    # wandb.define_metric("predicted_sql_execution_time", summary="mean")
    # wandb.define_metric("gold_sql_execution_time", summary="mean")

    # llm = ChatOpenAI(
    #     openai_api_key=api_key, 
    #     model_name=config.llm_settings.model,
    #     temperature=config.llm_settings.temperature,
    #     request_timeout=config.llm_settings.request_timeout
    # )

    # dataset = get_dataset(config.dataset)    
    # c3_agent = C3Agent(llm)

    # # wandb.config['prompt'] = c3_agent.prompt_template
    
    # no_data_points = dataset.get_number_of_data_points()
    # score = 0
    # accuracy = 0
    # for i in range(no_data_points):

    #     ### Generic main method starts
    #     # if i == 5 or i == 26 or i == 27:
    #     #     continue

    #     data_point = dataset.get_data_point(i)
    #     evidence = data_point['evidence']
    #     golden_sql = data_point['SQL']
    #     db_id = data_point['db_id']            
    #     question = data_point['question']

    #     sql_schema = dataset.get_schema_and_sample_data(db_id)
    #     predicted_sql = c3_agent.generate_query(sql_schema, question, evidence)        
    #     success = dataset.execute_queries_and_match_data(predicted_sql, golden_sql, db_id)

    #     score += success
    #     accuracy = score / (i + 1)

    #     table.add_data(question, golden_sql, predicted_sql, success)
    #     wandb.log({
    #         "accuracy": accuracy,
    #         "total_tokens": c3_agent.total_tokens,
    #         "prompt_tokens": c3_agent.prompt_tokens,
    #         "completion_tokens": c3_agent.completion_tokens,
    #         "total_cost": c3_agent.total_cost,
    #         "openAPI_call_execution_time": c3_agent.last_call_execution_time,
    #         "predicted_sql_execution_time": dataset.last_predicted_execution_time,
    #         "gold_sql_execution_time": dataset.last_gold_execution_time
    #     }, step=i+1)
    
    #     print("Percentage done: ", round(i / no_data_points * 100, 2), "% Domain: ", 
    #           db_id, " Success: ", success, " Accuracy: ", accuracy)
            

    # wandb.run.summary['number_of_questions']                = no_data_points
    # wandb.run.summary["accuracy"]                           = accuracy
    # wandb.run.summary["total_tokens"]                       = c3_agent.total_tokens
    # wandb.run.summary["prompt_tokens"]                      = c3_agent.prompt_tokens
    # wandb.run.summary["completion_tokens"]                  = c3_agent.completion_tokens
    # wandb.run.summary["total_cost"]                         = c3_agent.total_cost
    # wandb.run.summary['total_predicted_execution_time']     = dataset.total_predicted_execution_time
    # wandb.run.summary['total_gold_execution_time']          = dataset.total_gold_execution_time
    # wandb.run.summary['total_openAPI_execution_time']       = c3_agent.total_call_execution_time

    # artifact.add(table, "query_results")
    # wandb.log_artifact(artifact)

    # artifact_code = wandb.Artifact('code', type='code')
    # artifact_code.add_file("src/agents/c3_agent.py")
    # wandb.log_artifact(artifact_code)

    # wandb.finish()
