
import os
from datasets import Dataset
from utils.utils import load_json
from langchain.chat_models import ChatOpenAI
from sql_agents.zero_shot import ZeroShotAgent
from config import config, api_key
import wandb

QUESTIONS_PATH = os.path.abspath(
    os.path.join(os.path.dirname( __file__ ), '../data/questions.json'))

# If you don't want your script to sync to the cloud
os.environ["WANDB_MODE"] = "offline"

def main():
    wandb.init(
        project=config.project,
        config=config,
        name= config.current_experiment,
        entity=config.entity
    )

    wandb.define_metric("gold_sql_execution_time", summary="last")
    wandb.define_metric("gold_sql_execution_time", summary="mean")
    
    llm = ChatOpenAI(
        openai_api_key=api_key, 
        model_name=config.llm_settings.model,
        temperature=config.llm_settings.temperature,
        request_timeout=config.llm_settings.request_timeout
    )

    questions = load_json(QUESTIONS_PATH)
    questions = [question for question in questions if question['db_id'] in config.domains]
    #questions = [question for question in questions if question['difficulty'] in config.difficulties]
    
    data_loader = Dataset()    
    for i, row in enumerate(questions):        
        golden_sql = row['SQL']
        db_id = row['db_id']
        data_loader.execute_query(golden_sql, db_id)

    wandb.finish()



if __name__ == "__main__":
    main()