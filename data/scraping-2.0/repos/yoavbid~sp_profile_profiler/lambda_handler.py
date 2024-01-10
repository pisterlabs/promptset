import os
import openai.error
import openai
from pathlib import Path

from profile_profiler import get_summary
from jt_sql_alt import execute_query, get_conn

# params
PROMPT_PATH = "prompt.txt"

  
def run_profile(event, context):
  # set API key in two ways to support both local and remote execution
  openai.api_key = os.environ['OPENAI_API_KEY']
  
  profile_id = event['queryStringParameters']['profile_id']
  
  sql_conn = get_conn()
  
  summary = get_summary(profile_id, Path('summarized_log.txt'), 
                                                      Path('events.txt'), 'dlc_names_dict.json', Path('.'), PROMPT_PATH,
                                                      sql_conn)
    
  body = {
        "summary": summary
    }

  response = {"statusCode": 200, "body": json.dumps(body)}

  return response
