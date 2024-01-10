import requests
import json
import config
import openai
import tqdm
import html
from datetime import datetime
   
# Function to fetch open user stories that haven't been "Instructor Reviewed"
def fetch_open_user_stories():
    url = f"https://dev.azure.com/"
    url += f"{config.DEVOPS_ORG}/" 
    url += f"_apis/wit/wiql"
    
    params = {"api-version":config.API_VERSION}
    
    payload = json.dumps({
          "query": """SELECT [System.Id] FROM workitems WHERE [System.WorkItemType] = 'User Story' 
          AND [Custom.AIReviewed] = 'False'
          AND [System.CreatedDate] >= @Today - 60
          
          """
    })
    # Get the list of work items
    response = requests.post(url, headers=config.PRIMARY_HEADERS, data=payload, params=params)
    if response.status_code != 200:
        print(f"Failed to fetch user stories. Error: {response.json()}")
        return []
    work_items = response.json()["workItems"]
    print(f'Found {len(work_items)} user stories to review.')    
    return work_items
    
# Function to fetch the details of a work item
# Function to fetch the details of a work item by its ID
def fetch_work_item_details(work_item_id, fields):

    url = f"https://dev.azure.com/{config.DEVOPS_ORG}/_apis/wit/workitems/{work_item_id}"
    params = {"api-version":config.API_VERSION,
              "fields":",".join(fields)
              }
    response = requests.get(url, headers=config.PRIMARY_HEADERS,params=params)
    if response.status_code != 200:
        print(f"Failed to fetch work item details. Error: {response.json()}")
        return None
    return response.json()
    
class WorkItem(object):
    
    def __init__(self, id, description, acceptance_criteria, story_points=0):
        self.id = id
        self.description = description
        self.acceptance_criteria = acceptance_criteria
        self.feedback = None
        self.story_points = story_points
    

def get_work_items_with_details():
    work_items = []

    # Get the list of work items
    for wid in fetch_open_user_stories():
        # Get the details of the work item
        id = wid.get("id")
        work_items.append(get_work_item(id))
    
    return work_items

def get_work_item(id):
    '''
    Gather the work item details and create a new work_item object
    
    : returns : WorkItem
    '''
    fields = ["System.Description",
              "Microsoft.VSTS.Common.AcceptanceCriteria",
              "System.TeamProject",
              "Microsoft.VSTS.Scheduling.StoryPoints"
    ]
    wid_details = fetch_work_item_details(id,fields)
        
        # Gather the fields from the response
    _fields = wid_details.get('fields', {})
    description = _fields.get("System.Description","No description")
    acceptance_criteria = _fields.get("Microsoft.VSTS.Common.AcceptanceCriteria","No acceptance criteria")
    story_points = _fields.get("Microsoft.VSTS.Scheduling.StoryPoints",0)
    work_item = WorkItem(id, description, acceptance_criteria,story_points=story_points)
    return work_item


def get_chat_feedback(work_item):
    '''
    Go to ChatGPT and ask for feedback on the user story.
    
    :returns: response from ChatGPT
    '''
    conversation= [
               {"role":"system", "content":config.STORY_BACKGROUND},
               {"role":"user", "content":work_item.description},
               {"role":"user", "content":work_item.acceptance_criteria}
    ]
    
    response = openai.ChatCompletion.create(model='gpt-3.5-turbo',messages=conversation)
    message = response['choices'][0]['message']['content']
    return message

def update_work_item(work_item_id, comment):
    url = f"https://dev.azure.com/{config.DEVOPS_ORG}/_apis/wit/workitems/{work_item_id}"

    headers = {
        'Content-Type': 'application/json-patch+json',
        'Authorization': f'Basic {config.DEVOPS_AUTH_TOKEN}',
    }
    params = {"api-version":config.API_VERSION}

    # Prepare the payload
    payload = [
        {
            "op": "add",
            "path": "/fields/System.History",
            "value": comment
        },
        {   "op": "add",
            "path": "/fields/Custom.AIReviewed",
            "value": "True"
        }
    ]
    response = requests.request("PATCH", url, headers=headers, data=json.dumps(payload), params=params)

    if response.status_code == 200:
        print("Work item updated successfully.")
    else:
        print(f"Failed to update work item. Status code: {response.status_code}, Error: {response.text}")

if __name__ == "__main__":
    openai.api_key = config.OPEN_API_KEY
    work_items = get_work_items_with_details()
    
    for work_item in tqdm.tqdm(work_items):
        print(f'Processing work item {work_item.id}')
        feedback = get_chat_feedback(work_item)
        fixed_feedback = html.escape(feedback).replace('\n', '<br>')
        work_item.feedback = fixed_feedback
        update_work_item(work_item.id, fixed_feedback)
        
    now = datetime.now().strftime('%m%d%H%M')
    json.dump([wi.__dict__ for wi in work_items], open(f"work_items{now}.json","w"))
        