from fastapi import APIRouter
from langchain.llms import OpenAI
import requests
import json
import time
import os
# it doesnt work other  (with env file)way on my side 
import openai
openai.api_key = os.getenv('OPENAI_API_KEY')

from app.db import db


router = APIRouter()

@router.get("/")
async def example():
    return {"it": "works"}

@router.get("/init")
async def init():
    await download_board_task()
    await download_user()
    await download_boards()
    await format_data()
    await import_data()

    return {"status": "ok"}

@router.get("/format")
async def format_data():
    with open('data/tasksAndValues.json') as json_file:
        data = json.load(json_file)

    times_formatted = []

    tasks_formatted = []
    for board in data['data']['boards']:
        for item in board['items']:
            tasks_formatted.append({'task_id': item['id'],
                                    'task_name': item['name'],
                                    'created_at': item['created_at'],
                                    'board_id': board['id']})

            for column in item['column_values']:
                if column['type'] == 'duration':
                    if column['value']:
                        values = json.loads(column['value'])

                        for time_val in values['additional_value']:
                            tmp_val = time_val
                            tmp_val['board_id'] = board['id']
                            tmp_val['task_id'] = item['id']
                            times_formatted.append(tmp_val)

    with open("data/tasks_formatted.json", "w") as new_file:
        new_file.write('\n'.join([json.dumps(task) for task in tasks_formatted]))

    with open("data/times_formatted.json", "w") as new_file:
        new_file.write('\n'.join([json.dumps(time) for time in times_formatted]))

    with open('data/users.json') as json_file:
        data = json.load(json_file)

    with open("data/users_formatted.json", "w") as new_file:
        new_file.write('\n'.join([json.dumps(user) for user in data['data']['users']]))

    with open('data/boards.json') as json_file:
        data = json.load(json_file)

    with open("data/boards_formatted.json", "w") as new_file:
        new_file.write('\n'.join([json.dumps(board) for board in data['data']['boards'] if board['type'] == 'board']))

@router.get("/import")
async def import_data():
    f = open("database.sql", "r")
    query = f.read()
    f.close()

    sqlCommands = query.split(';')
    for command in sqlCommands:
        await db.execute(command)

    return {"status": "ok"}

@router.get("/download-board-task")
async def download_board_task():
    # Base of your GraphQL query
    query_template = """
    query {{
        boards(limit: 1, page: {page}) {{
            id
            items {{
                name
                id
                state
                created_at
                updated_at
                group {{
                    id
                }}
                column_values {{
                    text
                    description
                    type
                    id
                    value
                }}
            }}
        }}
    }}
    """

    def download_and_save_data(api_key, query, filename):
        url = "https://api.monday.com/v2"
        headers = {
            "Content-Type": "application/json",
            "Authorization": api_key
        }

        page = 1
        has_more_items = True
        all_boards = []

        while has_more_items:
            query = query_template.format(page=page)
            page_query = f'{query}'
            response = requests.post(url, headers=headers, json={"query": page_query})
            # Parse the response
            data = json.loads(response.text)

            if 'status_code' in data and data['status_code'] == 429:
                print("Throttling limit reached. Waiting for a minute...")
                time.sleep(60)
            else:
                if response.status_code == 200:
                    print(response.text)
                    boards = data['data']['boards']
                    if not boards:
                        # If it's empty, break the loop
                        break
                    all_boards.extend(data['data']['boards'])
                    page += 1
                else:
                    print("Error:", response.status_code)
                    break

        new_dict = {
            "data": {
                "boards": all_boards
            }
        }

        with open(filename, "w") as file:
            json.dump(new_dict, file, indent=4)

        print("Data saved successfully.")
    query = query_template.format(page=1)
    filename = "data/tasksAndValues.json"
    download_and_save_data(os.environ.get('MONDAY_API_KEY'), query, filename)

    return {"it": "works"}

@router.get("/download-user")
async def download_user():
    def download_and_save_data(api_key, query, filename):
        url = "https://api.monday.com/v2"
        headers = {
            "Content-Type": "application/json",
            "Authorization": api_key
        }

        response = requests.post(url, headers=headers, json={"query": query})

        if response.status_code == 200:
            data = response.json()
            with open(filename, "w") as file:
                json.dump(data, file, indent=4)
            print("Data saved successfully.")
        else:
            print("Error:", response.status_code)

    query = '''
    {
        users {
            name
            email
            id
            birthday
            is_admin
            is_guest
            is_pending
            is_verified
            time_zone_identifier
        }
    }
    '''
    filename = "data/users.json"

    download_and_save_data(os.environ.get('MONDAY_API_KEY'), query, filename)

    return {"it": "works"}

@router.get("/download-boards")
async def download_boards():
    def download_and_save_data(api_key, query, filename):
        url = "https://api.monday.com/v2"
        headers = {
            "Content-Type": "application/json",
            "Authorization": api_key
        }

        response = requests.post(url, headers=headers, json={"query": query})

        if response.status_code == 200:
            data = response.json()
            with open(filename, "w") as file:
                json.dump(data, file, indent=4)
            print("Data saved successfully.")
        else:
            print("Error:", response.status_code)

    query = '''
    query {
      boards {
        name
        id
        type
        creator {
          id
        }
      }
    }
    '''
    filename = "data/boards.json"

    download_and_save_data(os.environ.get('MONDAY_API_KEY'), query, filename)

    return {"it": "works"}
@router.get("/talkWithAI")
async def talk_with_AI():
    return {"Answer": "Now I am become Death, the destroyer of worlds "}

@router.get("/kowalskiAnalysis")
async def talk_with_AI2(monday_location: str = "Hakaton"):

    # print(await mistake1())

    async def get_problems(monday_location, location_type="Board"):
        #
        # get data from DB 
        # get users wit
        #
        param1 = await mistake1()
        param2 = await mistake2()
        param3 = await mistake3()
        forecast=[]

        if(param1):


            usernames = [entry["username"] for entry in param1]

            usernames_string = "\n".join(usernames)

            tasks = [entry["task_name"] for entry in param1]

            tasks_string = "\n".join(tasks)

            forecast.append( f'Users count time multiple timers at the same time , please show this user everytime{usernames_string}')
        if(param2):
            usernames = [entry["ended_username"] for entry in param2]

            usernames_string = "\n".join(usernames)

            forecast.append(f"List of users who count time in multiple timers, please show this usernames everytime {usernames_string}")
        if(param3):
            usernames = [entry["started_user_email"] for entry in param3]

            usernames_string = "\n".join(usernames)

            forecast.append(f"User stop time counter of another user.{usernames_string}")

        # end of DB quering

        """Get the current weather in a given location"""
        issues_info = {
            "clickup_location": monday_location,
            "problems_detected": len(forecast),
            "location_type": location_type,
            "forecast": forecast,
        }
        return json.dumps(issues_info)
    async def run_conversation():
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=[{"role": "user", "content": "What is wrong in my data on board "+monday_location+"?"}],
            functions=[
                {
                    "name": "get_problems",
                    "description": "Identify problems in users data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "monday_location": {
                                "type": "string",
                                "description": "Monday item location name",
                            },
                            "board_type": {"type": "string", "enum": ["board", "document", "List"]},
                        },
                        "required": ["monday_location"],
                    },
                }
            ],
            function_call="auto",
        )
        message = response["choices"][0]["message"]
        if message.get("function_call"):
            function_name = message["function_call"]["name"]
        function_response = await get_problems(
                monday_location=message.get("monday_location"),
                location_type=message.get("location_type"),
            )
        second_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[
                    {"role": "user", "content": "What is wrong in my data on Board "+monday_location+" ?" },
                    message,
                    {
                        "role": "function",
                        "name": function_name,
                        "content": function_response,
                    },
                ],
            )
        return second_response
    answer = await run_conversation()
 
    return {"Kowalski": answer["choices"][0]["message"]["content"]}

@router.get("/wrong/3")
async def mistake3():
    query = """
        SELECT tt.task_id, tt.board_id, tt.started_at, tt.ended_at,
               started_user.username AS started_username, started_user.email AS started_user_email,started_user.id AS started_user_id,
               ended_user.username AS ended_username,ended_user.id AS ended_id, ended_user.email AS ended_user_email, tasks_data.task_name,
               boards_data.board_name
        FROM monday_src.time_tracking tt
        LEFT JOIN monday_src.users started_user ON tt.started_user_id = started_user.id
        LEFT JOIN monday_src.users ended_user ON tt.ended_user_id = ended_user.id
        LEFT JOIN monday_src.tasks tasks_data ON tt.task_id = tasks_data.id
        LEFT JOIN monday_src.boards boards_data ON tt.board_id = boards_data.id
        WHERE tt.started_user_id != tt.ended_user_id;
    """

    rows = await db.fetch_all(query=query)

    return rows

@router.get("/wrong/1")
async def mistake1():
    query = """
WITH user_times AS (
	SELECT t.task_id,
			ta.task_name,
			t.started_user_id AS user_id,
			u1.username,
			started_at,
			ended_at,
			lag(started_at,1) OVER(PARTITION BY t.started_user_id ORDER BY started_at) AS start2,
			lag(ended_at,1) OVER(PARTITION BY t.started_user_id ORDER BY started_at) AS end2
	FROM monday_src.time_tracking t
		LEFT JOIN monday_src.users u1
		ON t.started_user_id = u1.id
		LEFT JOIN monday_src.tasks ta
		ON ta.id = t.task_id
		)

SELECT 	user_id,
		username,
		task_id,
		task_name,
		started_at,
		ended_at

FROM user_times
WHERE (started_at, ended_at)  OVERLAPS (COALESCE(start2, '1900-01-01'),COALESCE(end2, '1900-01-01'))
    """

    rows = await db.fetch_all(query=query)

    return rows

@router.get("/wrong/2")
async def mistake2():
    query = """
SELECT task_id, started_user_id, ended_user_id, started_user.username as starter_username,
ended_user.username as ended_username,
tasks_data.task_name as task_name, ended_at - started_at  as interval
FROM monday_src.time_tracking tt
 LEFT JOIN monday_src.users started_user ON tt.started_user_id = started_user.id
  LEFT JOIN monday_src.users ended_user ON tt.ended_user_id = ended_user.id
    LEFT JOIN monday_src.tasks tasks_data ON tt.task_id = tasks_data.id

WHERE ended_at - started_at > interval '8 hours';
    """

    rows = await db.fetch_all(query=query)

    return rows
